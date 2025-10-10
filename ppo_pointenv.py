#!/usr/bin/env python3
"""
ppo_pointenv.py
PPO (Clipped surrogate) en PointEnv discreto.
Guarda: modelo, convergence.png, PPO_PointEnv_Report.pdf, README.txt
Ejecutar: python ppo_pointenv.py
"""
import os, random
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

OUT_DIR = os.environ.get("OUT_PPO", "./output_ppo_pointenv")
os.makedirs(OUT_DIR, exist_ok=True)

# Same discrete environment as A2C
class PointEnvDiscrete:
    def __init__(self, max_steps=50, force=0.6, noise_scale=0.02):
        self.max_steps = max_steps
        self.force = force
        self.noise_scale = noise_scale
        self.reset()
    def reset(self):
        self.pos = np.random.uniform(-1.5, 1.5)
        self.steps = 0
        self.target = 0.0
        return np.array([self.pos], dtype=np.float32)
    def step(self, action_index):
        act = [-1, 0, 1][action_index]
        noise = np.random.normal(scale=self.noise_scale)
        self.pos = self.pos + act * self.force + noise
        self.steps += 1
        dist = abs(self.pos - self.target)
        reward = -dist
        done = False
        if dist < 0.05:
            reward += 1.0
            done = True
        if self.steps >= self.max_steps:
            done = True
        return np.array([self.pos], dtype=np.float32), float(reward), done, {}
    @property
    def obs_shape(self):
        return (1,)
    @property
    def n_actions(self):
        return 3

# Network: shared base, logits and value
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh())
        self.policy = nn.Sequential(nn.Linear(hidden, hidden//2), nn.Tanh(), nn.Linear(hidden//2, act_dim))
        self.value  = nn.Sequential(nn.Linear(hidden, hidden//2), nn.Tanh(), nn.Linear(hidden//2, 1))
    def forward(self, x):
        h = self.shared(x)
        logits = self.policy(h)
        v = self.value(h).squeeze(-1)
        return logits, v

def sample_action_and_logp(logits):
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    a = dist.sample()
    return a.item(), dist.log_prob(a), dist.entropy()

def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    values = list(values) + [last_value]
    gae = 0.0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step+1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        returns.insert(0, gae + values[step])
    adv = np.array(returns) - np.array(values[:-1])
    return returns, adv

# Hiperparams
SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
NUM_EPOCHS = 400
MAX_STEPS = 50
GAMMA = 0.99; LAM = 0.95
LR = 3e-4
CLIP = 0.2
UPDATE_EPOCHS = 6
BATCH_SIZE = 64
ENTROPY_COEF = 1e-3
VALUE_COEF = 0.5

# Setup
env = PointEnvDiscrete(max_steps=MAX_STEPS)
obs_dim = env.obs_shape[0]; act_dim = env.n_actions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ActorCritic(obs_dim, act_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

episode_rewards = []
smoothed = []; alpha = 0.04

# Training loop - collect trajectories then update
for ep in range(1, NUM_EPOCHS+1):
    # collect one episode traj
    obs = env.reset()
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    done = False
    ep_reward = 0.0

    obs_buf = []; actions_buf = []; logp_buf = []; rewards_buf = []; dones_buf = []; values_buf = []

    for step in range(MAX_STEPS):
        logits, value = model(obs_t.unsqueeze(0))
        logits = logits.squeeze(0); value = value.squeeze(0)
        a, logp, entropy = sample_action_and_logp(logits)
        next_obs, reward, done, _ = env.step(a)
        ep_reward += reward

        obs_buf.append(obs_t.cpu().numpy())
        actions_buf.append(a)
        logp_buf.append(logp.item())
        rewards_buf.append(reward)
        dones_buf.append(done)
        values_buf.append(value.item())

        obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
        if done:
            break

    if done:
        last_value = 0.0
    else:
        with torch.no_grad():
            _, last_value = model(obs_t.unsqueeze(0))
            last_value = last_value.item()

    returns, advantages = compute_gae(rewards_buf, values_buf, dones_buf, last_value, GAMMA, LAM)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
    obs_arr = torch.tensor(np.vstack(obs_buf), dtype=torch.float32, device=device)
    actions_arr = torch.tensor(actions_buf, dtype=torch.long, device=device)
    old_logps = torch.tensor(logp_buf, dtype=torch.float32, device=device)

    # PPO update - multiple epochs over the collected batch
    dataset_size = len(actions_buf)
    for _ in range(UPDATE_EPOCHS):
        # simple mini-batching
        idxs = np.arange(dataset_size)
        np.random.shuffle(idxs)
        for start in range(0, dataset_size, BATCH_SIZE):
            end = start + BATCH_SIZE
            mb_idx = idxs[start:end]
            mb_obs = obs_arr[mb_idx]
            mb_actions = actions_arr[mb_idx]
            mb_oldlogp = old_logps[mb_idx]
            mb_returns = returns[mb_idx]
            mb_adv = advantages[mb_idx]

            logits, vals = model(mb_obs)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            mb_logp = dist.log_prob(mb_actions)
            ratio = torch.exp(mb_logp - mb_oldlogp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (mb_returns - vals).pow(2).mean()
            entropy_loss = -dist.entropy().mean()
            loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    episode_rewards.append(ep_reward)
    if len(smoothed) == 0:
        smoothed.append(ep_reward)
    else:
        smoothed.append(smoothed[-1] * (1 - alpha) + ep_reward * alpha)

    if (ep % 25 == 0) or (ep == 1):
        avg100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 1 else 0.0
        print(f"Ep {ep}/{NUM_EPOCHS} Reward {ep_reward:.3f} Avg100 {avg100:.3f}")

# Save
model_path = os.path.join(OUT_DIR, "ppo_pointenv_model.pth")
torch.save(model.state_dict(), model_path)
# Plot
episodes = np.arange(1, len(episode_rewards)+1)
plt.figure(figsize=(9,5))
plt.plot(episodes, episode_rewards, alpha=0.3, label="Reward por episodio")
plt.plot(episodes, smoothed, label="Media exponencial")
plt.xlabel("Episodio"); plt.ylabel("Reward"); plt.title("PPO Convergencia - PointEnv")
plt.legend(); plt.grid(True)
plot_path = os.path.join(OUT_DIR, "convergence_ppo.png")
plt.savefig(plot_path, bbox_inches="tight"); plt.close()

# PDF
pdf_path = os.path.join(OUT_DIR, "PPO_PointEnv_Report.pdf")
with PdfPages(pdf_path) as pdf:
    plt.figure(figsize=(8.27,11.69)); plt.axis("off")
    txt = ["PPO sobre PointEnv (discreto)", "", f"Fecha: {datetime.now().isoformat()}"]
    plt.text(0.02, 0.98, "\n".join(txt), va="top", wrap=True, fontsize=11); pdf.savefig(); plt.close()
    plt.figure(figsize=(8.27,11.69)); plt.axis("off")
    code = ["Fragmentos: PPO clip, GAE, update_epochs", "", "Clip:", str(CLIP)]
    plt.text(0.02, 0.98, "\n".join(code), va="top", wrap=True, fontsize=10); pdf.savefig(); plt.close()
    img = plt.imread(plot_path); plt.figure(figsize=(8.27,11.69)); plt.imshow(img); plt.axis("off"); pdf.savefig(); plt.close()

readme_path = os.path.join(OUT_DIR, "README.txt")
with open(readme_path, "w") as f:
    f.write("PPO PointEnv - Output\n"); f.write(f"Model: {model_path}\nPlot: {plot_path}\nPDF: {pdf_path}\n")
print("PPO: terminado. Archivos en:", OUT_DIR)
