#!/usr/bin/env python3
"""
a2c_pointenv.py
A2C (Advantage Actor-Critic) en PointEnv discreto (acciones {-1,0,+1}).
Guarda: modelo, convergence.png, A2C_PointEnv_Report.pdf, README.txt
Requerimientos: torch, matplotlib, numpy
Ejecutar: python a2c_pointenv.py
"""
import os, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

# ----------------------------
# Output dir (portable)
OUT_DIR = os.environ.get("OUT_A2C", "./output_a2c_pointenv")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Entorno discreto simple
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

# ----------------------------
# Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh())
        self.policy = nn.Sequential(nn.Linear(hidden, hidden//2), nn.Tanh(), nn.Linear(hidden//2, act_dim))
        self.value  = nn.Sequential(nn.Linear(hidden, hidden//2), nn.Tanh(), nn.Linear(hidden//2, 1))
    def forward(self, x):
        h = self.shared(x)
        logits = self.policy(h)
        value = self.value(h).squeeze(-1)
        return logits, value

# ----------------------------
# Helpers
def select_action(logits):
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    a = dist.sample()
    return a.item(), dist.log_prob(a), dist.entropy()

def discounted_returns(rewards, dones, last_value, gamma=0.99):
    R = last_value
    returns = []
    for r, d in zip(rewards[::-1], dones[::-1]):
        if d:
            R = 0.0
        R = r + gamma * R
        returns.insert(0, R)
    return returns

# ----------------------------
# Hiperparámetros
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
NUM_EPISODES = 400
MAX_STEPS = 50
GAMMA = 0.99
LR = 2.5e-4
VALUE_COEF = 0.5
ENTROPY_COEF = 1e-3
PRINT_EVERY = 25

# ----------------------------
# Setup
env = PointEnvDiscrete(max_steps=MAX_STEPS)
obs_dim = env.obs_shape[0]
act_dim = env.n_actions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ActorCritic(obs_dim, act_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

episode_rewards = []
smoothed = []
alpha_smooth = 0.04

# Training loop
for ep in range(1, NUM_EPISODES + 1):
    obs = env.reset()
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    done = False
    ep_reward = 0.0

    # buffers
    logps = []
    values = []
    rewards = []
    dones = []
    entropies = []

    for step in range(MAX_STEPS):
        logits, value = model(obs_t.unsqueeze(0))
        logits = logits.squeeze(0)
        value = value.squeeze(0)
        action, logp, entropy = select_action(logits)
        next_obs, reward, done, _ = env.step(action)
        ep_reward += reward

        logps.append(logp)
        values.append(value)
        rewards.append(reward)
        dones.append(done)
        entropies.append(entropy)

        obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
        if done:
            break

    if done:
        last_value = 0.0
    else:
        with torch.no_grad():
            _, last_value = model(obs_t.unsqueeze(0))
            last_value = last_value.item()

    returns = discounted_returns(rewards, dones, last_value, GAMMA)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    values = torch.stack(values)
    logps = torch.stack(logps)
    entropies = torch.stack(entropies)
    advantages = returns - values.detach()

    value_loss = (advantages ** 2).mean()
    policy_loss = -(logps * advantages).mean()
    entropy_loss = -entropies.mean()

    loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss

    optimizer.zero_grad()
    loss.backward()
    # optional: torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    episode_rewards.append(ep_reward)
    if len(smoothed) == 0:
        smoothed.append(ep_reward)
    else:
        smoothed.append(smoothed[-1] * (1 - alpha_smooth) + ep_reward * alpha_smooth)

    if (ep % PRINT_EVERY == 0) or (ep == 1):
        avg100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 1 else 0.0
        print(f"Ep {ep}/{NUM_EPISODES} Reward {ep_reward:.3f} Avg100 {avg100:.3f} Loss {loss.item():.4f}")

# Save model and results
model_path = os.path.join(OUT_DIR, "a2c_pointenv_model.pth")
torch.save(model.state_dict(), model_path)

# Plot
episodes = np.arange(1, len(episode_rewards) + 1)
plt.figure(figsize=(9,5))
plt.plot(episodes, episode_rewards, alpha=0.25, label="Reward por episodio")
plt.plot(episodes, smoothed, label=f"Media exponencial α={alpha_smooth}")
plt.xlabel("Episodio"); plt.ylabel("Reward"); plt.title("A2C Convergencia - PointEnv (discreto)")
plt.legend(); plt.grid(True)
plot_path = os.path.join(OUT_DIR, "convergence_a2c.png")
plt.savefig(plot_path, bbox_inches="tight")
plt.close()

# PDF report (3 páginas)
pdf_path = os.path.join(OUT_DIR, "A2C_PointEnv_Report.pdf")
with PdfPages(pdf_path) as pdf:
    # Page 1 - description
    plt.figure(figsize=(8.27, 11.69)); plt.axis("off")
    txt = [
        "A2C aplicado a PointEnv (discreto)",
        "",
        f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Problema: punto 1D que debe llegar a 0.0. Acciones discretas {-1,0,+1}.",
        "Algoritmo: A2C (Actor-Critic).",
        "",
        f"Episodes: {NUM_EPISODES}, max_steps: {MAX_STEPS}",
        "",
        "Se incluyen fragmentos de código y gráfica de convergencia."
    ]
    plt.text(0.02, 0.98, "\n".join(txt), va="top", wrap=True, fontsize=11)
    pdf.savefig(); plt.close()

    # Page 2 - code snippets
    plt.figure(figsize=(8.27, 11.69)); plt.axis("off")
    code = [
        "Fragmentos de código:",
        "",
        "ActorCritic: shared -> policy logits ; value scalar",
        "select_action: Categorical(softmax(logits))",
        "returns = discounted_rewards; advantages = returns - values.detach()",
        "loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss"
    ]
    plt.text(0.02, 0.98, "\n".join(code), va="top", wrap=True, fontsize=10)
    pdf.savefig(); plt.close()

    # Page 3 - plot
    img = plt.imread(plot_path)
    plt.figure(figsize=(8.27, 11.69)); plt.imshow(img); plt.axis("off")
    pdf.savefig(); plt.close()

# README
readme_path = os.path.join(OUT_DIR, "README.txt")
with open(readme_path, "w") as f:
    f.write("A2C PointEnv - Output\n")
    f.write(f"Model: {model_path}\nPlot: {plot_path}\nPDF: {pdf_path}\nEpisodes: {NUM_EPISODES}\n")
print("A2C: entrenamiento finalizado. Archivos en:", OUT_DIR)
