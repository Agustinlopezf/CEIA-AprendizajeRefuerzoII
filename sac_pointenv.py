#!/usr/bin/env python3
"""
sac_pointenv.py
Soft Actor-Critic en entorno continuo 1D (PointEnvContinuous).
Guarda: modelo (actor + critics), convergence.png (reward por episodio),
SAC_PointEnv_Report.pdf, README.txt
Ejecutar: python sac_pointenv.py
Requerimientos: torch, numpy, matplotlib
"""
import os, random, math
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from collections import deque

OUT_DIR = os.environ.get("OUT_SAC", "./output_sac_pointenv")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Entorno continuo
class PointEnvContinuous:
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
    def step(self, action):
        # action in [-1,1] scalar
        act = float(np.clip(action, -1.0, 1.0))
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
    def action_dim(self):
        return 1

# ----------------------------
# Replay buffer
class ReplayBuffer:
    def __init__(self, maxlen=100000):
        self.maxlen = maxlen
        self.buf = deque(maxlen=maxlen)
    def push(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.vstack(s), np.vstack(a), np.array(r, dtype=np.float32),
                np.vstack(s2), np.array(d, dtype=np.float32))
    def __len__(self):
        return len(self.buf)

# ----------------------------
# Networks
LOG_STD_MIN = -20
LOG_STD_MAX = 2

class GaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=64):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)
    def forward(self, x):
        h = self.fc(x)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mean, std
    def sample(self, x):
        mean, std = self.forward(x)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        logp = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        logp = logp.sum(-1, keepdim=True)
        return action, logp, torch.tanh(mean)

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim + action_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.net(x).squeeze(-1)

# ----------------------------
# Hyperparams
SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
NUM_EPISODES = 400
MAX_STEPS = 50
GAMMA = 0.99
LR = 3e-4
BATCH_SIZE = 64
REPLAY_INIT = 1000
REPLAY_SIZE = 100000
TAU = 0.005
AUTO_ALPHA = True
TARGET_ENTROPY = -1.0  # for 1-d action space
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup
env = PointEnvContinuous(max_steps=MAX_STEPS)
obs_dim = env.obs_shape[0]; action_dim = env.action_dim
buffer = ReplayBuffer(maxlen=REPLAY_SIZE)

actor = GaussianActor(obs_dim, action_dim).to(DEVICE)
q1 = QNetwork(obs_dim, action_dim).to(DEVICE)
q2 = QNetwork(obs_dim, action_dim).to(DEVICE)
q1_target = QNetwork(obs_dim, action_dim).to(DEVICE); q2_target = QNetwork(obs_dim, action_dim).to(DEVICE)
q1_target.load_state_dict(q1.state_dict()); q2_target.load_state_dict(q2.state_dict())

actor_opt = optim.Adam(actor.parameters(), lr=LR)
q1_opt = optim.Adam(q1.parameters(), lr=LR)
q2_opt = optim.Adam(q2.parameters(), lr=LR)

# alpha
if AUTO_ALPHA:
    log_alpha = torch.tensor(0.0, requires_grad=True, device=DEVICE)
    alpha_opt = optim.Adam([log_alpha], lr=LR)
else:
    alpha = 0.2

episode_rewards = []
smoothed = []
alpha_smooth = 0.04

# Interaction and training
total_steps = 0
for ep in range(1, NUM_EPISODES + 1):
    s = env.reset()
    s_t = torch.tensor(s, dtype=torch.float32, device=DEVICE)
    ep_reward = 0.0
    for step in range(MAX_STEPS):
        # sample action from current policy (for exploration)
        with torch.no_grad():
            action_t, _, _ = actor.sample(s_t.unsqueeze(0))
            action = action_t.cpu().numpy().reshape(-1)
        next_s, r, done, _ = env.step(float(action[0]))
        buffer.push(s.reshape(1,-1), action.reshape(1,-1), r, next_s.reshape(1,-1), done)
        s = next_s
        s_t = torch.tensor(s, dtype=torch.float32, device=DEVICE)
        ep_reward += r
        total_steps += 1

        # update if enough data
        if len(buffer) > REPLAY_INIT:
            s_b, a_b, r_b, s2_b, d_b = buffer.sample(BATCH_SIZE)
            s_b = torch.tensor(s_b, dtype=torch.float32, device=DEVICE)
            a_b = torch.tensor(a_b, dtype=torch.float32, device=DEVICE)
            r_b = torch.tensor(r_b, dtype=torch.float32, device=DEVICE)
            s2_b = torch.tensor(s2_b, dtype=torch.float32, device=DEVICE)
            d_b = torch.tensor(d_b, dtype=torch.float32, device=DEVICE)

            # target
            with torch.no_grad():
                a2, logp_a2, _ = actor.sample(s2_b)
                q1_t = q1_target(s2_b, a2)
                q2_t = q2_target(s2_b, a2)
                qmin = torch.min(q1_t, q2_t)
                if AUTO_ALPHA:
                    alpha = log_alpha.exp()
                target = r_b + GAMMA * (1 - d_b) * (qmin - alpha * logp_a2.squeeze(-1))

            # Q losses
            q1_pred = q1(s_b, a_b)
            q2_pred = q2(s_b, a_b)
            q1_loss = nn.MSELoss()(q1_pred, target)
            q2_loss = nn.MSELoss()(q2_pred, target)
            q1_opt.zero_grad(); q1_loss.backward(); q1_opt.step()
            q2_opt.zero_grad(); q2_loss.backward(); q2_opt.step()

            # actor loss
            a_pi, logp_pi, _ = actor.sample(s_b)
            q1_pi = q1(s_b, a_pi); q2_pi = q2(s_b, a_pi)
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (alpha * logp_pi.squeeze(-1) - q_pi).mean()
            actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

            # alpha loss
            if AUTO_ALPHA:
                alpha_loss = -(log_alpha * (logp_pi + TARGET_ENTROPY).detach()).mean()
                alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()
                alpha = log_alpha.exp().item()

            # soft updates
            for param, target_param in zip(q1.parameters(), q1_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            for param, target_param in zip(q2.parameters(), q2_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        if done:
            break

    episode_rewards.append(ep_reward)
    if len(smoothed) == 0:
        smoothed.append(ep_reward)
    else:
        smoothed.append(smoothed[-1] * (1 - alpha_smooth) + ep_reward * alpha_smooth)

    if ep % 25 == 0 or ep == 1:
        print(f"Ep {ep}/{NUM_EPISODES} Reward {ep_reward:.3f} Avg100 {np.mean(episode_rewards[-100:]):.3f}")

# Save actor+critics
torch.save({
    'actor': actor.state_dict(),
    'q1': q1.state_dict(),
    'q2': q2.state_dict()
}, os.path.join(OUT_DIR, "sac_pointenv_models.pth"))

# Plot
episodes = np.arange(1, len(episode_rewards) + 1)
plt.figure(figsize=(9,5))
plt.plot(episodes, episode_rewards, alpha=0.25, label="Reward por episodio")
plt.plot(episodes, smoothed, label="Media exponencial")
plt.xlabel("Episodio"); plt.ylabel("Reward"); plt.title("SAC Convergencia - PointEnv Continuous")
plt.legend(); plt.grid(True)
plot_path = os.path.join(OUT_DIR, "convergence_sac.png")
plt.savefig(plot_path, bbox_inches="tight"); plt.close()

# PDF
pdf_path = os.path.join(OUT_DIR, "SAC_PointEnv_Report.pdf")
with PdfPages(pdf_path) as pdf:
    plt.figure(figsize=(8.27,11.69)); plt.axis("off")
    txt = ["SAC aplicado a PointEnv (continuo)", "", f"Fecha: {datetime.now().isoformat()}"]
    plt.text(0.02, 0.98, "\n".join(txt), va="top", wrap=True, fontsize=11); pdf.savefig(); plt.close()
    plt.figure(figsize=(8.27,11.69)); plt.axis("off")
    code = ["Fragmentos: actor gaussiano (tanh), 2 Qs, auto-alpha, replay buffer", "", f"REPLAY_INIT: {REPLAY_INIT}"]
    plt.text(0.02, 0.98, "\n".join(code), va="top", wrap=True, fontsize=10); pdf.savefig(); plt.close()
    img = plt.imread(plot_path); plt.figure(figsize=(8.27,11.69)); plt.imshow(img); plt.axis("off"); pdf.savefig(); plt.close()

# README
readme_path = os.path.join(OUT_DIR, "README.txt")
with open(readme_path, "w") as f:
    f.write("SAC PointEnv - Output\n"); f.write(f"Model bundle: {os.path.join(OUT_DIR,'sac_pointenv_models.pth')}\n")
    f.write(f"Plot: {plot_path}\nPDF: {pdf_path}\nEpisodes: {NUM_EPISODES}\n")
print("SAC: terminado. Archivos en:", OUT_DIR)
