import os
import numpy as np
import matplotlib.pyplot as plt

def plot_max_q(max_qs, out_path):
    plt.figure()
    plt.plot(max_qs)
    plt.xlabel("Episode")
    plt.ylabel("Max Q (mean over minibatch)")
    plt.title("Max Q vs Episodes")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_rewards(rewards, out_path, window=100):
    rewards = np.array(rewards, dtype=np.float32)
    ma = np.copy(rewards)
    if len(rewards) > 0:
        w = min(window, len(rewards))
        ma = np.convolve(rewards, np.ones(w)/w, mode='same')
    plt.figure()
    plt.plot(rewards, label="Episode reward")
    plt.plot(ma, label=f"Moving avg ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards vs Episodes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_hist(rewards, out_path, bins=30, title="Reward Histogram"):
    plt.figure()
    plt.hist(rewards, bins=bins)
    plt.xlabel("Reward")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
