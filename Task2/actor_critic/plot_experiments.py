import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth

def main():
    experiments = {
        "Baseline (2x128 ReLU)": "results/actor_critic_baseline/actor_critic_training_log.npz",
        "Tiny (2x32 ReLU)": "results/actor_critic_tiny/actor_critic_training_log.npz",
        "Deep (3x256 ReLU)": "results/actor_critic_deep_wide/actor_critic_training_log.npz",
        "Tanh Activations": "results/actor_critic_tanh/actor_critic_training_log.npz",
        "Energy Reward": "results/actor_critic_energy/actor_critic_training_log.npz",
    }
    
    plt.figure(figsize=(10, 6))
    
    for name, path in experiments.items():
        if not os.path.exists(path):
            print(f"Warning: Could not find {path}")
            continue
            
        data = np.load(path)
        returns = data['episode_returns']
        
        # Apply smoothing
        smooth_returns = smooth(returns, 25)
        
        plt.plot(smooth_returns, label=name, linewidth=2, alpha=0.8)
        
    plt.title("Actor-Critic Architecture Ablation (Training Return)", fontsize=14)
    plt.xlabel("Episode (Smoothed window=25)", fontsize=12)
    plt.ylabel("Episode Return", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    out_path = "results/architecture_comparison_curve.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved comparative plot to {out_path}")

if __name__ == "__main__":
    main()
