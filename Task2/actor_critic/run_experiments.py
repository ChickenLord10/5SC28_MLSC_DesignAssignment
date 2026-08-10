import subprocess
import sys
import os

def run_experiment(name, command):
    print(f"\n{'='*50}")
    print(f"Starting Experiment: {name}")
    print(f"Running: {' '.join(command)}")
    print(f"{'='*50}\n")
    
    # Run the command and wait for it to complete
    result = subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr)
    if result.returncode != 0:
        print(f"\n[ERROR] Experiment '{name}' failed with return code {result.returncode}.")
        sys.exit(result.returncode)
        
    print(f"\n[SUCCESS] Experiment '{name}' finished.\n")

def main():
    base_cmd = [sys.executable, "train_actor_critic.py"]
    eval_cmd = [sys.executable, "evaluate_actor_critic.py", "--steps", "1200"]
    
    experiments = [
        {
            "name": "1. Baseline (2x128 ReLU)",
            "train": base_cmd + ["--out-dir", "results/actor_critic_baseline"],
            "eval_model": "results/actor_critic_baseline/actor_critic_policy.pt",
            "eval_out": "results/actor_critic_baseline"
        },
        {
            "name": "2. Tiny Network (2x32 ReLU)",
            "train": base_cmd + ["--hidden-dims", "32", "32", "--out-dir", "results/actor_critic_tiny"],
            "eval_model": "results/actor_critic_tiny/actor_critic_policy.pt",
            "eval_out": "results/actor_critic_tiny"
        },
        {
            "name": "3. Deep & Wide (3x256 ReLU)",
            "train": base_cmd + ["--hidden-dims", "256", "256", "256", "--out-dir", "results/actor_critic_deep_wide"],
            "eval_model": "results/actor_critic_deep_wide/actor_critic_policy.pt",
            "eval_out": "results/actor_critic_deep_wide"
        },
        {
            "name": "4. Smooth Activations (2x128 Tanh)",
            "train": base_cmd + ["--activation", "tanh", "--out-dir", "results/actor_critic_tanh"],
            "eval_model": "results/actor_critic_tanh/actor_critic_policy.pt",
            "eval_out": "results/actor_critic_tanh"
        },
        {
            "name": "5. Energy-Efficient Reward",
            "train": [sys.executable, "train_actor_critic_energy.py", "--out-dir", "results/actor_critic_energy"],
            "eval_model": "results/actor_critic_energy/actor_critic_policy.pt",
            "eval_out": "results/actor_critic_energy"
        }
    ]
    
    for exp in experiments:
        # Run Training
        run_experiment(f"{exp['name']} [TRAIN]", exp["train"])
        
        # Run Evaluation
        e_cmd = eval_cmd + ["--model", exp["eval_model"], "--out-dir", exp["eval_out"]]
        run_experiment(f"{exp['name']} [EVAL]", e_cmd)
        
    print("\nAll experiments completed successfully! Running plotting script...")
    
    # Run the plotting script
    subprocess.run([sys.executable, "plot_experiments.py"])

if __name__ == "__main__":
    main()
