import optuna
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, 'results', 'optuna_sweep.db')
    
    study = optuna.load_study(study_name="NOE_GRU_Sweep", storage=f"sqlite:///{db_path}")

    # Evaluate importance 100 times
    num_runs = 100
    all_importances = {'seq_length': [], 'learning_rate': [], 'hidden_dim': [], 'batch_size': []}
    
    print(f"Running fANOVA {num_runs} times to calculate mean and std...")
    
    for i in range(num_runs):
        imp = optuna.importance.get_param_importances(study)
        for k in all_importances.keys():
            all_importances[k].append(imp.get(k, 0.0))
            
    means = {k: np.mean(v) for k, v in all_importances.items()}
    stds = {k: np.std(v) for k, v in all_importances.items()}
    
    # Sort by mean
    sorted_keys = sorted(means.keys(), key=lambda k: means[k], reverse=True)
    
    print("\nRobust Parameter Importances (Mean ± Std over 100 runs):")
    for k in sorted_keys:
        print(f"  {k}: {means[k]:.3f} ± {stds[k]:.3f}")
        
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    
    y_pos = np.arange(len(sorted_keys))
    ax.barh(y_pos, [means[k] for k in sorted_keys], xerr=[stds[k] for k in sorted_keys], align='center', capsize=5, color='skyblue', edgecolor='black')
    
    for i, k in enumerate(sorted_keys):
        ax.text(means[k] + stds[k] + 0.01, i, f"{means[k]:.3f} \u00b1 {stds[k]:.3f}", va='center')
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_keys)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Importance')
    ax.set_title('Robust Hyperparameter Importance (100 runs)')
    
    plt.tight_layout()
    out_path = os.path.join(base_dir, 'results', 'robust_param_importances.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved {out_path}")

if __name__ == '__main__':
    main()
