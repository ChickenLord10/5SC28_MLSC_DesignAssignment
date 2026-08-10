import optuna
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_all_trials():
    all_trials = []
    
    if os.path.exists('optuna_gp_prediction.db'):
        try:
            study = optuna.load_study(study_name="GP_Prediction_Sweep", storage="sqlite:///optuna_gp_prediction.db")
            for t in study.trials:
                if t.state == optuna.trial.TrialState.COMPLETE:
                    sim_val = t.user_attrs.get('sim_rms_deg')
                    if sim_val is not None:
                        all_trials.append({'na': t.params['na'], 'nb': t.params['nb'], 'pred': t.value, 'sim': sim_val})
        except: pass
            
    if os.path.exists('optuna_gp_simulation.db'):
        try:
            study = optuna.load_study(study_name="GP_Simulation_Sweep", storage="sqlite:///optuna_gp_simulation.db")
            for t in study.trials:
                if t.state == optuna.trial.TrialState.COMPLETE:
                    pred_val = t.user_attrs.get('pred_rms_deg')
                    if pred_val is not None:
                        all_trials.append({'na': t.params['na'], 'nb': t.params['nb'], 'pred': pred_val, 'sim': t.value})
        except: pass
            
    return all_trials

def main():
    trials = load_all_trials()
    if not trials:
        print("No trials found.")
        return
        
    pred_grid = {}
    sim_grid = {}
    
    for t in trials:
        k = (t['na'], t['nb'])
        if k not in pred_grid or t['pred'] < pred_grid[k]: pred_grid[k] = t['pred']
        if k not in sim_grid or t['sim'] < sim_grid[k]: sim_grid[k] = t['sim']
            
    # Create 9x9 matrices filled with NaN
    nas = list(range(2, 11))
    nbs = list(range(2, 11))
    
    pred_mat = np.full((len(nbs), len(nas)), np.nan)
    sim_mat = np.full((len(nbs), len(nas)), np.nan)
    
    for i, nb in enumerate(nbs):
        for j, na in enumerate(nas):
            if (na, nb) in pred_grid:
                pred_mat[i, j] = pred_grid[(na, nb)]
            if (na, nb) in sim_grid:
                sim_mat[i, j] = sim_grid[(na, nb)]
                
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Custom color map (dark blue for good/low error, light red for bad/high error)
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    
    # 1. Prediction Heatmap
    sns.heatmap(pred_mat, annot=True, fmt=".4f", cmap="viridis_r", ax=axes[0],
                xticklabels=nas, yticklabels=nbs, cbar_kws={'label': 'RMSE (deg)'},
                mask=np.isnan(pred_mat))
    axes[0].set_title('GP Prediction Error Sweep', fontsize=14, pad=15)
    axes[0].set_xlabel('$n_a$', fontsize=12)
    axes[0].set_ylabel('$n_b$', fontsize=12)
    
    # 2. Simulation Heatmap
    sns.heatmap(sim_mat, annot=True, fmt=".2f", cmap="viridis_r", ax=axes[1],
                xticklabels=nas, yticklabels=nbs, cbar_kws={'label': 'RMSE (deg)'},
                mask=np.isnan(sim_mat))
    axes[1].set_title('GP Simulation Error Sweep', fontsize=14, pad=15)
    axes[1].set_xlabel('$n_a$', fontsize=12)
    axes[1].set_ylabel('$n_b$', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('gp_sweep_heatmaps.png', dpi=300, bbox_inches='tight')
    print("Saved gp_sweep_heatmaps.png!")

if __name__ == '__main__':
    main()
