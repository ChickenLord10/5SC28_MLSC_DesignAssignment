import optuna
import os

def main():
    pred_db_path = 'optuna_gp_prediction.db'
    sim_db_path = 'optuna_gp_simulation.db'
    
    all_trials = []
    
    # 1. Load Prediction Study
    if os.path.exists(pred_db_path):
        try:
            pred_study = optuna.load_study(study_name="GP_Prediction_Sweep", storage=f"sqlite:///{pred_db_path}")
            for t in pred_study.trials:
                if t.state == optuna.trial.TrialState.COMPLETE:
                    pred_err = t.value
                    sim_err = t.user_attrs.get('sim_rms_deg')
                    if sim_err is not None:
                        all_trials.append({
                            'source': 'Prediction_Sweep',
                            'na': t.params['na'],
                            'nb': t.params['nb'],
                            'pred_rms_deg': pred_err,
                            'sim_rms_deg': sim_err,
                            'duration_sec': t.duration.total_seconds() if t.duration else 0.0
                        })
        except Exception as e:
            print(f"Error loading Prediction Study: {e}")
            
    # 2. Load Simulation Study
    if os.path.exists(sim_db_path):
        try:
            sim_study = optuna.load_study(study_name="GP_Simulation_Sweep", storage=f"sqlite:///{sim_db_path}")
            for t in sim_study.trials:
                if t.state == optuna.trial.TrialState.COMPLETE:
                    sim_err = t.value
                    pred_err = t.user_attrs.get('pred_rms_deg')
                    if pred_err is not None:
                        all_trials.append({
                            'source': 'Simulation_Sweep',
                            'na': t.params['na'],
                            'nb': t.params['nb'],
                            'pred_rms_deg': pred_err,
                            'sim_rms_deg': sim_err,
                            'duration_sec': t.duration.total_seconds() if t.duration else 0.0
                        })
        except Exception as e:
            print(f"Error loading Simulation Study: {e}")
            
    if not all_trials:
        print("No completed trials found in either database.")
        return
        
    print(f"\n--- Found {len(all_trials)} total completed trials across both databases ---")
    
    # Find Global Best Prediction
    best_pred = min(all_trials, key=lambda x: x['pred_rms_deg'])
    
    # Find Global Best Simulation
    best_sim = min(all_trials, key=lambda x: x['sim_rms_deg'])
    
    print("\n[GLOBAL BEST PREDICTION CONFIGURATION]")
    print(f"Source: {best_pred['source']}")
    print(f"na: {best_pred['na']}, nb: {best_pred['nb']}")
    print(f"Prediction RMSE (deg): {best_pred['pred_rms_deg']:.4f}")
    print(f"Simulation RMSE (deg): {best_pred['sim_rms_deg']:.4f}")
    print(f"Trial Evaluation Time: {best_pred['duration_sec']:.2f} seconds")
    
    print("\n[GLOBAL BEST SIMULATION CONFIGURATION]")
    print(f"Source: {best_sim['source']}")
    print(f"na: {best_sim['na']}, nb: {best_sim['nb']}")
    print(f"Prediction RMSE (deg): {best_sim['pred_rms_deg']:.4f}")
    print(f"Simulation RMSE (deg): {best_sim['sim_rms_deg']:.4f}")
    print(f"Trial Evaluation Time: {best_sim['duration_sec']:.2f} seconds")
    
    print("\n------------------------------------------------------------\n")

if __name__ == "__main__":
    main()
