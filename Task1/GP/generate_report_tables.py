import optuna
import os

def load_all_trials():
    all_trials = []
    
    # 1. Load Prediction Sweep
    if os.path.exists('optuna_gp_prediction.db'):
        try:
            study = optuna.load_study(study_name="GP_Prediction_Sweep", storage="sqlite:///optuna_gp_prediction.db")
            for t in study.trials:
                if t.state == optuna.trial.TrialState.COMPLETE:
                    sim_val = t.user_attrs.get('sim_rms_deg')
                    if sim_val is not None:
                        all_trials.append({
                            'na': t.params['na'],
                            'nb': t.params['nb'],
                            'pred': t.value,
                            'sim': sim_val,
                            'time': t.duration.total_seconds() if t.duration else 0.0
                        })
        except Exception as e:
            print(f"% Could not read Prediction DB: {e}")
            
    # 2. Load Simulation Sweep
    if os.path.exists('optuna_gp_simulation.db'):
        try:
            study = optuna.load_study(study_name="GP_Simulation_Sweep", storage="sqlite:///optuna_gp_simulation.db")
            for t in study.trials:
                if t.state == optuna.trial.TrialState.COMPLETE:
                    pred_val = t.user_attrs.get('pred_rms_deg')
                    if pred_val is not None:
                        all_trials.append({
                            'na': t.params['na'],
                            'nb': t.params['nb'],
                            'pred': pred_val,
                            'sim': t.value,
                            'time': t.duration.total_seconds() if t.duration else 0.0
                        })
        except Exception as e:
            print(f"% Could not read Simulation DB: {e}")
            
    return all_trials

def print_latex_table(grid, title, label, is_time=False):
    nas = range(2, 11)
    nbs = range(2, 11)
    print("\\begin{table*}[t]")
    print("\\centering")
    print(f"\\caption{{{title}}}")
    print(f"\\label{{{label}}}")
    print("\\begin{tabular}{|c|" + "c|"*len(nas) + "}")
    print("\\hline")
    print("$n_b$ \\textbackslash{} $n_a$ & " + " & ".join(map(str, nas)) + " \\\\ \\hline")
    
    for nb in nbs:
        row = [str(nb)]
        for na in nas:
            val = grid.get((na, nb))
            if val is not None:
                # Format to 4 decimals for errors, 1 decimal for seconds
                if is_time:
                    row.append(f"{val:.1f}")
                else:
                    # Optional: We could find the global minimum and bold it here if we wanted
                    row.append(f"{val:.4f}")
            else:
                row.append("-")
        print(" & ".join(row) + " \\\\ \\hline")
        
    print("\\end{tabular}")
    print("\\end{table*}\n")

def main():
    trials = load_all_trials()
    if not trials:
        print("% No completed Optuna trials found yet!")
        return
        
    # Aggregate data (if duplicate configs exist, we take the minimum/best value)
    pred_grid = {}
    sim_grid = {}
    time_grid = {}
    
    for t in trials:
        k = (t['na'], t['nb'])
        
        # Best prediction error
        if k not in pred_grid or t['pred'] < pred_grid[k]:
            pred_grid[k] = t['pred']
            
        # Best simulation error
        if k not in sim_grid or t['sim'] < sim_grid[k]:
            sim_grid[k] = t['sim']
            
        # Evaluation time
        if k not in time_grid or t['time'] > time_grid[k]:
            time_grid[k] = t['time']
            
    print(f"% --- Generating 3 LaTeX Tables from {len(trials)} total trials! ---\n")
    
    print_latex_table(pred_grid, "Optuna GP Prediction Error Sweep (RMSE [deg])", "tab:gp_pred")
    print_latex_table(sim_grid,  "Optuna GP Simulation Error Sweep (RMSE [deg])", "tab:gp_sim")
    print_latex_table(time_grid, "Optuna GP Evaluation Time (Seconds)", "tab:gp_time", is_time=True)
    
if __name__ == '__main__':
    main()
