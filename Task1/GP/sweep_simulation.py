"""
Task 1 -- SIMULATION hyperparameter sweep with Optuna.
Imports the shared model from gp_core (same model gp_simulation.py submits with).
Uses Optuna to find the best na/nb configuration.
"""
import optuna
import os
import time
from gp_core import GPNarxModel, create_IO_data, rmse, load_traintest

# ---------------- config ----------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../gym-unbalanced-disk-master/disc-benchmark-files/training-val-test-data.npz')
USE_TRIG  = False
RESTARTS  = 0
NUM_INDUCING = 200

# We'll need a wrapper objective that accepts optuna trial
class Objective:
    def __init__(self, u_train, th_train, u_test, th_test):
        self.u_train = u_train
        self.th_train = th_train
        self.u_test = u_test
        self.th_test = th_test

    def __call__(self, trial):
        na = trial.suggest_int('na', 2, 10)
        nb = trial.suggest_int('nb', 2, 10)
        
        model = GPNarxModel(na, nb, use_trig=USE_TRIG, num_inducing=NUM_INDUCING)
        model.fit(self.u_train, self.th_train, restarts=RESTARTS)
        
        # Evaluate Prediction
        Xte, Yte = create_IO_data(self.u_test, self.th_test, na, nb, USE_TRIG)
        pred_rad, pred_deg = rmse(model.predict_rows(Xte), Yte)
        
        # Evaluate Simulation
        skip = max(na, nb)
        th_sim = model.simulate(self.u_test, self.th_test, skip=skip)
        sim_rad, sim_deg = rmse(th_sim[skip:], self.th_test[skip:])
        
        # Log both metrics
        trial.set_user_attr('pred_rms_deg', pred_deg)
        trial.set_user_attr('sim_rms_deg', sim_deg)
        
        # Return Simulation Error for Optuna to minimize
        return sim_deg

def main():
    start_time = time.time()
    (u_train, th_train), (u_test, th_test) = load_traintest(DATA_PATH)
    print(f'train points: {len(th_train)}, test points: {len(th_test)}\n')

    db_path = 'optuna_gp_simulation.db'
    pred_db_path = 'optuna_gp_prediction.db'
    
    if os.path.exists(db_path):
        os.remove(db_path)

    study = optuna.create_study(
        direction="minimize", 
        study_name="GP_Simulation_Sweep",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(constant_liar=True)
    )
    
    # --- WARM START LOGIC ---
    if os.path.exists(pred_db_path):
        print("Found Prediction database. Warm-starting from its trials...")
        try:
            pred_study = optuna.load_study(study_name="GP_Prediction_Sweep", storage=f"sqlite:///{pred_db_path}")
            for t in pred_study.trials:
                if t.state == optuna.trial.TrialState.COMPLETE:
                    # For simulation, our objective is the simulation error (sim_rms_deg).
                    # We saved this as a user attribute in the prediction run!
                    sim_val = t.user_attrs.get('sim_rms_deg')
                    pred_val = t.value # The prediction run returned pred_rms_deg
                    
                    if sim_val is not None:
                        study.add_trial(
                            optuna.trial.create_trial(
                                params={"na": t.params["na"], "nb": t.params["nb"]},
                                distributions={
                                    "na": optuna.distributions.IntDistribution(2, 10),
                                    "nb": optuna.distributions.IntDistribution(2, 10)
                                },
                                value=sim_val,
                                user_attrs={'pred_rms_deg': pred_val, 'sim_rms_deg': sim_val}
                            )
                        )
            print(f"Successfully warm-started {len(study.trials)} trials!")
        except Exception as e:
            print(f"Could not warm-start from prediction database: {e}")
    else:
        print("No Prediction database found to warm-start from.")

    objective = Objective(u_train, th_train, u_test, th_test)
    
    # Run 20 trials across 8 CPU workers
    study.optimize(objective, n_trials=20, n_jobs=8, show_progress_bar=True)

    print("\n--- Sweep Complete ---")
    best_trial = study.best_trial
    print(f"Best Simulation RMSE (deg): {best_trial.value:.4f}")
    print(f"Best Params: {best_trial.params}")
    
    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()