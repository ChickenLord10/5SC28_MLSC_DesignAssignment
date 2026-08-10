import optuna
import os
import time
from gp_core import GPNarxModel, create_IO_data, rmse, load_traintest

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../gym-unbalanced-disk-master/disc-benchmark-files/training-val-test-data.npz')
USE_TRIG = False
RESTARTS = 0
NUM_INDUCING = 200

def objective(trial):
    # Suggest variables so they map correctly to the enqueued parameters
    na = trial.suggest_int('na', 2, 10)
    nb = trial.suggest_int('nb', 2, 10)
    
    print(f"Worker started evaluating diagonal na={na}, nb={nb}...")
    
    (u_train, th_train), (u_test, th_test) = load_traintest(DATA_PATH)
    
    start = time.time()
    model = GPNarxModel(na, nb, use_trig=USE_TRIG, num_inducing=NUM_INDUCING)
    model.fit(u_train, th_train, restarts=RESTARTS)
    
    Xte, Yte = create_IO_data(u_test, th_test, na, nb, USE_TRIG)
    _, pred_deg = rmse(model.predict_rows(Xte), Yte)
    
    th_sim = model.simulate(u_test, th_test, skip=max(na, nb))
    _, sim_deg = rmse(th_sim[max(na,nb):], th_test[max(na,nb):])
    
    duration = time.time() - start
    
    trial.set_user_attr('sim_rms_deg', sim_deg)
    print(f"Worker FINISHED na={na}, nb={nb} in {duration:.1f}s | Pred: {pred_deg:.4f} | Sim: {sim_deg:.4f}")
    
    return pred_deg

def main():
    db_path = 'optuna_gp_prediction.db'
    evaluated_diags = set()
    
    try:
        study = optuna.load_study(study_name="GP_Prediction_Sweep", storage=f"sqlite:///{db_path}")
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                na = t.params.get('na')
                nb = t.params.get('nb')
                if na == nb and na is not None:
                    evaluated_diags.add(na)
    except Exception as e:
        print(f"Error loading DB: {e}")
        return

    missing = [i for i in range(2, 11) if i not in evaluated_diags]
    if not missing:
        print("All diagonals (na=nb) from 2 to 10 have already been checked by Optuna!")
        return

    print(f"Enqueuing missing diagonals {missing} into Optuna...")
    
    for i in missing:
        study.enqueue_trial({"na": i, "nb": i})
        
    print("Running them in parallel with 8 workers!...")
    
    # Run the exact number of missing trials using 8 parallel threads
    study.optimize(objective, n_trials=len(missing), n_jobs=8)
    
    print("Done! They are now permanently saved in the optuna_gp_prediction.db database.")

if __name__ == '__main__':
    main()
