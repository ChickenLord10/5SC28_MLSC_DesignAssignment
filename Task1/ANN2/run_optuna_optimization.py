import optuna
import torch
import os
import json
import multiprocessing
from dataset_handler import prepare_dataloaders
from model import NOE_GRU
from trainer import train_model

def objective(trial):
    # Hyperparameter Search Space
    hidden_dim = trial.suggest_categorical("hidden_dim", [8, 16, 32, 64])
    num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])
    use_mlp_head = trial.suggest_categorical("use_mlp_head", [True, False])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FILE_PATH = os.path.join(
        BASE_DIR, '..', '..', 'gym-unbalanced-disk-master', 
        'disc-benchmark-files', 'training-val-test-data.csv'
    )
    
    # Consistent hyperparameters
    SEQ_LENGTH = 100       
    BATCH_SIZE = 256 # Increased batch size for better GPU utilization
    NUM_EPOCHS = 15 
    # For tiny sequential RNNs, CPU is actually faster than GPU due to kernel launch overhead!
    device = torch.device("cpu")

    train_loader, val_loader, test_loader, u_scaler, y_scaler, df = prepare_dataloaders(
        filepath=FILE_PATH, seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE
    )

    model = NOE_GRU(
        input_dim=1, 
        hidden_dim=hidden_dim, 
        output_dim=1, 
        num_layers=num_layers, 
        use_mlp_head=use_mlp_head
    )
    
    # We don't save intermediate models during Optuna to save space
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=learning_rate,
        device=device,
        save_path=None, 
        optuna_trial=trial,
        loss_type="Huber", 
        use_teacher_forcing=False 
    )
    
    # Return the best validation loss achieved during this trial's training
    return min(val_losses)

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'Optuna_Optimization')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Using In-Memory Storage! Absolutely zero SQL or file IO bottlenecks.
    study = optuna.create_study(
        direction="minimize", 
        study_name="NOE_Arch_Optimization"
    )
    
    TOTAL_TRIALS = 30
    NUM_WORKERS = 4
    
    print(f"Starting In-Memory Optuna with {NUM_WORKERS} parallel CPU threads...")
    
    # Optuna natively handles the parallel processing internally using Joblib!
    study.optimize(objective, n_trials=TOTAL_TRIALS, n_jobs=NUM_WORKERS)
        
    print("\n" + "="*50)
    print("Optimization Finished!")
    
    best_trial = study.best_trial
    print(f"Best Trial:")
    print(f"  Value (Validation Loss): {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
        
    params_path = os.path.join(RESULTS_DIR, "best_params.json")
    with open(params_path, 'w') as f:
        json.dump(best_trial.params, f, indent=4)
        
    print(f"\nSaved best params to file:///{params_path.replace(os.sep, '/')}")

if __name__ == "__main__":
    main()
