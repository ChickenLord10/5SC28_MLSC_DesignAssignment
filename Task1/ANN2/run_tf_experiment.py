import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import json
import datetime
from dataset_handler import prepare_dataloaders
from model import NOE_GRU
from trainer import train_model
from compare_models import compare_models

def run_experiment(name, use_tf, tf_decay_factor=0.0):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'TF_Experiment')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    FILE_PATH = os.path.join(
        BASE_DIR, '..', '..', 'gym-unbalanced-disk-master', 
        'disc-benchmark-files', 'training-val-test-data.csv'
    )
    
    # Consistent hyperparameters for direct comparison
    SEQ_LENGTH = 100       
    BATCH_SIZE = 64
    HIDDEN_DIM = 16        
    LEARNING_RATE = 0.005
    NUM_EPOCHS = 20
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}\nStarting TF Experiment: {name}\nDecay Factor: {tf_decay_factor}\n{'='*50}")

    train_loader, val_loader, test_loader, u_scaler, y_scaler, df = prepare_dataloaders(
        filepath=FILE_PATH, seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE
    )

    model = NOE_GRU(input_dim=1, hidden_dim=HIDDEN_DIM, output_dim=1)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    base_filename = f"TFExp_{name}_{HIDDEN_DIM}units_{timestamp}"
    model_save_path = os.path.join(RESULTS_DIR, f"best_model_{base_filename}.pth")

    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device,
        save_path=model_save_path,
        loss_type="Huber", # Using the optimal Huber loss discovered in the Loss Experiment
        use_teacher_forcing=use_tf,
        tf_decay_factor=tf_decay_factor
    )

    data_to_save = {
        "experiment_name": name,
        "use_teacher_forcing": use_tf,
        "tf_decay_factor": tf_decay_factor,
        "train_loss": train_losses,
        "val_loss": val_losses
    }
    
    json_path = os.path.join(RESULTS_DIR, f"learning_curve_{base_filename}.json")
    with open(json_path, 'w') as f:
        json.dump(data_to_save, f, indent=4)
        
    return train_losses, val_losses, base_filename

def main():
    # 1. No TF
    t0, v0, n0 = run_experiment("No_TF", use_tf=False, tf_decay_factor=0.0)
    
    # 2. Fast Decay (over 25% of epochs)
    t25, v25, n25 = run_experiment("TF_Decay_25", use_tf=True, tf_decay_factor=0.25)
    
    # 3. Medium Decay (over 50% of epochs)
    t50, v50, n50 = run_experiment("TF_Decay_50", use_tf=True, tf_decay_factor=0.50)
    
    # 4. Slow Decay (over 75% of epochs)
    t75, v75, n75 = run_experiment("TF_Decay_75", use_tf=True, tf_decay_factor=0.75)
    
    # Compare
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'TF_Experiment')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    plt.figure(figsize=(12, 7))
    plt.plot(v0, label='No Teacher Forcing', color='red', linestyle='solid')
    plt.plot(v25, label='Fast Decay (25%)', color='orange', linestyle='dashed')
    plt.plot(v50, label='Medium Decay (50%)', color='green', linestyle='dashdot')
    plt.plot(v75, label='Slow Decay (75%)', color='blue', linestyle='dotted')
    
    plt.title('Teacher Forcing Decay Rates (Validation Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss Value')
    plt.yscale('log')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(RESULTS_DIR, 'TF_Experiment_Comparison.png')
    plt.savefig(plot_path)
    print(f"\nSuccessfully finished TF experiments!")
    print(f"Comparison plot saved to: file:///{plot_path.replace(os.sep, '/')}")
    
    # NEW: Automatically run the rigorous evaluation
    print("\nRunning head-to-head physical RMSE simulation...")
    compare_models("TF_Experiment")
    
if __name__ == "__main__":
    main()
