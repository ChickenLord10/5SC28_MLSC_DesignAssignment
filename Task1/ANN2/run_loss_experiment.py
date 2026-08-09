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

def run_experiment(name, loss_type):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'Loss_Experiment')
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
    print(f"\n{'='*50}\nStarting Loss Experiment: {name}\nLoss: {loss_type} (Pure NOE, No TF)\n{'='*50}")

    train_loader, val_loader, test_loader, u_scaler, y_scaler, df = prepare_dataloaders(
        filepath=FILE_PATH, seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE
    )

    model = NOE_GRU(input_dim=1, hidden_dim=HIDDEN_DIM, output_dim=1)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    base_filename = f"LossExp_{name}_{HIDDEN_DIM}units_{timestamp}"
    model_save_path = os.path.join(RESULTS_DIR, f"best_model_{base_filename}.pth")

    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device,
        save_path=model_save_path,
        loss_type=loss_type,
        use_teacher_forcing=False # Strict ablation for loss
    )

    data_to_save = {
        "experiment_name": name,
        "loss_type": loss_type,
        "train_loss": train_losses,
        "val_loss": val_losses
    }
    
    json_path = os.path.join(RESULTS_DIR, f"learning_curve_{base_filename}.json")
    with open(json_path, 'w') as f:
        json.dump(data_to_save, f, indent=4)
        
    return train_losses, val_losses, base_filename

def main():
    # 1. MSE
    t_mse, v_mse, name_mse = run_experiment("MSE_Only", loss_type="MSE")
    
    # 2. Huber
    t_huber, v_huber, name_huber = run_experiment("Huber_Only", loss_type="Huber")
    
    # Compare
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'Loss_Experiment')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(v_mse, label='Standard MSE Loss', color='red', linestyle='solid')
    plt.plot(v_huber, label='Robust Huber Loss', color='blue', linestyle='dashed')
    plt.title('Loss Function Comparison (Validation)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.yscale('log')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(RESULTS_DIR, 'Loss_Experiment_Comparison.png')
    plt.savefig(plot_path)
    print(f"\nSuccessfully finished Loss experiments!")
    print(f"Comparison plot saved to: file:///{plot_path.replace(os.sep, '/')}")
    
    # NEW: Automatically run the rigorous evaluation
    print("\nRunning head-to-head physical RMSE simulation...")
    compare_models("Loss_Experiment")
    
if __name__ == "__main__":
    main()
