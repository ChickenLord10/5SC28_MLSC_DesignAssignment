import torch
import matplotlib.pyplot as plt
import os
import glob
import json
from dataset_handler import prepare_dataloaders
from model import NOE_GRU

def evaluate_and_plot():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    
    # 1. Find the most recently trained model in results/
    pth_files = glob.glob(os.path.join(RESULTS_DIR, '*.pth'))
    if not pth_files:
        print("No .pth model files found in results/")
        return
        
    latest_model_path = max(pth_files, key=os.path.getctime)
    print(f"Loading latest model: {os.path.basename(latest_model_path)}")
    
    # Try to find corresponding JSON to get hidden_dim
    base_name = os.path.basename(latest_model_path).replace('best_model_', '').replace('.pth', '')
    json_path = os.path.join(RESULTS_DIR, f"learning_curve_{base_name}.json")
    
    hidden_dim = 16 # Default
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            info = json.load(f)
            hidden_dim = info.get("hidden_dim", 16)
            
    # 2. Prepare Data (we want a long continuous sequence for visualization)
    FILE_PATH = os.path.join(
        BASE_DIR, '..', '..', 'gym-unbalanced-disk-master', 
        'disc-benchmark-files', 'training-val-test-data.csv'
    )
    
    # We use a long sequence length for visualization
    VISUALIZATION_SEQ_LENGTH = 1000 
    
    print("Loading test data...")
    train_loader, val_loader, test_loader, u_scaler, y_scaler, df = prepare_dataloaders(
        filepath=FILE_PATH,
        seq_length=VISUALIZATION_SEQ_LENGTH,
        batch_size=1, # One sequence at a time
        train_split=0.6,
        val_split=0.2
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = NOE_GRU(input_dim=1, hidden_dim=hidden_dim, output_dim=1)
    model.load_state_dict(torch.load(latest_model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 3. Get the first batch from the Test Set
    try:
        u_seq, y_seq = next(iter(test_loader))
    except StopIteration:
        print("Test dataset is too short for the chosen sequence length.")
        return

    u_seq, y_seq = u_seq.to(device), y_seq.to(device)
    y_init = y_seq[:, 0, :]
    
    # 4. Predict
    print("Running NOE simulation on test data (free-running mode)...")
    with torch.no_grad():
        y_pred_seq = model(u_seq, y_init)
        
    # 5. Inverse scale back to physical units
    u_seq_np = u_seq.cpu().squeeze().numpy().reshape(-1, 1)
    y_true_np = y_seq.cpu().squeeze().numpy().reshape(-1, 1)
    y_pred_np = y_pred_seq.cpu().squeeze().numpy().reshape(-1, 1)
    
    y_true_real = y_scaler.inverse_transform(y_true_np)
    y_pred_real = y_scaler.inverse_transform(y_pred_np)
    u_real = u_scaler.inverse_transform(u_seq_np)
    
    # 6. Plotting - Full Sequence
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(y_true_real, label='True Angle (y)', color='blue', alpha=0.7)
    plt.plot(y_pred_real, label=r'Simulated Angle ($\hat{y}$)', color='red', linestyle='dashed', alpha=0.9)
    plt.title(f'NOE Simulation vs Reality (Test Set) - Full - {base_name}')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(u_real, label='Input Voltage (u)', color='orange')
    plt.xlabel('Time Step')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path_full = os.path.join(RESULTS_DIR, f'simulation_viz_full_{base_name}.png')
    plt.savefig(plot_path_full)
    print(f"Full visualization saved to file:///{plot_path_full.replace(os.sep, '/')}")
    
    # 7. Plotting - Zoomed Sequence (200 to 400)
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(range(200, 400), y_true_real[200:400], label='True Angle (y)', color='blue', alpha=0.7, marker='o', markersize=3)
    plt.plot(range(200, 400), y_pred_real[200:400], label=r'Simulated Angle ($\hat{y}$)', color='red', linestyle='dashed', alpha=0.9, marker='x', markersize=3)
    plt.title(f'NOE Simulation vs Reality (Test Set) - ZOOM 200:400 - {base_name}')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(range(200, 400), u_real[200:400], label='Input Voltage (u)', color='orange', marker='s', markersize=3)
    plt.xlabel('Time Step')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path_zoomed = os.path.join(RESULTS_DIR, f'simulation_viz_zoomed_{base_name}.png')
    plt.savefig(plot_path_zoomed)
    print(f"Zoomed visualization saved to file:///{plot_path_zoomed.replace(os.sep, '/')}")
    
if __name__ == "__main__":
    evaluate_and_plot()
