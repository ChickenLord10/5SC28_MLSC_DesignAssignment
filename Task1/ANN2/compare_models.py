import torch
import matplotlib.pyplot as plt
import os
import glob
import json
import numpy as np
from dataset_handler import prepare_dataloaders
from model import NOE_GRU

def compare_models(experiment_dir_name):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    EXP_DIR = os.path.join(BASE_DIR, 'results', experiment_dir_name)
    
    pth_files = glob.glob(os.path.join(EXP_DIR, '*.pth'))
    if not pth_files:
        print(f"No .pth files found in {EXP_DIR}")
        return
        
    FILE_PATH = os.path.join(
        BASE_DIR, '..', '..', 'gym-unbalanced-disk-master', 
        'disc-benchmark-files', 'training-val-test-data.csv'
    )
    
    VISUALIZATION_SEQ_LENGTH = 1000 
    print(f"Loading test data for {experiment_dir_name} comparison...")
    train_loader, val_loader, test_loader, u_scaler, y_scaler, df = prepare_dataloaders(
        filepath=FILE_PATH,
        seq_length=VISUALIZATION_SEQ_LENGTH,
        batch_size=1,
        train_split=0.6,
        val_split=0.2
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        u_seq, y_seq = next(iter(test_loader))
    except StopIteration:
        print("Test dataset too short.")
        return
        
    u_seq, y_seq = u_seq.to(device), y_seq.to(device)
    y_init = y_seq[:, 0, :]
    
    y_true_np = y_seq.cpu().squeeze().numpy().reshape(-1, 1)
    y_true_real = y_scaler.inverse_transform(y_true_np)
    u_seq_np = u_seq.cpu().squeeze().numpy().reshape(-1, 1)
    u_real = u_scaler.inverse_transform(u_seq_np)
    
    colors = ['red', 'blue', 'green', 'purple', 'magenta']
    
    print("\n--- Final RMSE Results (Unscaled Physical rads) ---")
    
    # Store results for text file and plotting
    results_data = []
    
    for idx, pth_path in enumerate(pth_files):
        base_name = os.path.basename(pth_path).replace('best_model_', '').replace('.pth', '')
        
        json_path = os.path.join(EXP_DIR, f"learning_curve_{base_name}.json")
        hidden_dim = 16
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                info = json.load(f)
                hidden_dim = info.get("hidden_dim", 16)
                
        model = NOE_GRU(input_dim=1, hidden_dim=hidden_dim, output_dim=1)
        model.load_state_dict(torch.load(pth_path, map_location=device))
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            y_pred_seq = model(u_seq, y_init)
            
        y_pred_np = y_pred_seq.cpu().squeeze().numpy().reshape(-1, 1)
        y_pred_real = y_scaler.inverse_transform(y_pred_np)
        
        mse = np.mean((y_true_real - y_pred_real)**2)
        rmse = np.sqrt(mse)
        
        display_parts = base_name.split('_')
        if len(display_parts) >= 4:
            display_name = "_".join(display_parts[1:-2]) 
        else:
            display_name = base_name
            
        print(f"{display_name:20s}: {rmse:.4f} rad")
        
        results_data.append({
            'name': display_name,
            'rmse': rmse,
            'y_pred': y_pred_real,
            'color': colors[idx % len(colors)]
        })
        
    # Write to text file
    txt_path = os.path.join(EXP_DIR, 'rmse_results.txt')
    with open(txt_path, 'w') as f:
        f.write(f"--- Final RMSE Results ({experiment_dir_name}) ---\n")
        for res in results_data:
            f.write(f"{res['name']:20s}: {res['rmse']:.4f} rad\n")
    print(f"Saved RMSE results to file:///{txt_path.replace(os.sep, '/')}")
            
    # 1. Full Overlay Plot
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.plot(y_true_real, label='True Angle (y)', color='black', alpha=0.5, linewidth=2)
    for res in results_data:
        plt.plot(res['y_pred'], label=f"Sim: {res['name']} (RMSE: {res['rmse']:.4f})", color=res['color'], linestyle='dashed', alpha=0.8)
    plt.title(f'{experiment_dir_name} - Physical Simulation Overlay (Full)')
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
    plot_path_full = os.path.join(EXP_DIR, f'physical_overlay_full_{experiment_dir_name}.png')
    plt.savefig(plot_path_full)
    
    # 2. Zoomed Overlay Plot (200 to 400)
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.plot(range(200, 400), y_true_real[200:400], label='True Angle (y)', color='black', alpha=0.5, linewidth=2)
    for res in results_data:
        plt.plot(range(200, 400), res['y_pred'][200:400], label=f"Sim: {res['name']}", color=res['color'], linestyle='dashed', alpha=0.9, linewidth=1.5)
    plt.title(f'{experiment_dir_name} - Physical Simulation Overlay (Zoomed 200-400)')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(range(200, 400), u_real[200:400], label='Input Voltage (u)', color='orange')
    plt.xlabel('Time Step')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path_zoomed = os.path.join(EXP_DIR, f'physical_overlay_zoomed_{experiment_dir_name}.png')
    plt.savefig(plot_path_zoomed)
    
    print(f"Saved full plot to file:///{plot_path_full.replace(os.sep, '/')}")
    print(f"Saved zoomed plot to file:///{plot_path_zoomed.replace(os.sep, '/')}")

if __name__ == "__main__":
    import sys
    exp_dir = "Loss_Experiment"
    if len(sys.argv) > 1:
        exp_dir = sys.argv[1]
    compare_models(exp_dir)
