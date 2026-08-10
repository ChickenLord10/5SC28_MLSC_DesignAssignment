import optuna
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from optuna.visualization.matplotlib import plot_param_importances, plot_optimization_history

def main():
    # Construct absolute path to DB
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, 'results', 'optuna_sweep.db')
    
    study = optuna.load_study(study_name="NOE_GRU_Sweep", storage=f"sqlite:///{db_path}")

    # 1. Plot Parameter Importances
    try:
        ax = plot_param_importances(study)
        fig = ax.figure
        fig.set_size_inches(8, 6)
        plt.tight_layout()
        out_path1 = os.path.join(base_dir, 'results', 'param_importances.png')
        fig.savefig(out_path1, dpi=300, bbox_inches='tight')
        print(f"Saved {out_path1}")
        plt.close(fig)
    except Exception as e:
        print(f"Failed to plot importances: {e}")

    # 2. Plot Optimization History
    try:
        ax2 = plot_optimization_history(study)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig2 = ax2.figure
        fig2.set_size_inches(8, 6)
        plt.tight_layout()
        out_path2 = os.path.join(base_dir, 'results', 'optimization_history.png')
        fig2.savefig(out_path2, dpi=300, bbox_inches='tight')
        print(f"Saved {out_path2}")
        plt.close(fig2)
    except Exception as e:
        print(f"Failed to plot optimization history: {e}")

if __name__ == '__main__':
    main()
