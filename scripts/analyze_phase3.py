import json
import os
import matplotlib.pyplot as plt

# --- UPDATE THIS PATH to where you downloaded the folder on your Mac ---
DRIVE_DIR = "/Users/manishsingh/Desktop/zero-lora/zero-lora-phase3"

def load_results(base_dir):
    results = []
    if not os.path.exists(base_dir):
        print(f"Error: Could not find directory {base_dir}")
        return results
        
    for folder in os.listdir(base_dir):
        summary_path = os.path.join(base_dir, folder, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                data = json.load(f)
                results.append(data)
    return results

def print_table(results):
    print(f"\n| Loss | Lambda | Frozen Task Acc | Final Task Acc | Delta Task | Frozen ZS | Final ZS | Delta ZS |")
    print(f"|---|---|---|---|---|---|---|---|")
    
    # Sort by loss, then lambda
    results.sort(key=lambda x: (x['config']['loss'], x['config']['lam']))
    
    for r in results:
        loss = r['config']['loss']
        lam = r['config']['lam']
        f_task = r['frozen']['dev_task_acc']
        l_task = r['final']['dev_task_acc']
        d_task = r['delta_dev_task_acc']
        f_zs = r['frozen']['zs_mean']
        l_zs = r['final']['zs_mean']
        d_zs = r['delta_zs_mean']
        
        print(f"| {loss} | {lam} | {f_task:.4f} | {l_task:.4f} | +{d_task:.4f} | {f_zs:.4f} | {l_zs:.4f} | {d_zs:+.4f} |")

def plot_pareto(results):
    plt.figure(figsize=(10, 6))
    
    # Separate by loss type for coloring
    raw_points = [r for r in results if r['config']['loss'] == 'raw']
    gnsp_points = [r for r in results if r['config']['loss'] == 'gnsp']
    
    # Plot Raw
    for r in raw_points:
        lam = r['config']['lam']
        x = r['delta_dev_task_acc'] * 100 # Convert to percentage points
        y = r['delta_zs_mean'] * 100
        
        color = 'blue' if lam != 0.0 else 'black'
        marker = 'o' if lam != 0.0 else 'X'
        size = 100 if lam != 0.0 else 150
        
        plt.scatter(x, y, color=color, marker=marker, s=size, label='Raw Frobenius' if lam == 0.1 else "")
        plt.annotate(f"λ={lam}", (x, y), textcoords="offset points", xytext=(0,10), ha='center')

    # Plot GNSP
    for r in gnsp_points:
        lam = r['config']['lam']
        x = r['delta_dev_task_acc'] * 100
        y = r['delta_zs_mean'] * 100
        
        plt.scatter(x, y, color='red', marker='s', s=100, label='GNSP-KL' if lam == 10.0 else "")
        plt.annotate(f"λ={lam}", (x, y), textcoords="offset points", xytext=(0,10), ha='center')

    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title('Zero-LoRA: Task Gain vs. Zero-Shot Drop (DTD, 3 Epochs)')
    plt.xlabel('Delta Task Accuracy (Percentage Points)')
    plt.ylabel('Delta Zero-Shot Retention (Percentage Points)')
    plt.grid(True, alpha=0.3)
    
    # Custom legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='black', marker='X', linestyle='None', markersize=10),
                    Line2D([0], [0], color='blue', marker='o', linestyle='None', markersize=10),
                    Line2D([0], [0], color='red', marker='s', linestyle='None', markersize=10)]
    plt.legend(custom_lines, ['Baseline (Plain LoRA)', 'Raw Frobenius', 'GNSP-KL (Distributional)'])
    
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(DRIVE_DIR), "pareto_frontier.png")
    plt.savefig(out_path, dpi=300)
    print(f"\nPlot saved to: {out_path}")
    plt.show()

if __name__ == "__main__":
    results = load_results(DRIVE_DIR)
    if results:
        print_table(results)
        plot_pareto(results)