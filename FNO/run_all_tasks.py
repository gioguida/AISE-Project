import sys
import io
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import contextlib
import torch
import numpy as np

# Import task modules
import task1_one2one
import task2_resolution_invariance
import task3_all2all
import task4_finetuning

def run_tasks():
    tasks = [
        (task1_one2one, "Task 1: One-to-One"),
        (task2_resolution_invariance, "Task 2: Resolution Invariance"),
        (task3_all2all, "Task 3: All-to-All"),
        (task4_finetuning, "Task 4: Finetuning")
    ]

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    output_filename = "results/all_outputs.txt"

    print(f"Starting execution of {len(tasks)} tasks...")
    print(f"Outputs will be saved to {output_filename}")
    print(f"Graphs will be saved as individual PDF files in results/ folder.")

    # State for tracking current task and figure count
    state = {"task_name": "", "fig_count": 0}

    with open(output_filename, "w") as f_out:
        
        # Helper class to redirect stdout to both console and file
        class Tee(object):
            def __init__(self, *files):
                self.files = files
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()
            def flush(self):
                for f in self.files:
                    f.flush()

        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, f_out)

        # Monkey patch plt.show to save to pdf instead of displaying
        original_show = plt.show
        def custom_show():
            try:
                state["fig_count"] += 1
                # Create a safe filename
                safe_name = state["task_name"].replace(":", "").replace(" ", "_")
                filename = f"results/{safe_name}_fig_{state['fig_count']}.pdf"
                
                plt.savefig(filename)  # saves the current figure
                print(f"Saved graph to {filename}")
                plt.close()
            except Exception as e:
                print(f"Error saving plot: {e}")
        
        plt.show = custom_show

        try:
            for module, name in tasks:
                print(f"\n{'='*40}\nRunning {name}...\n{'='*40}\n")
                
                # Update state for the current task
                state["task_name"] = name
                state["fig_count"] = 0

                # Reset seeds for consistency between runs if needed, 
                # though modules set them globally.
                torch.manual_seed(0)
                np.random.seed(0)

                try:
                    if hasattr(module, 'main'):
                        module.main()
                    else:
                        print(f"Module {name} does not have a main() function.")
                except Exception as e:
                    print(f"Error in {name}: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Ensure all figures are closed/saved even if plt.show wasn't called explicitly
                # or if multiple figures were created
                for i in plt.get_fignums():
                    plt.figure(i)
                    custom_show()
                
                # Force garbage collection to free memory
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        finally:
            sys.stdout = original_stdout
            plt.show = original_show
            print(f"\n{'='*40}")
            print(f"All tasks completed.")
            print(f"Outputs saved to {output_filename}")

if __name__ == "__main__":
    run_tasks()
