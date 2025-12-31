import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt
import copy

torch.manual_seed(0)
np.random.seed(0)

from FNO_bn import FNO1d_bn
from task3_all2all import ( Config_all2all, 
                            PDEDataset,
                            create_dataloaders,
                            train_all2all,
                            load_test_set_at_time,
                            test_at_one_time,
                            test_at_all_times,
                            check_data_statistics
                            )                           

class Config_finetune:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODES = 16
    WIDTH = 64
    CHANNELS = 2

    N_TRAIN = 32
    BATCH_SIZE = 16
    
    LEARNING_RATE_ADAM = 0.001
    EPOCHS_FINETUNE = 100
    EPOCHS_SCRATCH = 100
    EPOCHS_ADAM = EPOCHS_FINETUNE # Default

    FACTOR_SCHEDULER = 0.5
    PATIENCE_SCHEDULER = 5
    MIN_LR_SCHEDULER = 1e-6  

    BATCH_SIZE_TEST = 20
    
    SAVE_FINETUNED_MODEL = False
    SAVE_SCRATCH_MODEL = False

    # Flags to control execution
    RUN_ZERO_SHOT = True
    
    RUN_FINETUNE = True
    TRAIN_FINETUNE = False # If False, load saved model
    
    RUN_SCRATCH = True
    TRAIN_SCRATCH = False # If False, load saved model
    
    PLOT_RESULTS = True

def run_zero_shot(config):
    print("\n--- Zero-shot Test on Unknown Data ---")
    model_pretrained = FNO1d_bn(config.MODES, config.WIDTH).to(config.DEVICE)
    try:
        model_pretrained.load_state_dict(torch.load(f"models/fno_{config.MODES}_bn_model.pth", map_location=torch.device(config.DEVICE)))
        print(f"Loaded models/fno_{config.MODES}_bn_model.pth")
    except FileNotFoundError:
        print("Pre-trained model not found. Please run task3_all2all.py first.")
        return None, None, None

    time_steps, errors_zeroshot = test_at_all_times(config, model_pretrained, finetune=True)
    print("Zero-shot errors:", errors_zeroshot)
    return time_steps, errors_zeroshot, model_pretrained

def run_finetuning(config, training_set, testing_set, model_pretrained):
    print("\n--- Finetuning Model ---")
    model_finetuned = None

    if config.TRAIN_FINETUNE:
        if model_pretrained is None:
            print("Pre-trained model is None, cannot finetune.")
            return None

        print("Training finetuned model...")
        config.EPOCHS_ADAM = config.EPOCHS_FINETUNE
        model_finetuned = copy.deepcopy(model_pretrained)
        
        model_finetuned, history_finetune = train_all2all(config,
                                                        model_finetuned, 
                                                        training_set, 
                                                        testing_set, 
                                                        save_model=False # We save manually below
                                                        )
        
        # Plot training history
        plt.figure(figsize=(8, 5))
        plt.plot(history_finetune['train_loss'])
        plt.title('Training Loss (Finetuning)', fontweight="bold")
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(history_finetune['test_relative_l2'])
        plt.title('Test Relative L2 Error (Finetuning)', fontweight="bold")
        plt.xlabel('Epoch')
        plt.ylabel('Relative L2 Error (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
        
        if config.SAVE_FINETUNED_MODEL:
            torch.save(model_finetuned.state_dict(), f"models/fno_{config.MODES}_bn_finetuned.pth")
            print(f"Saved finetuned model to models/fno_{config.MODES}_bn_finetuned.pth")
    else:
        print("Loading finetuned model from file (TRAIN_FINETUNE=False)...")
        model_finetuned = FNO1d_bn(config.MODES, config.WIDTH).to(config.DEVICE)
        try:
            model_finetuned.load_state_dict(torch.load(f"models/fno_{config.MODES}_bn_finetuned.pth", map_location=torch.device(config.DEVICE)))
            print(f"Loaded models/fno_{config.MODES}_bn_finetuned.pth")
        except FileNotFoundError:
            print(f"File models/fno_{config.MODES}_bn_finetuned.pth not found. Cannot test.")
            return None

    print("\n--- Testing Finetuned Model ---")
    _, errors_finetuned = test_at_all_times(config, model_finetuned, finetune=True)
    print("Finetuned errors:", errors_finetuned)
    return errors_finetuned

def run_scratch(config, training_set, testing_set):
    print("\n--- Training from Scratch on Finetune Dataset ---")
    model_scratch = None

    if config.TRAIN_SCRATCH:
        print("Training model from scratch...")
        config.EPOCHS_ADAM = config.EPOCHS_SCRATCH
        model_scratch = FNO1d_bn(config.MODES, config.WIDTH).to(config.DEVICE)
        model_scratch, history_scratch = train_all2all(config, 
                                                    model_scratch, 
                                                    training_set, 
                                                    testing_set, 
                                                    save_model=False # We save manually below
                                                    )
        
        # Plot training history
        plt.figure(figsize=(8, 5))
        plt.plot(history_scratch['train_loss'])
        plt.title('Training Loss (Scratch)', fontweight="bold")
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(history_scratch['test_relative_l2'])
        plt.title('Test Relative L2 Error (Scratch)', fontweight="bold")
        plt.xlabel('Epoch')
        plt.ylabel('Relative L2 Error (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
        
        if config.SAVE_SCRATCH_MODEL:
            torch.save(model_scratch.state_dict(), f"models/fno_{config.MODES}_bn_scratch.pth")
            print(f"Saved scratch model to models/fno_{config.MODES}_bn_scratch.pth")
    else:
        print("Loading scratch model from file (TRAIN_SCRATCH=False)...")
        model_scratch = FNO1d_bn(config.MODES, config.WIDTH).to(config.DEVICE)
        try:
            model_scratch.load_state_dict(torch.load(f"models/fno_{config.MODES}_bn_scratch.pth", map_location=torch.device(config.DEVICE)))
            print(f"Loaded models/fno_{config.MODES}_bn_scratch.pth")
        except FileNotFoundError:
            print(f"File models/fno_{config.MODES}_bn_scratch.pth not found. Cannot test.")
            return None

    print("\n--- Testing Model Trained from Scratch ---")
    _, errors_scratch = test_at_all_times(config, model_scratch, finetune=True)
    print("Scratch errors:", errors_scratch)
    return errors_scratch

def run_experiment_finetuning(config):
    print("Using device:", config.DEVICE)
    
    time_steps = None
    errors_zeroshot = None
    errors_finetuned = None
    errors_scratch = None
    model_pretrained = None

    # 1. Zero-shot Test
    if config.RUN_ZERO_SHOT or config.RUN_FINETUNE: # We need pretrained model for finetuning too
        time_steps, errors_zeroshot, model_pretrained = run_zero_shot(config)

    # Load data if needed for training
    if config.RUN_FINETUNE or config.RUN_SCRATCH:
        training_set, testing_set = create_dataloaders(config, finetune=True)

    # 2. Finetune Model
    if config.RUN_FINETUNE:
        errors_finetuned = run_finetuning(config, training_set, testing_set, model_pretrained)

    # 3. Train from Scratch
    if config.RUN_SCRATCH:
        errors_scratch = run_scratch(config, training_set, testing_set)

    # 4. Plot Results
    if config.PLOT_RESULTS and time_steps is not None:
        plt.figure(figsize=(10, 6))
        if errors_zeroshot is not None:
            plt.plot(time_steps, errors_zeroshot, marker='o', label='Zero-shot (Pre-trained)')
        if errors_finetuned is not None:
            plt.plot(time_steps, errors_finetuned, marker='s', label='Finetuned')
        if errors_scratch is not None:
            plt.plot(time_steps, errors_scratch, marker='^', label='Trained from Scratch')
        
        plt.title('Relative L2 Error Comparison on Unknown Data', fontweight="bold")
        plt.xlabel('Time Steps')
        plt.xticks(time_steps)
        plt.ylabel('Relative L2 Error (%)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

def main():
    config = Config_finetune()
    run_experiment_finetuning(config)

if __name__ == "__main__":
    main()

