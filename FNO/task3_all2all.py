import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

from FNO_bn import FNO1d_bn

class Config_all2all:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODES = 16
    WIDTH = 64

    N_TRAIN = 1024
    BATCH_SIZE = 128

    LEARNING_RATE_ADAM = 0.001
    EPOCHS_ADAM = 200

    # StepLR scheduler parameters
    # STEP_SIZE = 10
    # GAMMA = 0.5

    # ReduceLROnPlateau scheduler parameters
    FACTOR_SCHEDULER = 0.5          # Multiply LR by 0.5
    PATIENCE_SCHEDULER = 5          # Wait 5 epochs before reducing
    MIN_LR_SCHEDULER = 1e-6  

    BATCH_SIZE_TEST = 200

    RE_TRAIN = False  # Whether to retrain the model or load existing weights
    SAVE_MODEL = False  # Whether to save the trained model

def check_data_statistics():
    train_data = np.load("data/data_train_128.npy")
    class _stats:
        def __init__(self, data):
            self.mean = np.mean(data)
            self.std = np.std(data)
            self.min = np.min(data)
            self.max = np.max(data)    
    stats = _stats(train_data)
    return stats

class PDEDataset(Dataset):
    def __init__(self,
                 stats,
                 which="training",
                 training_samples = 256,
                 resolution = 128,
                 device='cpu',
                 finetune=False
                 ):

        self.resolution = resolution
        self.device = device
        
        self.T = 5
        # Precompute all possible (t_initial, t_final) pairs within the specified range.
        self.time_pairs = [(i, j) for i in range(0, self.T) for j in range(i + 1, self.T)]
        self.len_times  = len(self.time_pairs)

        if not finetune:
            self.data = np.load(f"data/data_train_{resolution}.npy")
            total_samples = self.data.shape[0]
            self.n_val = 32
            self.n_test = 32
            
            if which == "training":
                self.length = training_samples * self.len_times
                self.start_sample = 0
            elif which == "validation":
                self.length = self.n_val * self.len_times
                self.start_sample = total_samples - self.n_val - self.n_test
            elif which == "test":
                self.length = self.n_test * self.len_times
                self.start_sample = total_samples - self.n_test
        else:
            # Finetuning case: separate files for train and val
            if which == "training":
                self.data = np.load(f"data/data_finetune_train_unknown_{resolution}.npy")
                training_samples = min(training_samples, self.data.shape[0])
                self.length = training_samples * self.len_times
                self.start_sample = 0
            elif which == "validation":
                self.data = np.load(f"data/data_finetune_val_unknown_{resolution}.npy")
                total_samples = self.data.shape[0]
                self.length = total_samples * self.len_times
                self.start_sample = 0
            elif which == "test":
                self.data = np.load(f"data/data_test_unknown_{resolution}.npy")
                total_samples = self.data.shape[0]
                self.length = total_samples * self.len_times
                self.start_sample = 0

        self.mean = stats.mean # 0.018484
        self.std  = stats.std # 0.685405
        
        # Pre-create grid to avoid recreating it each time
        self.grid = torch.linspace(0, 1, 128, dtype=torch.float32).reshape(128, 1).to(device)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        sample_idx = self.start_sample + index // self.len_times
        time_pair_idx = index % self.len_times
        t_inp, t_out = self.time_pairs[time_pair_idx]
        time = torch.tensor((t_out - t_inp)/4. + float(np.random.rand(1)[0]/10**6), dtype=torch.float32, device=self.device)

        inputs = torch.from_numpy(self.data[sample_idx, t_inp]).type(torch.float32).reshape(128, 1).to(self.device)
        inputs = (inputs - self.mean)/self.std #Normalize
        
        # Add grid coordinates (already on correct device and dtype)
        inputs = torch.cat((inputs, self.grid), dim=-1)  # (128, 2)

        outputs = torch.from_numpy(self.data[sample_idx, t_out]).type(torch.float32).reshape(128).to(self.device)
        outputs = (outputs - self.mean)/self.std #Normalize

        return time, inputs, outputs

def create_dataloaders(config, finetune=False):
    n_train = config.N_TRAIN
    batch_size = config.BATCH_SIZE
    device = config.DEVICE
    training_set = DataLoader(PDEDataset(check_data_statistics(),
                                        "training",
                                        n_train, 
                                        device=device, 
                                        finetune=finetune), 
                                batch_size=batch_size, 
                                shuffle=True)
    testing_set = DataLoader(PDEDataset(check_data_statistics(), 
                                        "validation", 
                                        device=device, 
                                        finetune=finetune), 
                                batch_size=batch_size, 
                                shuffle=False)
    return training_set, testing_set

def create_optimizer_and_scheduler(model, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE_ADAM, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.STEP_SIZE, gamma=config.GAMMA)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',           # Minimize validation loss
        factor=config.FACTOR_SCHEDULER,         # Multiply LR by 0.5
        patience=config.PATIENCE_SCHEDULER,          # Wait 5 epochs before reducing
        min_lr=config.MIN_LR_SCHEDULER         # Don't go below this
    )
    return optimizer, scheduler

def train_all2all(config, model, training_set, testing_set, save_model=True, verbose=True, freq_print=1):
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    # Define the error function
    def relative_l2_error(pred, true):
        diff_norm = torch.norm(pred - true, p=2, dim=1)
        true_norm = torch.norm(true, p=2, dim=1)
        return torch.mean(diff_norm / true_norm) * 100

    l = nn.MSELoss()  
    history = {'train_loss': [], 'test_relative_l2': []}
    for epoch in range(config.EPOCHS_ADAM):
        model.train()
        train_mse = 0.0
        for step, (time_batch, input_batch, output_batch) in enumerate(training_set):
            optimizer.zero_grad()
            output_pred_batch = model(input_batch, time_batch).squeeze(-1)
            loss_f = l(output_pred_batch, output_batch)
            loss_f.backward()
            optimizer.step()
            train_mse += loss_f.item()
        train_mse /= len(training_set)
        history['train_loss'].append(train_mse)

        with torch.no_grad():
            model.eval()
            test_relative_l2 = 0.0
            for step, (time_batch, input_batch, output_batch) in enumerate(testing_set):
                output_pred_batch = model(input_batch, time_batch).squeeze(-1)
                loss_f = relative_l2_error(output_pred_batch, output_batch)
                test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(testing_set)
        scheduler.step(test_relative_l2)
        history['test_relative_l2'].append(test_relative_l2)

        if verbose and epoch % freq_print == 0: 
            print("######### Epoch:", epoch, 
                  " ######### Train Loss:", train_mse, 
                  " ######### Relative L2 Test Norm:", test_relative_l2)
        
        # save trained model
    if save_model:
        torch.save(model.state_dict(), "models/fno1d_bn_model.pth")

    return model, history

def load_test_set_at_time(config, time, stats, finetune=False):
    ''' load test set at a specific time '''
    if time not in [0.25, 0.50, 0.75, 1.0]:
        raise ValueError("Time must be one of [0.25, 0.50, 0.75, 1.0]")
    time_index = int(time * 4)  # since we have 5 time steps (0, 0.25, 0.5, 0.75, 1.0)

    if not finetune:
        test_data = np.load("data/data_test_128.npy")
    else:
        test_data = np.load("data/data_test_unknown_128.npy")

    initial_conditions = test_data[:, 0, :]  # Shape: (n_samples, 128)
    final_time_solutions = test_data[:, time_index, :]  # Shape: (n_samples, 128)

    # Prepare initial conditions with grid coordinates
    spatial_resolution = initial_conditions.shape[1]
    grid = torch.linspace(0, 1, spatial_resolution, dtype=torch.float32).reshape(spatial_resolution, 1)
    initial_conditions_tensor = torch.from_numpy(initial_conditions).type(torch.float32).reshape(-1, spatial_resolution, 1)
    # Add grid coordinates to each sample
    initial_conditions_with_grid = torch.cat([initial_conditions_tensor,
                                            grid.repeat(initial_conditions_tensor.shape[0], 1, 1)],
                                            dim=-1)
    # Normalize initial conditions
    mean = stats.mean # 0.018484
    std = stats.std # 0.685405
    # Only normalize the first channel (function values)
    initial_conditions_with_grid[:, :, 0] = (initial_conditions_with_grid[:, :, 0] - mean) / std
    return initial_conditions_with_grid, final_time_solutions

def test_at_one_time(config, model, time, initial_conditions_with_grid, final_time_solutions):
    '''Test the model predictions from initial condition to a specific time'''
    # Load test set at the specified time
    test_data_raw = final_time_solutions  # Shape: (n_samples, 128)
    stats = check_data_statistics()

    # move to device
    initial_conditions_with_grid = initial_conditions_with_grid.to(config.DEVICE)

    relative_l2_error = 0.0
    n_test_samples = initial_conditions_with_grid.shape[0]

    time_tensor = torch.ones(initial_conditions_with_grid.shape[0], dtype=torch.float32) * time
    time_tensor = time_tensor.to(config.DEVICE)

    model.eval()
    all_predictions = []
    with torch.no_grad():
        for i in range(0, n_test_samples, config.BATCH_SIZE_TEST):
            input_batch = initial_conditions_with_grid[i:i+config.BATCH_SIZE_TEST, :, :].to(config.DEVICE)
            time_batch = time_tensor[i:i+config.BATCH_SIZE_TEST].to(config.DEVICE)
            output_pred_batch = model(input_batch, time_batch).squeeze(-1)  # Shape: (batch_size, 128)
            all_predictions.append(output_pred_batch.cpu())
            
    prediction = torch.cat(all_predictions, dim=0)

    # denormalize predictions
    prediction_denorm = prediction * stats.std + stats.mean

    # Get ground truth 
    ground_truth = test_data_raw
    ground_truth_tensor = torch.from_numpy(ground_truth).type(torch.float32)

    # Calculate relative L2 error
    rel_l2_error_t = torch.mean(torch.norm(prediction_denorm - ground_truth_tensor, dim=1) / 
                                torch.norm(ground_truth_tensor, dim=1)) * 100
    relative_l2_error = rel_l2_error_t.item()
    print(f"Relative L2 Error at t={time}: {relative_l2_error:.4f}%")
    return relative_l2_error

def test_at_all_times(config, model, finetune=False):
    '''Test the model predictions from initial condition to all times'''
    time_steps = [0.25, 0.50, 0.75, 1.0]
    relative_l2_errors = []
    for time in time_steps:
        stats = check_data_statistics()
        initial_conditions_with_grid, test_data_raw = load_test_set_at_time(config, time, stats, finetune=finetune)
        rel_l2_error = test_at_one_time(config, model, time, initial_conditions_with_grid, test_data_raw)
        relative_l2_errors.append(rel_l2_error)
    return time_steps, relative_l2_errors

def run_experiment_all2all(config):
    model = FNO1d_bn(config.MODES, config.WIDTH).to(config.DEVICE)  # model
    if config.RE_TRAIN:
        training_set, testing_set = create_dataloaders(config)
        model, history = train_all2all(config, model, training_set, testing_set, save_model=config.SAVE_MODEL, verbose=True)
        
        # Plot training history
        plt.figure(figsize=(8, 5))
        plt.plot(history['train_loss'])
        plt.title('Training Loss (All2All)')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(history['test_relative_l2'])
        plt.title('Test Relative L2 Error (All2All)')
        plt.xlabel('Epoch')
        plt.ylabel('Relative L2 Error (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
    else:
        model.load_state_dict(torch.load("models/fno1d_bn_model.pth", map_location=torch.device(config.DEVICE)))

    time_steps, relative_l2_errors = test_at_all_times(config, model)
    # plot relative L2 errors over time steps
    plt.figure(figsize=(8, 5))
    plt.plot(time_steps, relative_l2_errors, marker='o')
    plt.title('Relative L2 Error over Time Steps')
    plt.xlabel('Time Steps')
    plt.xticks(time_steps)
    plt.ylabel('Relative L2 Error (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    return time_steps, relative_l2_errors

def main():
    config = Config_all2all()
    print('using device:', config.DEVICE)
    run_experiment_all2all(config)

if __name__ == "__main__":
    main()