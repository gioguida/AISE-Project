import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from FNO import FNO1d

torch.manual_seed(0)
np.random.seed(0)

class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MODES = 16
    WIDTH = 64

    LR_ADAM = 0.001
    EPOCHS_ADAM = 200
    # StepLR scheduler parameters
    STEP_SIZE_SCHEDULER = 50
    GAMMA_SCHEDULER = 0.5
    # ReduceLROnPlateau scheduler parameters
    FACTOR_SCHEDULER = 0.5          # Multiply LR by 0.5
    PATIENCE_SCHEDULER = 5          # Wait 5 epochs before reducing
    MIN_LR_SCHEDULER = 1e-6  

    BATCH_SIZE = 20

    RE_TRAIN = False  # Whether to retrain the model or load existing weights
    SAVE_MODEL = True # Whether to save the model after training


# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)


def create_dataloaders(config):
    
    train_dataset = torch.from_numpy(np.load("data/data_train_128.npy")).type(torch.float32)
    test_dataset = torch.from_numpy(np.load("data/data_test_128.npy")).type(torch.float32)

    u0_train = train_dataset[:, 0, :] 
    u0_test = test_dataset[:, 0, :]
    u1_train = train_dataset[:, -1, :]
    u1_test = test_dataset[:, -1, :]

    grid_size = u0_train.shape[1]
    grid = torch.linspace(0, 1, grid_size).reshape(1, grid_size, 1)

    u0_train_with_grid = torch.cat([u0_train.unsqueeze(-1), grid.repeat(u0_train.shape[0], 1, 1)], dim=-1)
    u0_test_with_grid = torch.cat([u0_test.unsqueeze(-1), grid.repeat(u0_test.shape[0], 1, 1)], dim=-1)

    u0_train_with_grid = u0_train_with_grid.to(config.DEVICE)
    u0_test_with_grid = u0_test_with_grid.to(config.DEVICE)
    u1_train = u1_train.to(config.DEVICE)
    u1_test = u1_test.to(config.DEVICE)

    training_set = DataLoader(TensorDataset(u0_train_with_grid, u1_train), batch_size=config.BATCH_SIZE, shuffle=True)
    test_set = DataLoader(TensorDataset(u0_test_with_grid, u1_test), batch_size=config.BATCH_SIZE, shuffle=False)

    return training_set, test_set


def create_optimizer_and_scheduler(model, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR_ADAM, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.STEP_SIZE_SCHEDULER, gamma=config.GAMMA_SCHEDULER)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',           # Minimize validation loss
        factor=config.FACTOR_SCHEDULER,         # Multiply LR by 0.5
        patience=config.PATIENCE_SCHEDULER,          # Wait 5 epochs before reducing
        min_lr=config.MIN_LR_SCHEDULER         # Don't go below this
    )
    return optimizer, scheduler


def train_fno(config, verbose=True):
    device = config.DEVICE
    epochs = config.EPOCHS_ADAM

    training_set, test_set = create_dataloaders(config)
    fno = FNO1d(config.MODES, config.WIDTH).to(device)  # model

    optimizer, scheduler = create_optimizer_and_scheduler(fno, config)

    # Training loop
    history = {'train_loss': [], 'test_relative_l2': []}
    l = torch.nn.MSELoss()
    freq_print = 1

    for epoch in range(epochs):
        fno.train()
        train_mse = 0.0
        for step, (input_batch, output_batch) in enumerate(training_set):
            optimizer.zero_grad()
            input_batch = input_batch.to(device)
            output_batch = output_batch.to(device)
            output_pred_batch = fno(input_batch).squeeze(2)
            loss_f = l(output_pred_batch, output_batch)
            loss_f.backward()
            optimizer.step()
            train_mse += loss_f.item()
            history['train_loss'].append(loss_f.item())
        train_mse /= len(training_set)

        with torch.no_grad():
            fno.eval()
            test_relative_l2 = 0.0
            for step, (input_batch, output_batch) in enumerate(test_set):
                output_pred_batch = fno(input_batch).squeeze(2)
                loss_f = (torch.mean((output_pred_batch - output_batch) ** 2) / torch.mean(output_batch ** 2)) ** 0.5 * 100
                test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(test_set)
            history['test_relative_l2'].append(test_relative_l2)
        
        scheduler.step(test_relative_l2)

        if verbose and epoch % freq_print == 0:
            print("######### Epoch:", epoch, " ######### Train Loss:", train_mse, " ######### Relative L2 Test Norm:", test_relative_l2)

    return fno, history

def save_model(model, path):
    torch.save(model.state_dict(), path)

def eval_model(model, config):
    """Evaluate model and compute L2 relative error on data_test_128.npy only at final time step."""
    _, test_set = create_dataloaders(config)
    device = config.DEVICE
    model.eval()
    relative_l2 = 0.0
    with torch.no_grad():
        for step, (input_batch, output_batch) in enumerate(test_set):
            input_batch = input_batch.to(device)
            output_batch = output_batch.to(device)
            output_pred_batch = model(input_batch).squeeze(2)
            loss_f = (torch.mean((output_pred_batch - output_batch) ** 2) / torch.mean(output_batch ** 2)) ** 0.5 * 100
            relative_l2 += loss_f.item()
        relative_l2 /= len(test_set)
    return relative_l2

def run_experiment(config):
    if config.RE_TRAIN:
        print("Training model from scratch...")
        model, history = train_fno(config)
        if config.SAVE_MODEL:
            save_model(model, f"models/fno_{config.MODES}_model.pth")

        # Plot training history
        plt.figure(figsize=(8, 5))
        plt.plot(history['train_loss'])
        plt.title('Training Loss', fontweight="bold")
        plt.xlabel('Iteration')
        plt.ylabel('MSE Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(history['test_relative_l2'])
        plt.title('Test Relative L2 Error', fontweight="bold")
        plt.xlabel('Epoch')
        plt.ylabel('Relative L2 Error (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
    else:
        model = FNO1d(config.MODES, config.WIDTH).to(config.DEVICE)
        model.load_state_dict(torch.load(f"models/fno_{config.MODES}_model.pth", map_location=torch.device(config.DEVICE)))
    test_relative_l2 = eval_model(model, config)
    print("Final Test Relative L2 Error: ", test_relative_l2)

    

def main():
    config = Config()
    print("Using device:", config.DEVICE)
    run_experiment(config)

if __name__ == "__main__":
    main()