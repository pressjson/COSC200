#!/usr/bin/env python3


import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import time
import math  # For isnan check

# thanks perplexity (;


class ImageDataset(Dataset):
    def __init__(self, root_dir="../data/chunks", transform=None):
        self.lq_images = []
        self.hq_images = []
        self.transform = transform
        self.root_dir = root_dir

        hq_files = {}
        lq_files = {}

        for subdirectory in sorted(os.listdir(root_dir)):
            for filename in sorted(os.listdir(os.path.join(root_dir, subdirectory))):
                if "hq" in filename:
                    self.hq_images.append(
                        os.path.join(root_dir, subdirectory, filename)
                    )
                else:
                    self.lq_images.append(
                        os.path.join(root_dir, subdirectory, filename)
                    )
        for base_name, hq_path in hq_files.items():
            if base_name in lq_files:
                self.hq_images.append(hq_path)
                self.lq_images.append(lq_files[base_name])
            else:
                print(f"Warning: HQ image {hq_path} found without matching LQ image.")

        print(f"Found {len(self.hq_images)} paired images.")
        if len(self.hq_images) == 0:
            raise ValueError(
                f"No paired images found in {root_dir}. Check file naming and structure."
            )

    def __len__(self):
        return len(self.hq_images)

    def __getitem__(self, i):
        lq_image = Image.open(self.lq_images[i]).convert("RGB")
        hq_image = Image.open(self.hq_images[i]).convert("RGB")
        if self.transform:
            lq_image = self.transform(lq_image)
            hq_image = self.transform(hq_image)
        return lq_image, hq_image


class ImageEnhancementNet(nn.Module):
    # first model

    def __init__(self):
        super(ImageEnhancementNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

    # second model
    # def __init__(self):
    #     super(ImageEnhancementNet, self).__init__()
    #     self.conv1 = nn.Conv2d(3, 64, kernel_size=11, padding=5)
    #     self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
    #     self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
    #     self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
    #     self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
    #     self.conv6 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
    #     self.relu = nn.ReLU(inplace=True)
    #     self.bn1 = nn.BatchNorm2d(64)
    #     self.bn2 = nn.BatchNorm2d(128)
    #     self.bn3 = nn.BatchNorm2d(256)
    #     self.bn4 = nn.BatchNorm2d(128)
    #     self.bn5 = nn.BatchNorm2d(64)

    # def forward(self, x):
    #     residual = x
    #     x = self.relu(self.bn1(self.conv1(x)))
    #     x = self.relu(self.bn2(self.conv2(x)))
    #     x = self.relu(self.bn3(self.conv3(x)))
    #     x = self.relu(self.bn4(self.conv4(x)))
    #     x = self.relu(self.bn5(self.conv5(x)))
    #     x = self.conv6(x)
    #     return x + residual
    #


# chatgpt's code


def train_model(
    data_dir="../data/chunks",
    image_size=(256, 256),
    batch_size=64,  # Reduced from 128 for potentially lower memory
    num_epochs=501,
    learning_rate=0.001,  # Often start a bit lower than 0.003
    lr_step_size=100,  # Decay LR every N epochs
    lr_gamma=0.5,  # Factor to decay LR by
    val_split=0.1,  # Fraction of data for validation
    num_workers=4,  # Number of CPU cores for data loading
    use_data_parallel=True,
    use_amp=True,  # Use Automatic Mixed Precision
    checkpoint_dir="checkpoints",
    log_interval=50,  # Print training loss every N batches
    save_interval=50,  # Save checkpoint every N epochs
):
    """
    Trains the ImageEnhancementNet model.

    Args:
        data_dir (str): Path to the root directory of the dataset chunks.
        image_size (tuple): Target size (height, width) for images.
        batch_size (int): Number of samples per batch.
        num_epochs (int): Total number of training epochs.
        learning_rate (float): Initial learning rate for the optimizer.
        lr_step_size (int): Period of learning rate decay.
        lr_gamma (float): Multiplicative factor of learning rate decay.
        val_split (float): Fraction of the dataset to use for validation.
        num_workers (int): How many subprocesses to use for data loading.
        use_data_parallel (bool): Whether to use nn.DataParallel for multi-GPU training.
        use_amp (bool): Whether to use Automatic Mixed Precision (requires CUDA).
        checkpoint_dir (str): Directory to save checkpoints.
        log_interval (int): Print training stats every `log_interval` batches.
        save_interval (int): Save a checkpoint every `save_interval` epochs.
    """

    # --- Setup ---
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoint directory: {checkpoint_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # AMP Scaler (only if using CUDA and AMP enabled)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
    print(
        f"Automatic Mixed Precision (AMP): {'Enabled' if scaler.is_enabled() else 'Disabled'}"
    )

    # --- Dataset and Dataloaders ---
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            # Add more augmentations here if desired (e.g., RandomHorizontalFlip)
            # transforms.RandomHorizontalFlip(),
        ]
    )

    full_dataset = ImageDataset(root_dir=data_dir, transform=transform)

    if len(full_dataset) == 0:
        print(
            "Error: Dataset is empty. Please check the data directory and image pairing."
        )
        return

    # Split dataset
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    if train_size == 0 or val_size == 0:
        print(
            "Warning: Dataset size is too small for validation split. Using entire dataset for training."
        )
        train_dataset = full_dataset
        val_dataset = None  # No validation possible
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = None
    else:
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        print(
            f"Dataset split: {train_size} training samples, {val_size} validation samples"
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # No shuffle for validation
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

    # --- Model, Loss, Optimizer, Scheduler ---
    model = ImageEnhancementNet().to(device)

    if use_data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using nn.DataParallel on {torch.cuda.device_count()} GPUs.")
    elif use_data_parallel and device.type == "cuda":
        print("Warning: use_data_parallel=True but only 1 GPU available.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_step_size, gamma=lr_gamma
    )

    # --- Training Loop ---
    best_val_loss = float("inf")
    start_epoch = 1  # Assuming starting from scratch

    # TODO: Add checkpoint loading logic here if needed

    print("\n--- Starting Training ---")
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        processed_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", ncols=100)
        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # Forward pass with AMP context
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Check for NaN loss
            if math.isnan(loss.item()):
                print(
                    f"Warning: NaN loss detected at epoch {epoch}, batch {i}. Skipping batch."
                )
                optimizer.zero_grad()  # Ensure grads are zero before next batch
                continue

            # Backward pass & Optimize (using scaler)
            scaler.scale(loss).backward()
            # Optional: Gradient clipping can help stabilize training
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            batch_loss = loss.item()
            running_loss += batch_loss
            processed_batches += 1

            # Log intermediate training loss
            if (i + 1) % log_interval == 0 or (i + 1) == len(train_loader):
                pbar.set_postfix(
                    {
                        "Batch Loss": f"{batch_loss:.4f}",
                        "Avg Loss": f"{running_loss / processed_batches:.4f}",
                        "LR": f"{scheduler.get_last_lr()[0]:.6f}",
                    }
                )

        # Calculate average training loss for the epoch
        avg_train_loss = (
            running_loss / processed_batches if processed_batches > 0 else 0.0
        )

        # --- Validation Step ---
        avg_val_loss = float("inf")  # Default if no validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                pbar_val = tqdm(
                    val_loader,
                    desc=f"Epoch {epoch}/{num_epochs} [Val]",
                    ncols=100,
                    leave=False,
                )
                for inputs_val, targets_val in pbar_val:
                    inputs_val, targets_val = inputs_val.to(device), targets_val.to(
                        device
                    )
                    # No need for AMP context manager in torch.no_grad() for inference
                    outputs_val = model(inputs_val)
                    loss_val = criterion(outputs_val, targets_val)
                    val_loss += loss_val.item()
                    pbar_val.set_postfix({"Val Loss": f"{loss_val.item():.4f}"})

            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0

        # --- Logging & Checkpointing ---
        epoch_duration = time.time() - epoch_start_time
        print(
            f"Epoch {epoch}/{num_epochs} Summary | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"Duration: {epoch_duration:.2f}s"
        )

        # Step the scheduler
        scheduler.step()

        # Prepare checkpoint data
        # Handle DataParallel: state_dict is under model.module
        model_state_dict = (
            model.module.state_dict()
            if isinstance(model, nn.DataParallel)
            else model.state_dict()
        )
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }

        # Save checkpoint periodically
        if epoch % save_interval == 0:
            save_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save(checkpoint_data, save_path)
            print(f"Checkpoint saved: {save_path}")

        # Save checkpoint if validation loss improved (only if validation is performed)
        if val_loader and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(checkpoint_dir, "checkpoint_best.pth")
            torch.save(checkpoint_data, save_path)
            print(
                f"*** Best validation loss ({best_val_loss:.4f}) achieved. Checkpoint saved: {save_path} ***"
            )

    # --- Final Save ---
    final_save_path = os.path.join(checkpoint_dir, "final_model.pth")
    final_checkpoint_data = checkpoint_data  # Use the last epoch's data
    torch.save(final_checkpoint_data, final_save_path)
    print(f"\nTraining completed. Final model state saved: {final_save_path}")


if __name__ == "__main__":
    # Example usage with some parameters adjusted
    train_model(
        num_epochs=500,  # Reduced for quicker testing
        batch_size=128,  # Smaller batch size if memory is limited
        learning_rate=0.003,
        lr_step_size=20,
        lr_gamma=0.5,
        num_workers=4,  # Adjust based on your CPU cores
        checkpoint_dir="enhanced_model_checkpoints",
        log_interval=10,
        use_data_parallel=True,
        save_interval=5,
    )
