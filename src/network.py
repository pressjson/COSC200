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
    batch_size=64,
    num_epochs=501,
    learning_rate=0.001,
    lr_step_size=100,
    lr_gamma=0.5,
    val_split=0.1,
    num_workers=4,
    use_data_parallel=True,
    use_amp=True,
    checkpoint_dir="checkpoints",
    log_interval=50,  # Print training progress every N batches
    save_interval=50,
):
    """
    Trains the ImageEnhancementNet model without using tqdm.
    (Args documentation remains the same)
    """

    # --- Setup ---
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoint directory: {checkpoint_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # AMP Scaler (Use updated torch.amp syntax)
    # Note: device.type works for 'cuda' or 'cpu'. Adjust if using other devices like 'mps'.
    scaler = torch.amp.GradScaler(
        device_type=device.type, enabled=(use_amp and device.type == "cuda")
    )
    print(
        f"Automatic Mixed Precision (AMP): {'Enabled' if scaler.is_enabled() else 'Disabled'}"
    )

    # --- Dataset and Dataloaders ---
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
    )

    try:
        full_dataset = ImageDataset(root_dir=data_dir, transform=transform)
    except ValueError as e:
        print(f"Error initializing dataset: {e}")
        return  # Exit if dataset is invalid

    if len(full_dataset) == 0:  # Should be caught by ValueError now, but keep as safety
        print("Error: Dataset is empty after initialization.")
        return

    # Split dataset
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    if train_size <= 0 or val_size <= 0:  # Ensure both splits are viable
        print(
            f"Warning: Dataset size ({len(full_dataset)}) too small for validation split ({val_split}). Using entire dataset for training."
        )
        train_dataset = full_dataset
        val_dataset = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            # Consider drop_last=True if batch size doesn't divide dataset size evenly
            # drop_last=True
        )
        val_loader = None
        total_train_batches = len(train_loader)
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
            # drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,  # Use same batch size or a different one for validation
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        total_train_batches = len(train_loader)
        total_val_batches = len(val_loader)

    # --- Model, Loss, Optimizer, Scheduler ---
    model = ImageEnhancementNet().to(device)

    # Handle DataParallel or single GPU
    model_to_save = model  # Keep ref to original model for saving state_dict
    if use_data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using nn.DataParallel on {torch.cuda.device_count()} GPUs.")
        model_to_save = model.module  # Get underlying model for saving
    elif use_data_parallel and device.type == "cuda":
        print(
            "Warning: use_data_parallel=True but only 1 GPU available. Running on single GPU."
        )
    elif device.type == "cpu":
        use_data_parallel = False  # Cannot use DataParallel on CPU
        print("Running on CPU.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_step_size, gamma=lr_gamma
    )

    # --- Training Loop ---
    best_val_loss = float("inf")
    start_epoch = 1

    # TODO: Add checkpoint loading logic here if needed
    # E.g., check for latest checkpoint, load state_dicts for model, optimizer, scheduler, epoch, best_val_loss

    print("\n--- Starting Training ---")
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        # Removed tqdm wrapper

        print(
            f"\n--- Epoch {epoch}/{num_epochs} [Train] ---"
        )  # Indicate start of training epoch
        batch_start_time = time.time()  # For estimating time per batch/log interval

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )  # Use set_to_none=True for potential efficiency gain

            # Forward pass with AMP context (Use updated torch.amp syntax)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=scaler.is_enabled(),
            ):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Check for NaN loss
            if math.isnan(loss.item()):
                print(
                    f"Warning: NaN loss detected at epoch {epoch}, batch {i+1}/{total_train_batches}. Skipping batch gradients."
                )
                # Don't step optimizer or scaler if loss is NaN
                continue  # Skip backward and step

            # Backward pass & Optimize (using scaler)
            scaler.scale(loss).backward()
            # Optional: Gradient clipping
            # scaler.unscale_(optimizer) # Unscale gradients before clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            batch_loss = loss.item()
            running_loss += batch_loss

            # Log intermediate training progress
            if (i + 1) % log_interval == 0 or (i + 1) == total_train_batches:
                batches_processed_interval = (
                    log_interval
                    if (i + 1) % log_interval == 0
                    else (i + 1) % log_interval
                )
                current_avg_loss = running_loss / (i + 1)
                time_elapsed_interval = time.time() - batch_start_time
                samples_per_sec = (
                    (batches_processed_interval * batch_size) / time_elapsed_interval
                    if time_elapsed_interval > 0
                    else 0
                )

                print(
                    f"  Batch {i+1}/{total_train_batches} | "
                    f"Loss: {batch_loss:.4f} | "
                    f"Avg Loss (Epoch): {current_avg_loss:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                    f"Speed: {samples_per_sec:.2f} samples/sec"
                )
                batch_start_time = time.time()  # Reset timer for next interval

        # Calculate average training loss for the epoch
        avg_train_loss = (
            running_loss / total_train_batches if total_train_batches > 0 else 0.0
        )

        # --- Validation Step ---
        avg_val_loss = float("inf")  # Default if no validation
        if val_loader:
            print(
                f"--- Epoch {epoch}/{num_epochs} [Validate] ---"
            )  # Indicate start of validation
            model.eval()
            val_loss = 0.0
            val_start_time = time.time()
            with torch.no_grad():
                # Removed tqdm wrapper
                for j, (inputs_val, targets_val) in enumerate(val_loader):
                    inputs_val, targets_val = inputs_val.to(device), targets_val.to(
                        device
                    )

                    # Forward pass - AMP context can optionally be used for inference too,
                    # but usually not necessary and torch.no_grad() is the main thing.
                    # with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                    outputs_val = model(inputs_val)
                    loss_val = criterion(outputs_val, targets_val)

                    if math.isnan(loss_val.item()):
                        print(
                            f"Warning: NaN validation loss detected at epoch {epoch}, validation batch {j+1}/{total_val_batches}."
                        )
                        # Decide how to handle: skip batch value, assign high loss?
                        # For average calculation, skipping might be best.
                        continue

                    val_loss += loss_val.item()
                    # Removed set_postfix. Print summary after the loop.

            # Calculate average validation loss
            avg_val_loss = (
                val_loss / total_val_batches if total_val_batches > 0 else 0.0
            )
            val_duration = time.time() - val_start_time
            print(
                f"  Validation complete in {val_duration:.2f}s. Avg Val Loss: {avg_val_loss:.4f}"
            )

        # --- Logging & Checkpointing ---
        epoch_duration = time.time() - epoch_start_time
        print("-" * 50)  # Separator line
        print(
            f"Epoch {epoch}/{num_epochs} Summary | "
            f"Duration: {epoch_duration:.2f}s | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "  # Display 'inf' or a placeholder if no validation
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )
        print("-" * 50)

        # Step the scheduler after validation and logging
        scheduler.step()

        # Prepare checkpoint data
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),  # Save the underlying model state
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,  # Save calculated avg_val_loss
        }

        # Save checkpoint periodically
        if epoch % save_interval == 0:
            save_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save(checkpoint_data, save_path)
            print(f"Checkpoint saved: {save_path}")

        # Save checkpoint if validation loss improved (only if validation is performed)
        if val_loader and avg_val_loss < best_val_loss:
            if not math.isinf(avg_val_loss):  # Ensure avg_val_loss is a valid number
                best_val_loss = avg_val_loss
                save_path = os.path.join(checkpoint_dir, "checkpoint_best.pth")
                torch.save(checkpoint_data, save_path)
                print(
                    f"*** Best validation loss ({best_val_loss:.4f}) achieved. Checkpoint saved: {save_path} ***"
                )
            else:
                print(
                    "Warning: Skipping best model save due to non-finite validation loss."
                )

    # --- Final Save ---
    final_save_path = os.path.join(checkpoint_dir, "final_model.pth")
    # Ensure checkpoint_data exists even if loop didn't run (e.g., num_epochs=0)
    # It will contain the state after the last completed epoch or initial state if start_epoch > num_epochs
    if "checkpoint_data" in locals():
        torch.save(checkpoint_data, final_save_path)
        print(f"\nTraining completed. Final model state saved: {final_save_path}")
    else:
        print(
            "\nTraining loop did not run (num_epochs might be 0 or less). No final model saved."
        )


if __name__ == "__main__":
    # Example usage remains the same, but output will be different (no progress bars)
    train_model(
        data_dir="../data/newchunks",  # Make sure this path is correct
        num_epochs=500,
        batch_size=128,
        learning_rate=0.003,
        lr_step_size=50,  # Adjusted step size
        lr_gamma=0.5,
        num_workers=8,  # Adjust based on your CPU cores
        checkpoint_dir="enhanced_model_checkpoints_no_tqdm",
        log_interval=20,  # Log more frequently if desired
        use_data_parallel=True,
        use_amp=True,  # Ensure AMP works correctly with your setup
        save_interval=25,  # Adjusted save interval
    )
