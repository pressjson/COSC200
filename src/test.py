#!/usr/bin/env python3

import network, stitcher, imslp_bootstrap as bootstrap, chunker
import settings

from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch
import os
import re
import sys  # For exiting on error
import shutil


def upscale_file(
    input_file_directory,
    input_file_name,
    model_name="enhanced_model_checkpoints_no_tqdm/checkpoint_best.pth",
    multiple_gpus=False,
):  # Example using a checkpoint

    # --- Define Key Paths (Relative to script location as per original) ---
    data_base_dir = "../data"
    test_dir = os.path.join(data_base_dir, "test")
    if os.path.exists(test_dir):
        print(f"Test directory {test_dir} exists, recreating new test directory")
        shutil.rmtree(test_dir)
        os.makedirs(test_dir)
    test_images_dir = os.path.join(test_dir, "images")
    test_chunks_dir = os.path.join(test_dir, "chunks")
    test_upscaled_dir = os.path.join(
        test_chunks_dir, "upscaled"
    )  # Upscaled chunks go here
    test_stitched_dir = os.path.join(test_dir, "stitched")  # Final output pages

    input_pdf_path = os.path.join(input_file_directory, input_file_name)

    # --- Basic Input Checks ---
    if not os.path.isfile(input_pdf_path):
        print(f"Error: Input PDF not found at {input_pdf_path}")
        sys.exit(1)
    if not os.path.isfile(model_name):
        print(f"Error: Model checkpoint not found at {model_name}")
        sys.exit(1)

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model Correctly ---
    print(f"Loading model state dict from: {model_name}")
    # 1. Instantiate the base model
    model = network.ImageEnhancementNet()

    # 2. Load the state dictionary (assuming train_model saved the underlying model's state)
    #    Load to CPU first to avoid potential GPU mismatch issues during loading
    checkpoint = torch.load(model_name, map_location=torch.device("cpu"))

    #    Extract the model state dict (handle checkpoints saved by the improved train_model)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"Loaded state dict from epoch {checkpoint.get('epoch', 'N/A')}")
    else:
        # Assume it's a raw state_dict file (older save method)
        state_dict = checkpoint
        print("Loaded raw state dict (epoch information not found in checkpoint).")

    # Optional: Clean keys if they were saved with 'module.' prefix (e.g., from saving DataParallel directly)
    # This shouldn't be needed if train_model saved model.module.state_dict() correctly
    cleaned_state_dict = {}
    prefix = "module."
    needs_cleaning = any(key.startswith(prefix) for key in state_dict.keys())
    if needs_cleaning:
        print("State dict keys have 'module.' prefix, cleaning...")
        for k, v in state_dict.items():
            if k.startswith(prefix):
                cleaned_state_dict[k[len(prefix) :]] = v
            else:
                cleaned_state_dict[k] = v  # Keep non-prefixed keys if any
        state_dict = cleaned_state_dict

    # 3. Load the state dict into the base model
    model.load_state_dict(state_dict)
    print("Model state dict loaded successfully.")

    # 4. Move model to the target device
    model = model.to(device)

    # 5. Wrap with DataParallel *after* loading state dict if using multiple GPUs
    if multiple_gpus:
        if device.type == "cuda" and torch.cuda.device_count() > 1:
            print(
                f"Wrapping model with DataParallel for {torch.cuda.device_count()} GPUs."
            )
            model = nn.DataParallel(model)
        elif device.type == "cuda":
            print("Running on single GPU.")
        else:
            print("Running on CPU.")
    else:
        if device.type == "cuda":
            print("Running on single GPU.")
        else:
            print("Running on CPU.")

    # 6. Set model to evaluation mode
    model.eval()
    # print(model) # Optional: print model structure

    # --- Define Transform ---
    # This should match the transform used during *validation* or *testing*,
    # usually the same as training minus data augmentation.
    transform = transforms.Compose(
        [
            # The model was trained on 256x256, so input chunks should be resized
            transforms.Resize(settings.IMAGE_SIZE),
            transforms.ToTensor(),  # Scales images to [0, 1]
        ]
    )

    # --- Step 1: Make images out of the PDF ---
    print("\n--- Converting PDF to images ---")
    print(f"Input PDF: {input_pdf_path}")
    print(f"Output directory: {test_images_dir}")
    os.makedirs(test_images_dir, exist_ok=True)  # Ensure output dir exists
    try:
        bootstrap.make_images(input_pdf_path, test_images_dir)
        print("PDF conversion complete.")
    except Exception as e:
        print(f"Error during PDF conversion using bootstrap.make_images: {e}")
        sys.exit(1)

    image_files = sorted(
        [
            f
            for f in os.listdir(test_images_dir)
            if os.path.isfile(os.path.join(test_images_dir, f))
        ]
    )
    if not image_files:
        print(f"Error: No image files found in {test_images_dir} after PDF conversion.")
        sys.exit(1)
    print(f"Found {len(image_files)} images from PDF.")

    # --- Step 2: Make chunks to pass to the model ---
    print("\n--- Creating image chunks ---")
    print(f"Chunk output base directory: {test_chunks_dir}")
    os.makedirs(test_chunks_dir, exist_ok=True)  # Ensure base chunk dir exists

    # Process pages sequentially based on sorted image file names
    for page_index, image_filename in enumerate(image_files):
        page_image_path = os.path.join(test_images_dir, image_filename)
        page_chunk_output_dir = os.path.join(
            test_chunks_dir, str(page_index)
        )  # Use index as page number
        print(
            f"Processing page {page_index}: {image_filename} -> {page_chunk_output_dir}"
        )
        os.makedirs(page_chunk_output_dir, exist_ok=True)
        try:
            # Assuming make_chunks creates files like 'lq_X_Y.jpg' or similar
            # The 'quality' argument might need adjustment based on chunker.py logic
            chunker.make_chunks(
                quality="lq",  # Or whatever designation your chunker uses
                input_file_location=page_image_path,
                output_dir=page_chunk_output_dir,
            )
        except Exception as e:
            print(
                f"Error creating chunks for page {page_index} ({image_filename}): {e}"
            )
            # Decide whether to continue with next page or exit
            # continue
            sys.exit(1)
    print("Chunk creation complete.")

    # --- Step 3: Use the model to upscale chunks ---
    print("\n--- Upscaling image chunks ---")
    os.makedirs(test_upscaled_dir, exist_ok=True)  # Ensure base upscaled dir exists

    processed_page_dirs = sorted(
        [
            d
            for d in os.listdir(test_chunks_dir)
            if os.path.isdir(os.path.join(test_chunks_dir, d)) and d.isdigit()
        ]
    )

    if not processed_page_dirs:
        print(f"Error: No valid page chunk directories found in {test_chunks_dir}")
        sys.exit(1)

    for page_subdir in processed_page_dirs:
        page_chunk_dir = os.path.join(test_chunks_dir, page_subdir)
        page_upscaled_output_dir = os.path.join(test_upscaled_dir, page_subdir)
        os.makedirs(page_upscaled_output_dir, exist_ok=True)

        print(f"Upscaling page {page_subdir}...")

        chunk_files = sorted(
            [
                f
                for f in os.listdir(page_chunk_dir)
                if os.path.isfile(os.path.join(page_chunk_dir, f))
            ]
        )
        if not chunk_files:
            print(
                f"Warning: No chunk files found in {page_chunk_dir}. Skipping page {page_subdir}."
            )
            continue

        for chunk_filename in chunk_files:
            chunk_input_path = os.path.join(page_chunk_dir, chunk_filename)

            # --- Parse chunk filename to get coordinates ---
            # Adjust regex if chunker.py uses a different naming convention
            # Example: assumes 'prefix_X_Y.jpg' or 'prefix_X_Y.png' etc.
            pattern = r"(\w+)_(\d+)_(\d+)\.\w+"  # Looser pattern for extension
            match = re.match(pattern, chunk_filename)

            if not match:
                print(
                    f"Warning: Could not parse coordinates from chunk filename '{chunk_filename}'. Skipping."
                )
                continue
            name_prefix, x_str, y_str = (
                match.groups()
            )  # prefix might be 'lq' or similar

            # --- Perform Upscaling ---
            try:
                # Load chunk, convert to RGB
                input_chunk_image = Image.open(chunk_input_path).convert("RGB")

                # Apply transform, add batch dimension, move to device
                input_tensor = transform(input_chunk_image).unsqueeze(0).to(device)

                # Run model inference
                with torch.no_grad():
                    output_tensor = model(input_tensor)  # Shape: [1, C, H, W]

                # Process output tensor: remove batch dim, move to CPU
                output_image_tensor = output_tensor.squeeze(0).cpu()

                # Clamp output to [0, 1] range before converting to PIL
                output_image_tensor = torch.clamp(output_image_tensor, 0, 1)

                # Convert tensor back to PIL Image
                # ToPILImage handles the scaling from [0, 1] float to [0, 255] uint8
                result_image = transforms.ToPILImage()(output_image_tensor)

                # --- Save Upscaled Chunk ---
                # Use a consistent naming scheme for stitched, e.g., "upscaled_X_Y.jpg"
                output_chunk_filename = f"upscaled_{x_str}_{y_str}.jpg"  # Saving as JPG
                output_chunk_path = os.path.join(
                    page_upscaled_output_dir, output_chunk_filename
                )
                result_image.save(output_chunk_path)

            except Exception as e:
                print(
                    f"Error processing chunk {chunk_filename} for page {page_subdir}: {e}"
                )
                # Decide whether to skip chunk or exit
                continue

        print(f"Finished upscaling page {page_subdir}.")
    print("Chunk upscaling complete.")

    # --- Step 4: Stitch upscaled chunks back together ---
    print("\n--- Stitching upscaled chunks ---")
    os.makedirs(test_stitched_dir, exist_ok=True)  # Ensure final output dir exists

    upscaled_page_dirs = sorted(
        [
            d
            for d in os.listdir(test_upscaled_dir)
            if os.path.isdir(os.path.join(test_upscaled_dir, d)) and d.isdigit()
        ]
    )

    if not upscaled_page_dirs:
        print(f"Error: No upscaled page directories found in {test_upscaled_dir}")
        sys.exit(1)

    for page_subdir in upscaled_page_dirs:
        upscaled_chunks_path = os.path.join(test_upscaled_dir, page_subdir)
        stitched_output_filename = (
            f"enhanced_page_{page_subdir}.jpg"  # Clearer output name
        )
        print(f"Stitching page {page_subdir} -> {stitched_output_filename}")

        try:
            # Assuming stitcher expects path to directory containing chunks named like 'upscaled_X_Y.jpg'
            stitcher.stitch_image(
                path=upscaled_chunks_path,
                name="upscaled",  # The prefix used in the saved upscaled chunks
                name_of_save=stitched_output_filename,
                save_path=test_stitched_dir,
            )
        except Exception as e:
            print(f"Error stitching page {page_subdir}: {e}")
            # Decide whether to continue or exit
            continue

    print(f"Stitching complete. Final images saved in: {test_stitched_dir}")
    print("--- Upscaling process finished ---")


# Example usage:
if __name__ == "__main__":
    # Make sure these paths are correct relative to where you run the script
    pdf_directory = "../"  # Example directory containing PDF
    pdf_filename = "moonlight_sonata.pdf"  # Example PDF name
    model_checkpoint = "../checkpoint_best.pth"  # Path to your best model

    # Create dummy input if it doesn't exist for testing
    if not os.path.exists(os.path.join(pdf_directory, pdf_filename)):
        print(
            f"Error loading {os.path.join(pdf_directory, pdf_filename)}: File does not exist"
        )
        sys.exit(1)

    upscale_file(pdf_directory, pdf_filename, model_checkpoint)  #!/usr/bin/env python3
