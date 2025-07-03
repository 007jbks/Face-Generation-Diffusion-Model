# Face-Generation-Diffusion-Model
A diffusion model built from scratch in Pytorch and trained on facial data to make synthetic faces

# How to use
```python
# --- Saving the trained model ---
# It's crucial to save the model after training.
MODEL_SAVE_PATH = "model.pth"
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# --- Image Generation (Evaluation) ---
print("\n--- Generating a sample image after training ---")

# 1. Load the trained model (if not already in memory or if running separately)
# Always instantiate the model architecture first
loaded_model = UNet(
    in_channels=IMAGE_CHANNELS,
    out_channels=IMAGE_CHANNELS,
    time_embedding_dim=64 * 8, # Must match the trained model's config
    initial_filters=64
).to(device)

# Load the saved state_dict
loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
loaded_model.eval() # Set to evaluation mode

# 2. Start with pure noise
# Generate one random noise image to start the reverse diffusion
sample_image = torch.randn(1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)

# 3. Iterate backward through timesteps to denoise
print(f"Starting image generation from pure noise (timestep {TIMESTEPS-1} down to 0)...")
with torch.no_grad():
    for i in reversed(range(0, TIMESTEPS)): # Iterate from T-1 down to 0
        current_t_tensor = torch.full((1,), i, dtype=torch.long, device=device) # Timestep for this step
        sample_image = backward_diffusion_sample(
            sample_image, current_t_tensor, loaded_model, betas, TIMESTEPS, device
        )
        # Optional: Print progress or save intermediate images
        if i % 100 == 0 or i == 0:
            print(f"Denoising at timestep {i}")

# 4. Un-normalize and save/display the generated image
# The generated image is currently in [-1, 1] range. Convert to [0, 255] for saving.
# Inverse of transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) is:
# value = (value * std) + mean
# For mean=0.5, std=0.5: value = (value * 0.5) + 0.5 which maps [-1, 1] to [0, 1]
# Then multiply by 255 for [0, 255]
generated_image_np = sample_image.cpu().squeeze(0).permute(1, 2, 0).numpy() # (C, H, W) -> (H, W, C)
generated_image_np = (generated_image_np * 0.5 + 0.5) * 255 # Un-normalize to [0, 255]
generated_image_np = generated_image_np.astype(np.uint8) # Convert to uint8

# Create a PIL Image and save it
output_image_path = "generated_face.png"
Image.fromarray(generated_image_np).save(output_image_path)
print(f"Generated image saved to {output_image_path}")

# You can also display it if you are in an environment like Jupyter/Colab
# import matplotlib.pyplot as plt
# plt.imshow(generated_image_np)
# plt.title("Generated Face")
# plt.axis('off')
# plt.show()
```


*Sample Output* \n
![generated_face](https://github.com/user-attachments/assets/663a2eb8-d548-44ae-93b0-86f40323e236)\n
It can be clearly observed that the model can successfully generate facial features such as eyes, nose and mouth however suffers to make the face and head shape coherent.

