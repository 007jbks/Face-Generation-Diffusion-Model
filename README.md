# Face-Generation-Diffusion-Model
A diffusion model built from scratch in Pytorch and trained on facial data to make synthetic faces

# How to use
```python

# 1. Load the trained model (if not already in memory or if running separately)
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
sample_image = torch.randn(1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)

# 3. Iterate backward through timesteps to denoise
with torch.no_grad():
    for i in reversed(range(0, TIMESTEPS)): # Iterate from T-1 down to 0
        current_t_tensor = torch.full((1,), i, dtype=torch.long, device=device) # Timestep for this step
        sample_image = backward_diffusion_sample(
            sample_image, current_t_tensor, loaded_model, betas, TIMESTEPS, device
        )
        if i % 100 == 0 or i == 0:
            print(f"Denoising at timestep {i}")

# 4. Un-normalize and save/display the generated image
generated_image_np = sample_image.cpu().squeeze(0).permute(1, 2, 0).numpy() # (C, H, W) -> (H, W, C)
generated_image_np = (generated_image_np * 0.5 + 0.5) * 255 # Un-normalize to [0, 255]
generated_image_np = generated_image_np.astype(np.uint8) # Convert to uint8

# Create a PIL Image and save it
output_image_path = "generated_face.png"
Image.fromarray(generated_image_np).save(output_image_path)
print(f"Generated image saved to {output_image_path}")

```


 # *Sample Output* 
![generated_face](https://github.com/user-attachments/assets/663a2eb8-d548-44ae-93b0-86f40323e236)
It can be clearly observed that the model can successfully generate facial features such as eyes, nose and mouth however suffers to make the face and head shape coherent.

