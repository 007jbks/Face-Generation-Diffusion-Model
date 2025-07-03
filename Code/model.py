import pandas as pd
import numpy as np
import torch
device = 'cuda'
import torch.nn.functional as F
iterations = 1000
beta0 = 0.0001 
betat = 0.02
betas = torch.linspace(beta0,betat,iterations)
def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t from a list of values vals,
    and reshapes it to match the shape of x for broadcasting.
    """
    # Ensure t is a tensor and has the correct device
    t_tensor = t.to(vals.device)
    batch_size = t_tensor.shape[0]
    out = vals.gather(-1, t_tensor.long())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t_tensor.device)

def forward_pass(x_0,betas,timesteps,device='cuda'):
    betas = betas.to(device)
    alphas = 1.-betas 
    alphas_cumprod = torch.cumprod(alphas, axis=0) # this is the cumulative product of all the aplhas
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    # Randomly sample a timestep for each image in the batch
    t = torch.randint(0, timesteps, (x_0.shape[0],), device=device).long()
    noise = torch.randn_like(x_0)

    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    noisy_image = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    return noisy_image, noise, t

from torch import nn
class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        # Calculate the 'denominators' for the sinusoidal frequencies
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # Apply the time to the frequencies
        embeddings = time[:, None] * embeddings[None, :]
        # Concatenate sine and cosine components
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if mid_channels is None:
            mid_channels = out_channels
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels) # Batch Normalization
        # Second convolutional layer
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) # Batch Normalization
        self.relu = nn.ReLU() # Activation function

    def forward(self, x):
        # Apply first conv -> bn -> relu
        x_processed = self.relu(self.bn1(self.conv1(x)))
        # Apply second conv -> bn -> relu
        x_processed = self.relu(self.bn2(self.conv2(x_processed)))
        # If residual is True, add the original input 'x' to the output
        if self.residual:
            return x_processed + x # This is a residual connection
        return x_processed
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Convolution with stride=2 to halve spatial dimensions
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Transposed Convolution to double spatial dimensions
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_embedding_dim=256, initial_filters=64):
        super().__init__()
        # The time_embedding_dim MUST match the bottleneck's channel size for direct addition
        # In this UNet, bottleneck has initial_filters * 8 channels.
        if time_embedding_dim != initial_filters * 8:
            print(f"Warning: time_embedding_dim ({time_embedding_dim}) does not match bottleneck channels ({initial_filters * 8}).")
            print(f"Adjusting time_embedding_dim to {initial_filters * 8} for compatibility.")
            time_embedding_dim = initial_filters * 8 # Adjust for compatibility

        self.time_embedding = SinusoidalPositionalEmbeddings(time_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.ReLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )

        # Encoder (Downsampling Path) - Contracting path
        # Each 'down' layer halves the spatial size and increases channels
        self.inc = ConvBlock(in_channels, initial_filters) # e.g., 3 -> 64 channels
        self.down1 = Downsample(initial_filters, initial_filters * 2) # 64 -> 128 channels
        self.conv1 = ConvBlock(initial_filters * 2, initial_filters * 2) # 128 -> 128 channels
        self.down2 = Downsample(initial_filters * 2, initial_filters * 4) # 128 -> 256 channels
        self.conv2 = ConvBlock(initial_filters * 4, initial_filters * 4) # 256 -> 256 channels
        self.down3 = Downsample(initial_filters * 4, initial_filters * 8) # 256 -> 512 channels
        self.conv3 = ConvBlock(initial_filters * 8, initial_filters * 8) # 512 -> 512 channels

        # Bottleneck - Deepest, most abstract features
        self.bottleneck = ConvBlock(initial_filters * 8, initial_filters * 8) # 512 -> 512 channels

        # Decoder (Upsampling Path) - Expansive path
        # Each 'up' layer doubles spatial size and decreases channels
        self.up1 = Upsample(initial_filters * 8, initial_filters * 4) # 512 -> 256 channels
        # Concatenates with skip connection from down3 (initial_filters * 8)
        self.conv4 = ConvBlock(initial_filters * 8, initial_filters * 4) # (256+256) -> 256 channels
        self.up2 = Upsample(initial_filters * 4, initial_filters * 2) # 256 -> 128 channels
        # Concatenates with skip connection from down2 (initial_filters * 4)
        self.conv5 = ConvBlock(initial_filters * 4, initial_filters * 2) # (128+128) -> 128 channels
        self.up3 = Upsample(initial_filters * 2, initial_filters) # 128 -> 64 channels
        # Concatenates with skip connection from down1 (initial_filters * 2)
        self.conv6 = ConvBlock(initial_filters * 2, initial_filters) # (64+64) -> 64 channels

        # Output layer - Maps features back to image channels
        self.outc = nn.Conv2d(initial_filters, out_channels, kernel_size=1) # 64 -> 3 channels (for noise)

    def forward(self, x, t):
        # 1. Process timestep 't' into an embedding
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)
        # Reshape time embedding to (batch_size, dim, 1, 1) for broadcasting
        t_emb = t_emb.view(t_emb.shape[0], t_emb.shape[1], 1, 1)

        # 2. Encoder Path (Downsampling)
        x1 = self.inc(x)
        x2 = self.down1(x1) # x2 is smaller spatially, more channels
        x2 = self.conv1(x2)
        x3 = self.down2(x2)
        x3 = self.conv2(x3)
        x4 = self.down3(x3)
        x4 = self.conv3(x4)

        # 3. Bottleneck
        x5 = self.bottleneck(x4)

        # 4. Add time embedding to bottleneck features
        # This is where the network learns to condition its noise prediction on the timestep
        x5 = x5 + t_emb

               # Decoder Path (Upsampling) with Corrected Skip Connections
        # Up 1: from (B, 512, 8, 8) to (B, 256, 16, 16)
        x = self.up1(x5)
        # Concatenate with x3 (which is (B, 256, 16, 16))
        # Total channels: 256 (from x) + 256 (from x3) = 512
        x = torch.cat([x, x3], dim=1)
        # Conv4 takes 512 channels, outputs 256
        x = self.conv4(x) # (B, 256, 16, 16)

        # Up 2: from (B, 256, 16, 16) to (B, 128, 32, 32)
        x = self.up2(x)
        # Concatenate with x2 (which is (B, 128, 32, 32))
        # Total channels: 128 (from x) + 128 (from x2) = 256
        x = torch.cat([x, x2], dim=1)
        # Conv5 takes 256 channels, outputs 128
        x = self.conv5(x) # (B, 128, 32, 32)

        # Up 3: from (B, 128, 32, 32) to (B, 64, 64, 64)
        x = self.up3(x)
        # Concatenate with x1 (which is (B, 64, 64, 64))
        # Total channels: 64 (from x) + 64 (from x1) = 128
        x = torch.cat([x, x1], dim=1)
        # Conv6 takes 128 channels, outputs 64
        x = self.conv6(x) # (B, 64, 64, 64)

        # 6. Output Layer
        output = self.outc(x) # Final convolution to get 3 channels (predicted noise)
        return output
def backward_diffusion_sample(x_t, t, model, betas, timesteps, device="cuda"):
    # Ensure all tensors are on the correct device
    betas = betas.to(device)
    x_t = x_t.to(device)
    t = t.to(device)

    # Pre-calculate alpha values and their cumulative products (if not already done globally)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

    # Get the relevant values for the current timestep t
    beta_t = get_index_from_list(betas, t, x_t.shape)
    alpha_t = get_index_from_list(alphas, t, x_t.shape)
    alphas_cumprod_t = get_index_from_list(alphas_cumprod, t, x_t.shape)
    alphas_cumprod_prev_t = get_index_from_list(alphas_cumprod_prev, t, x_t.shape)

    # Calculate the variance of the reverse process
    # This is a common choice for the variance (often called beta_tilde_t)
    posterior_variance = beta_t * (1. - alphas_cumprod_prev_t) / (1. - alphas_cumprod_t)

    # Set the model to evaluation mode for inference
    model.eval()
    with torch.no_grad(): # Disable gradient calculation for inference
        # Predict the noise using the UNet model
        predicted_noise = model(x_t, t)

    # Calculate the mean of the reverse process (mu_theta(x_t, t))
    # This formula is derived from the diffusion model theory
    mean_numerator = (x_t - beta_t * predicted_noise / torch.sqrt(1. - alphas_cumprod_t))
    mean = mean_numerator / torch.sqrt(alpha_t)

    # If t > 0, sample from the Gaussian distribution
    if t.item() > 0:
        # Generate random noise for sampling
        z = torch.randn_like(x_t)
        # x_{t-1} = mean + sqrt(variance) * z
        x_t_minus_1 = mean + torch.sqrt(posterior_variance) * z
    else:
        # If t is 0, we have reached the clean image, no more noise to add
        x_t_minus_1 = mean # Or simply the predicted x_0, which is the mean

    # Clamp the output to be within the typical image range [-1, 1] or [0, 1]
    # depending on your data normalization. Here, assuming [-1, 1]
    x_t_minus_1 = torch.clamp(x_t_minus_1, -1., 1.) # Adjust based on your image normalization

    return x_t_minus_1

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms # This is where 'transforms' is imported
from torchvision.datasets import ImageFolder
from PIL import Image
import os
# --- Training Configuration ---
IMAGE_CHANNELS = 3
IMAGE_SIZE = 64 # Assuming your human faces dataset will be resized to 64x64
TIMESTEPS = 1000
BETA_START = 0.0001
BETA_END = 0.02
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
NUM_DUMMY_SAMPLES = 100 # For testing with dummy data

# --- 1. Pre-compute Beta Schedule ---
betas = torch.linspace(BETA_START, BETA_END, TIMESTEPS).to(device)

# --- 2. Initialize UNet Model ---
model = UNet(
    in_channels=IMAGE_CHANNELS,
    out_channels=IMAGE_CHANNELS,
    time_embedding_dim=256,
    initial_filters=64
).to(device)

# --- 3. Define Optimizer and Loss Function ---
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss() # Mean Squared Error Loss

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE), # Ensure square images
    transforms.ToTensor(), # Converts to [0, 1] range
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalizes to [-1, 1]
])


dataset = ImageFolder(root='/kaggle/input/synthetic-faces-high-quality-sfhq-part-2/images', transform=transform)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 5. Training Loop ---
print("\nStarting training...")
for epoch in range(NUM_EPOCHS):
    model.train() # Set model to training mode
    total_loss = 0
    for step, batch_data in enumerate(dataloader):
        # If your dataset yields (image, label), you only need the image
        if isinstance(batch_data, (list, tuple)):
            batch_images = batch_data[0].to(device)
        else:
            batch_images = batch_data.to(device) # For DummyImageDataset

        # 1. Forward Diffusion (add noise)
        noisy_images, noise, t = forward_pass(batch_images, betas, TIMESTEPS, device)

        # 2. UNet Prediction
        predicted_noise = model(noisy_images, t)

        # 3. Calculate Loss
        loss = criterion(predicted_noise, noise)

        # 4. Backward Pass and Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 10 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Step {step}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}\n")

print("Training complete!")

