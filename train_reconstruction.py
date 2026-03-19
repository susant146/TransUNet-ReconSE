import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from TransUNet.utils import CombinedLoss
from Batch_Loader import _create_dataset, DataTransform, complex_to_magnitude
from data.subsample import create_mask_for_mask_type
from torch.optim.lr_scheduler import StepLR
import time

IMG_SIZE = 320
BATCH_SIZE = 4
NUM_EPOCHS = 80
LEARNING_RATE = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True

def save_sample(undersampled, img, target, save_path):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(undersampled.cpu().squeeze(), cmap='gray')
    axs[0].set_title("Undersampled Input")

    axs[1].imshow(img.cpu().squeeze(), cmap='gray')
    axs[1].set_title("Reconstructed Output")

    axs[2].imshow(target.cpu().squeeze(), cmap='gray')
    axs[2].set_title("Target Image")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train():
    mask_func = create_mask_for_mask_type('random', [0.04], [8])
    transform = DataTransform(resolution=320, which_challenge='multicoil', mask_func=mask_func, use_seed=True)

    train_path = ["../susant/Knee_Multicoil_train_batch0/"]

    val_path = ["../susant/Knee_Multicoil_train_batch1/"]
    train_loader = _create_dataset(train_path, transform, batch_size=4, shuffle=True, sample_rate=1.0, num_workers=16, pin_memory=True, mode='train')
    val_loader = _create_dataset(val_path, transform, batch_size=4, shuffle=False, sample_rate=1.0, num_workers=16, pin_memory=True, mode='val')

    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 1
    config_vit.n_skip = 3
    config_vit.patches.grid = (IMG_SIZE // 16, IMG_SIZE // 16)
    model = ViT_seg(config_vit, img_size=IMG_SIZE, num_classes=1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    criterion = nn.L1Loss()
    MODEL_NAME = "model11"
    model_save_dir = os.path.join("saved_models", MODEL_NAME)
    best_model_path = os.path.join(model_save_dir, "best_model.pth")
    os.makedirs(model_save_dir, exist_ok=True)

    log_file = os.path.join(model_save_dir, "training.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s", filemode='w')
    logger = logging.getLogger()
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)

    psnr_output_list = []
    ssim_output_list = []
    best_ssim = -1.0

    for epoch in range(NUM_EPOCHS):
        train_start = time.time()
        model.train()
        epoch_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in loop:
            image, target, *_ = batch
            image = complex_to_magnitude(image).float().to(device)
            target = complex_to_magnitude(target).float().to(device)

            output = model(image)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        train_time = time.time() - train_start
        print(f"Epoch {epoch+1}: Avg Train Loss = {epoch_loss / len(train_loader):.4f}")

        val_start = time.time()
        model.eval()
        psnr_vals, ssim_vals = [], []
        best_ssim_slice = -1.0
        best_sample = (None, None, None)

        with torch.no_grad():
            for val_batch in val_loader:
                val_image, val_target, *_ = val_batch
                val_image = complex_to_magnitude(val_image).float().to(device)
                val_target = complex_to_magnitude(val_target).float().to(device)

                val_output = model(val_image, baseline=val_image)

                for i in range(val_image.shape[0]):
                    input_np = val_image[i].cpu().numpy().squeeze()
                    target_np = val_target[i].cpu().numpy().squeeze()
                    output_np = val_output[i].cpu().numpy().squeeze()

                    output_psnr = psnr_metric(target_np, output_np, data_range=target_np.max() - target_np.min())
                    output_ssim = ssim_metric(target_np, output_np, data_range=target_np.max() - target_np.min())

                    psnr_vals.append(output_psnr)
                    ssim_vals.append(output_ssim)

                    if output_ssim > best_ssim_slice:
                        best_ssim_slice = output_ssim
                        best_sample = (val_image[i], val_output[i], val_target[i])

        avg_ssim_epoch = np.mean(ssim_vals)
        avg_psnr_epoch = np.mean(psnr_vals)
        val_time = time.time() - val_start

        logger.info(f"Epoch {epoch+1:02d} | Train Time: {train_time:.2f}s | Val Time: {val_time:.2f}s | "
                    f"Avg SSIM: {avg_ssim_epoch:.4f} | Avg PSNR: {avg_psnr_epoch:.2f}")
        print(f"Output-Avg PSNR: {avg_psnr_epoch:.2f}, Output-Avg SSIM: {avg_ssim_epoch:.4f}\n")

        # Saving model based on average SSIM
        if avg_ssim_epoch > best_ssim:
            best_ssim = avg_ssim_epoch
            print(f"New best model found at epoch {epoch+1}, saving to {best_model_path}")
            torch.save(model.state_dict(), best_model_path)
            save_sample(*best_sample, os.path.join(model_save_dir, "best_sample.png"))

        # Saving periodically
        if (epoch + 1) in [50, 60, 70, 80]:
            epoch_dir = os.path.join(model_save_dir, f"epoch_{epoch+1:02d}")
            os.makedirs(epoch_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(epoch_dir, "model.pth"))
            save_sample(*best_sample, os.path.join(epoch_dir, "sample_slice.png"))

        psnr_output_list.append(avg_psnr_epoch)
        ssim_output_list.append(avg_ssim_epoch)

    #Plot SSIM/PSNR
    epochs = list(range(1, NUM_EPOCHS + 1))
    plt.figure()
    plt.plot(epochs, psnr_output_list, label='PSNR (Output)', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR over Epochs')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(model_save_dir, 'psnr_plot.png'))
    plt.close()

    plt.figure()
    plt.plot(epochs, ssim_output_list, label='SSIM (Output)', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM over Epochs')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(model_save_dir, 'ssim_plot.png'))
    plt.close()

if __name__ == "__main__":
    train()
