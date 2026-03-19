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
from TransUNet.utils import CombinedLoss
from Batch_Loader import _create_dataset, DataTransform, complex_to_magnitude
from data.subsample import create_mask_for_mask_type
from torch.optim.lr_scheduler import StepLR
from TransUNet.networks.swin_config import get_swin_config
from TransUNet.networks.swin_transformer import SwinTransformer
from TransUNet.networks.swin_modelling import SwinUNet, DecoderCup, SegmentationHead, SwinEncoder
import time
from skimage.transform import resize


IMG_SIZE = 320
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 2e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True

def save_sample(undersampled, img, target, save_path):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    if undersampled.dim() == 3 and undersampled.size(0) == 3:
        undersampled = undersampled[0]
    if img.dim() == 3 and img.size(0) == 3:
        img = img[0]
    if target.dim() == 3 and target.size(0) == 3:
        target = target[0]

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
    mask_func = create_mask_for_mask_type('random', [0.08], [4])
    transform = DataTransform(resolution=320, which_challenge='multicoil', mask_func=mask_func, use_seed=True)

    train_path = ["../susant/Knee_Multicoil_train_batch0/"]
    val_path = ["../susant/Knee_Multicoil_train_batch1/"]
    train_loader = _create_dataset(train_path, transform, batch_size=8, shuffle=True, sample_rate=1.0, num_workers=8, pin_memory=True, mode='train')
    val_loader = _create_dataset(val_path, transform, batch_size=8, shuffle=False, sample_rate=1.0, num_workers=8, pin_memory=True, mode='val')
    print("Total slices in dataset:", len(val_loader))

    config_swin = get_swin_config()
    
    swin_backbone = SwinTransformer(
        img_size=config_swin.img_size,
        patch_size=config_swin.patch_size,
        in_chans=config_swin.in_chans,
        embed_dim=config_swin.embed_dim,
        depths=config_swin.depths,
        num_heads=config_swin.num_heads,
        window_size=config_swin.window_size,
        mlp_ratio=config_swin.mlp_ratio,
        qkv_bias=config_swin.qkv_bias,
        qk_scale=config_swin.qk_scale,
        drop_rate=config_swin.drop_rate,
        attn_drop_rate=config_swin.attn_drop_rate,
        drop_path_rate=config_swin.drop_path_rate,
        norm_layer=nn.LayerNorm,
        ape=config_swin.ape,
        patch_norm=config_swin.patch_norm,
        use_checkpoint=config_swin.use_checkpoint
    )

    encoder = SwinEncoder(swin_backbone)

    model = SwinUNet(
        encoder=encoder,
        decoder=DecoderCup(config_swin),
        segmentation_head=SegmentationHead(
            in_channels=config_swin.decoder_channels[-1],
            out_channels=1,
            upsampling=2
        )
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    criterion = nn.L1Loss()
    MODEL_NAME = "model1"
    model_save_dir = os.path.join("SWIN_models", MODEL_NAME)
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

    psnr_input_list = []
    ssim_input_list = []
    psnr_output_list = []
    ssim_output_list = []
    best_ssim = -1.0

    for epoch in range(NUM_EPOCHS):
        train_start=time.time()
        model.train()
        epoch_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in loop:
            image, target, *_ = batch
            image = complex_to_magnitude(image).float().to(device) # (B, 1, H, W)
            target = complex_to_magnitude(target).float().to(device) # (B, 1, H, W)
            if image.size(1) == 1:
                image = image.repeat(1, 3, 1, 1)
            
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            target = (target - target.min()) / (target.max() - target.min() + 1e-8)

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
        
        with torch.no_grad():
            model.eval()
            best_ssim_epoch = -1.0
            best_psnr_epoch = -1.0
            batch_best_psnrs = []
            batch_best_ssims = []
            best_val_image = None
            best_val_output = None
            best_val_target = None
            for val_batch in val_loader:
                val_image, val_target, *_ = val_batch
                val_image = complex_to_magnitude(val_image).float().to(device)
                val_target = complex_to_magnitude(val_target).float().to(device)
                if val_image.size(1) == 1:
                    val_image = val_image.repeat(1, 3, 1, 1)
                val_output = model(val_image)
                max_ssim = -1
                max_psnr = -1
                for j in range(val_image.size(0)):
                    input_np = val_image[j].cpu().numpy().squeeze()
                    target_np = val_target[j].cpu().numpy().squeeze()
                    output_np = val_output[j].cpu().numpy().squeeze()

                    if input_np.shape != target_np.shape:
                        input_np = resize(input_np, target_np.shape, mode='reflect', anti_aliasing=True)
                        input_np = input_np.mean(axis=0)
                    input_psnr = psnr_metric(target_np, input_np, data_range=target_np.max() - target_np.min())
                    output_psnr = psnr_metric(target_np, output_np, data_range=target_np.max() - target_np.min())
                    input_ssim = ssim_metric(target_np, input_np, data_range=target_np.max() - target_np.min())
                    output_ssim = ssim_metric(target_np, output_np, data_range=target_np.max() - target_np.min())

                    batch_best_psnrs.append(output_psnr)
                    batch_best_ssims.append(output_ssim)
                    if output_ssim > best_ssim_epoch:
                        best_ssim_epoch = output_ssim
                        best_psnr_epoch = output_psnr
                        best_val_image = val_image[j]
                        best_val_output = val_output[j]
                        best_val_target = val_target[j]

            avg_ssim_epoch = np.mean(batch_best_ssims)
            avg_psnr_epoch = np.mean(batch_best_psnrs)
            val_time = time.time() - val_start

            logger.info(f"Epoch {epoch+1:02d} | Train Time: {train_time:.2f}s | Val Time: {val_time:.2f}s | Avg SSIM: {avg_ssim_epoch:.4f} | Avg PSNR: {avg_psnr_epoch:.2f}")

            if best_ssim_epoch > best_ssim:
                best_ssim = best_ssim_epoch
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved at epoch {epoch+1} with SSIM: {best_ssim_epoch:.4f}, PSNR: {best_psnr_epoch:.2f}")
                save_sample(best_val_image, best_val_output, best_val_target, os.path.join(model_save_dir, "best_sample.png"))

            if (epoch + 1) in [30, 40, 50]:
                epoch_dir = os.path.join(model_save_dir, f"epoch_{epoch+1:02d}")
                os.makedirs(epoch_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(epoch_dir, "model.pth"))
                save_sample(best_val_image, best_val_output, best_val_target, os.path.join(epoch_dir, "sample_slice.png"))

            psnr_output_list.append(avg_psnr_epoch)
            ssim_output_list.append(avg_ssim_epoch)

        scheduler.step()
        
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