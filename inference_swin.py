import os
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import time
import logging
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

from Batch_Loader import DataTransform, complex_to_magnitude
from data.subsample import create_mask_for_mask_type

from TransUNet.networks.swin_config import get_swin_config
from TransUNet.networks.swin_transformer import SwinTransformer
from TransUNet.networks.swin_modelling import SwinUNet, DecoderCup, SegmentationHead, SwinEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(checkpoint_path, img_size):
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
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model

def test_single_file(model, h5_path, save_dir="SWIN_test_results"):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "test_results.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s", filemode='w')
    logger = logging.getLogger()

    IMG_SIZE = 320
    mask_func = create_mask_for_mask_type('random', [0.08], [4])
    transform = DataTransform(resolution=IMG_SIZE, which_challenge='multicoil', mask_func=mask_func, use_seed=False)

    with h5py.File(h5_path, 'r') as hf:
        kspace = hf['kspace']
        mask = hf['mask'][()] if 'mask' in hf else None
        target = hf['reconstruction_rss']
        attrs = dict(hf.attrs)

        best_psnr, best_ssim = -1, -1
        avg_psnr, avg_ssim = 0, 0
        best_slice_num = -1
        best_slice = None
        print("Total slices in file:", kspace.shape[0])

        slice_metrics = []
        for slice_idx in range(kspace.shape[0]):
            try:
                sample_kspace = kspace[slice_idx]
                sample_target = target[slice_idx]
                image, target_img, *_ = transform(sample_kspace, mask, sample_target, attrs, 'singlefile', slice_idx)
    
                image = image.unsqueeze(0).float().to(device)         # (1, 2, H, W)
                target_img = target_img.unsqueeze(0).float().to(device)  # (1, 2, H, W)
    
                with torch.no_grad():
                    input_mag = complex_to_magnitude(image)
                    target_mag = complex_to_magnitude(target_img)
                    output = model(input_mag)
    
                input_np = input_mag.squeeze().cpu().numpy()
                output_np = output.squeeze().cpu().numpy()
                target_np = target_mag.squeeze().cpu().numpy()
    
                psnr_val = psnr_metric(target_np, output_np, data_range=target_np.max() - target_np.min())
                ssim_val = ssim_metric(target_np, output_np, data_range=target_np.max() - target_np.min())

                slice_metrics.append((slice_idx, psnr_val, ssim_val))
                print(f"Slice {slice_idx:02d}: PSNR = {psnr_val:.2f}, SSIM = {ssim_val:.4f}")
                avg_psnr += psnr_val
                avg_ssim += ssim_val
                if ssim_val > best_ssim:
                    best_ssim = ssim_val
                    best_psnr = psnr_val
                    best_slice = (input_np, output_np, target_np)

            except Exception as e:
                print(f"Error on slice {slice_idx}: {e}")

    if best_slice:
        input_img, output_img, target_img = best_slice
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(input_img, cmap='gray')
        axs[0].set_title("Undersampled Input")
        axs[1].imshow(output_img, cmap='gray')
        axs[1].set_title("Reconstructed Output")
        axs[2].imshow(target_img, cmap='gray')
        axs[2].set_title("Ground Truth")
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "best_slice.png"))
        plt.close()

    print(f"\nBest PSNR: {best_psnr:.2f}")
    print(f"Best SSIM: {best_ssim:.4f}")
    avg_psnr /= kspace.shape[0]
    avg_ssim /= kspace.shape[0]
    print(f"\nAverage PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")

    logger.info(f"Tested File: {os.path.basename(h5_path)}")
    logger.info(f"Best Slice: {best_slice_num}")
    logger.info(f"Best PSNR: {best_psnr:.2f}")
    logger.info(f"Best SSIM: {best_ssim:.4f}")
    logger.info(f"Average PSNR: {avg_psnr:.2f}")
    logger.info(f"Average SSIM: {avg_ssim:.4f}")


if __name__ == "__main__":
    CHECKPOINT = "SWIN_models/model1/best_model.pth"
    H5_FILE = "../susant/knee_multicoil_val_subset/file1000153.h5"
    model = load_model(CHECKPOINT, img_size=320)
    test_single_file(model, H5_FILE)
