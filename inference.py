import os
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

from TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from Batch_Loader import DataTransform, complex_to_magnitude
from data.subsample import create_mask_for_mask_type

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(checkpoint_path, img_size):
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 1
    config_vit.n_skip = 3
    config_vit.patches.grid = (img_size // 16, img_size // 16)
    model = ViT_seg(config_vit, img_size=img_size, num_classes=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model

def test_single_file(model, h5_path, save_dir="test_results"):
    os.makedirs(save_dir, exist_ok=True)

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
        best_slice = None
        print("Total slices in file:", kspace.shape[0])

        slice_metrics = []
        for slice_idx in range(kspace.shape[0]):
            try:
                sample_kspace = kspace[slice_idx]
                sample_target = target[slice_idx]
                image, target_img, *_ = transform(sample_kspace, mask, sample_target, attrs, 'singlefile', slice_idx)
    
                image = image.unsqueeze(0).float().to(device) # (1, 2, H, W)
                target_img = target_img.unsqueeze(0).float().to(device)  # (1, 2, H, W)
    
                with torch.no_grad():
                    input_mag = complex_to_magnitude(image)
                    target_mag = complex_to_magnitude(target_img)
                    output = model(input_mag, baseline=input_mag)
    
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

if __name__ == "__main__":
    CHECKPOINT = "saved_models/model10/best_model.pth"
    H5_FILE = "../susant/knee_multicoil_val_subset/file1000153.h5"
    model = load_model(CHECKPOINT, img_size=320)
    test_single_file(model, H5_FILE)
