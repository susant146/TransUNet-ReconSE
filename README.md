# TransUNet-ReconSE
 TransUNet-ReconSE, a progressive architectural refinement strategy incorporating Squeeze-and-Excitation (SE) blocks, DropBlock regularization, and residual learning.

## Abstract
Magnetic Resonance Imaging (MRI) provides excellent soft tissue contrast, but suffers from long acquisition times from sequential k-space sampling. Accelerating MRI acquisition through undersampling introduces aliasing artifacts and leads to the loss of structural detail. In this study, we propose TransUNet-Recon, an adaptation of Transformer-augmented architectures for multicoil MRI reconstruction in the image domain. Specifically, we adopted the TransUNet framework by integrating a ResNetV2 encoder with a Vision Transformer (ViT) bottleneck and a U-Net-style decoder. To enhance robustness and performance, we further introduce TransUNet-ReconSE, a progressive architectural refinement strategy incorporating Squeeze-and-Excitation (SE) blocks, DropBlock regularization, and residual learning, each targeting core challenges of MRI reconstruction such as artifact suppression, anatomical coherence, and robust generalization. We conducted a comprehensive evaluation comprising quantitative metrics (NMSE, PSNR, SSIM), qualitative visual analysis, and cross-dataset validation to assess generalization across unseen acquisition settings. Experiments on the multicoil fastMRI knee dataset under 4$\times$ and 8$\times$ acceleration rates show that our final model consistently outperforms baseline and several other state-of-the-art models in both reconstruction fidelity and structural preservation.

## Problem Definition
<img width="4901" height="2067" alt="image" src="https://github.com/user-attachments/assets/cc4c6668-a5af-4be7-8881-3146e9708823" />


## Block Diagram
<img width="607" height="412" alt="image" src="https://github.com/user-attachments/assets/5a9a9130-a772-44a0-8090-009394d6009d" />

## Results
<img width="673" height="395" alt="image" src="https://github.com/user-attachments/assets/a3cd802d-d2d7-4e41-98bf-108b38e906ad" />


## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
* [TransUNet](https://github.com/beckschen/transunet)

## Citations

```bibtex
@article{panigrahi2025transunetrecon,
  title={TransUNet-Recon: A Transformer-Augmented UNet Architecture for Accelerated MRI Reconstruction},
  author={Panigrahi, Susant Kumar and Pal, Subhoshri and Sasmal, Pradipta and Sheet, Debdoot},
  conference={Indian Conference on Computer Vision, Graphics and Image Processing (ICVGIP)},
  year={2025}
}
```
