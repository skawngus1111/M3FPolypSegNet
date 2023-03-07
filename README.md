# M3FPolypSegNet: Multi-Frequency Feature Fusion Segmentation Network for Polyp Localization in Colonoscopy Images
Available code for ICIP2023
![M3FPolypSegNet](https://user-images.githubusercontent.com/77310264/223346757-0774fd2f-4dff-4753-a66c-1baa40ab32f2.png)

## Abstract
Polyp segmentation is crucial for preventing colorectal cancer a common type of cancer. Deep learning has been used to segment polyps automatically, which reduces the risk of misdiagnosis. Localizing small polyps in colonoscopy images is challenging because of its complex characteristics, such as color, occlusion, and various shapes of polyps. To address this challenge, a novel frequency-based fully convolutional neural network, \textit{Multi-Frequency Feature Fusion Polyp Segmentation Network (M3FPolypSegNet)} was proposed to decompose the input image into low/high/full-frequency components to use the characteristics of each component. We used three independent multi-frequency encoders to map multiple input images into a high-dimensional feature space. In the \textit{Frequency-ASPP Scalable Attention Module (F-ASPP SAM)}, ASPP was applied between each frequency component to preserve scale information. Subsequently, scalable attention was applied to emphasize polyp regions in a high-dimensional feature space. Finally, we designed three multi-task learning (i.e., region, edge, and distance) in four decoder blocks to learn structural characteristics of the region. An average IoU improvement of 1.63% was achieved compared with the existing state-of-the-art model PraNet on two polyp segmentation benchmark datasets (CVC-ClinicDB and BKAI-IGH-NeoPolyp).

## Main Contribution

- We propose a novel polyp segmentation model (M3F PolypSegNet) based on a multi-frequency encoder and a single-decoder architecture that utilizes unique characteristics for each frequency component.

- F-ASPP SAM introduces trainable parameters between the foreground/background attention of frequency and scale to prevent information loss during the gradual upsampling of the decoder block. Furthermore, the vanishing gradient problem was prevented by performing multi-task deep supervision training in each decoder block.

- We experimentally achieved state-of-the-art performance in various evaluation metrics when comparing various polyp image segmentation models on two datasets (CVC-ClinicDB and BKAI-IGH-NeoPolyp).

## Experiment Results
### Quantitative Results
![Screenshot from 2023-03-07 15-56-36](https://user-images.githubusercontent.com/77310264/223347217-835dfd0a-d559-47cd-941d-71224ee25b38.png)

### Qualitative Results
![Screenshot from 2023-03-07 15-57-36](https://user-images.githubusercontent.com/77310264/223347489-5a74a674-0224-48c2-a1bc-bf44657dd267.png)

### Ablation Study
![Screenshot from 2023-03-07 15-58-02](https://user-images.githubusercontent.com/77310264/223347420-4c4e5c4b-d892-45ff-a463-b60659042cc6.png)
