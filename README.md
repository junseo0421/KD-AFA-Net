# Deep learning-based image outpainting of finger-vein image
KD-AFA-Net is a lightweight outpainting network for contactless finger-vein images using FFT-based AFA and knowledge distillation, achieving EERs of 2.56/3.49/1.78% on HKPU/SDUMLA-HMT/MMCBNU_6000.

<details>
<summary>Abstract (click to expand)</summary>
Despite fast authentication and user convenience, the lack of a fixed frame in contactless finger-vein acquisition causes missing regions and discrepancies between enrolled and query images, thereby degrading recognition performance. Existing image outpainting-based methods restore missing regions but often contain a large number of parameters, making them slow and unsuitable for real-time applications. To overcome these issues, this paper proposes a lightweight image outpainting network called knowledge distilled adaptive frequency attention network (KD-AFA-Net). KD-AFA-Net is based on a lightweight model that uses thinner separable U-Net with knowledge distillation (KD) from a high-performance teacher. In addition, to compensate for the limitations of convolutional neural networks (CNNs) in capturing global information, a novel adaptive frequency attention (AFA) module is designed. The AFA module decomposes intermediate features via a two-dimensional fast Fourier transform (FFT), learns the importance of high-frequency and low-frequency components, and emphasizes the important ones. Furthermore, this paper also proposes the AFA KD loss which enables the student model to effectively learn the frequency-domain refined outputs of the teacher’s AFA module. Moreover, we analyze recognition performance and use large language models (LLMs), ChatGPT-4o and ChatGPT-5 to prioritize experiments and to examine utilization strategies for future image-based tasks. Experiments on the Hong Kong Polytechnic University finger-image database version 1, the Shandong University machine learning and applications-homologous multi-modal traits (SDUMLA-HMT) finger-vein database, and the MMCBNU_6000 database show that KD-AFA-Net achieves equal error rates (EERs) of 2.56%, 3.49%, and 1.78% respectively, outperforming state-of-the-art image outpainting and KD methods while supporting real-time efficiency.
</details>

## 🔥 Highlights
- **Lightweight outpainting** via a thinner-separable U-Net backbone + knowledge distillation (teacher → student)
- **Semantically fused input**: concatenates the original, edge-masked, and contrast-enhanced images to stabilize training and improve vein restoration and recognition
- **Adaptive Frequency Attention (AFA)**: FFT-based decomposition to emphasize informative low/high-frequency components
- **AFA-KD loss**: distills frequency-refined representations from the teacher’s AFA module
- **Strong recognition performance** on HKPU, SDUMLA-HMT, MMCBNU_6000 with real-time efficiency

## 📰 Paper
- **Expert Systems with Applications (ESWA)**  
- DOI: `10.1016/j.eswa.2025.131015`  
- Link: https://doi.org/10.1016/j.eswa.2025.131015

## 📝 Citation
```bibtex
@article{Kim2026KDAFANet,
  title   = {Deep learning-based image outpainting of finger-vein image},
  author  = {Junseo Kim and Jinseong Hong and Jungsoo Kim and Seongin Jeong and Seokjun Lim and Wonho Jang and Kangryoung Park},
  journal = {Expert Systems with Applications},
  year    = {2026}
}
```

## 🖼️ Overview
<img width="534" height="231" alt="image" src="https://github.com/user-attachments/assets/945522dd-6f57-407d-aa8d-dbe565f8fa3a" />

## Datasets
1. Hong Kong Polytechnic University finger-image database version 1 (HKPU) [1]
2. Shandong University machine learning and applications-homologous multi-modal traits (SDUMLA-HMT) finger-vein database [2]
3. MMCBNU_6000 database [3]

## 📊 Results (EER, %)
KD-AFA-Net achieves the best EER across all three datasets.
<details>
<summary><b>Full comparison table (EER, %)</b></summary>

| Method | HKPU-DB | SDU-DB | MMCBNU-DB |
|---|---:|---:|---:|
| Pix2Pix | 5.94 | 6.36 | 11.94 |
| CycleGAN | 22.2 | 19.31 | 33.3 |
| Pix2Pix-HD | 5.01 | 4.78 | 15.74 |
| Harmonization GAN | 9.7 | 16.7 | 13.8 |
| CUT | 21.14 | 18.71 | 21.46 |
| QueryOTR | 4.61 | 5.22 | 3.22 |
| U-transformer | 3.98 | 4.56 | 6.65 |
| SN-DCR | 21.42 | 19.62 | 12 |
| U-Net with CLE | 3.87 | 5.07 | 5.69 |
| Enhanced U-transformer | 3.01 | 4.33 | 1.87 |
| **KD-AFA-Net (Ours)** | **2.56** | **3.49** | **1.78** |

> The results are taken from Table 7 in our ESWA paper (DOI: 10.1016/j.eswa.2025.131015).

</details>

## Implementation
* Python >= 3.10
* PyTorch >= 2.5.1

## Model weights 
* [KD-AFA-Net weight](https://drive.google.com/file/d/1luNiUOc5KqOKvtTWz_5Q6IujI4MyKflW/view?usp=sharing)

## References
[1] A. Kumar, Y. Zhou, Human identification using finger images, IEEE Trans. Image Process. 21 (2012) 2228–2244. https://doi.org/10.1109/TIP.2011.2171697.

[2] Y. Yin, L. Liu, X. Sun, SDUMLA-HMT: A multimodal biometric database, in: Proc. the 6th Chinese Conference on Biometric Recognition, 2011, pp. 260–268. https://doi.org/10.1007/978-3-642-25449-9_33.

[3] Y. Lu, S.J. Xie, S. Yoon, Z. Wang, D.S. Park, An available database for the research of finger vein recognition, in: Proc. 2013 6th Int. Congr. Image Signal Process., 2013, vol. 1, pp. 410–415. https://doi.org/10.1109/CISP.2013.6744030.



