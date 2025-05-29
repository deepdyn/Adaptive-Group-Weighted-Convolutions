# Adaptive Group-Weighted Convolutions (AGWC)

Adaptive Group-Weighted Convolutions (AGWC) is a novel deep learning framework that extends classical group-equivariant convolutional neural networks (G-CNNs) by introducing learnable importance weights over symmetry group transformations. Unlike traditional G-CNNs that assume uniform relevance across all group elements (e.g., rotations and reflections), AGWC dynamically adapts the model's sensitivity to different transformations based on the data distribution, while rigorously preserving exact equivariance.

This approach improves model capacity allocation, leading to enhanced accuracy, better confidence calibration, and improved robustness, especially on datasets exhibiting skewed pose distributions. AGWC provides a mathematically principled and practical method for integrating adaptive symmetry priors into equivariant deep networks, with applications in computer vision and beyond.

## Abstract / Overview

Traditional group-equivariant convolutional neural networks enforce uniform parameter sharing across all symmetry transformations, an assumption that often conflicts with the uneven distribution of object poses found in natural images. To address this, we propose Adaptive Group-Weighted Convolutions (AGWC), an enhancement to the canonical \( p4m \) group-equivariant convolutional layer that introduces a learnable positive weighting function applied to each orientation channel. We rigorously prove that weighting by relative group displacements commutes with the group action, ensuring exact equivariance is maintained while adding minimal parameter overhead.

Our experiments demonstrate consistent improvements across standard benchmarks, including an increase of approximately 0.85% top-1 accuracy on CIFAR-10 and 0.9% on Fashion-MNIST, while preserving baseline performance on MNIST and Rotated MNIST. Notably, AGWC significantly reduces expected calibration error by over 50% on the GTSRB dataset. These results are statistically significant across multiple random seeds under fixed training conditions. The adaptive weighting framework provides a principled interpolation between strict parameter sharing and fully independent filters, all while preserving symmetry guarantees. Furthermore, the underlying theoretical framework is extensible to continuous groups through harmonic parameterizations.

## Installation and Dependencies

This project requires the following Python packages:

- `torch` (PyTorch)
- `torchvision`
- `numpy`
- `random` (built-in Python module)
- `e2cnn` (Equivariant CNN library)

You can install the main dependencies using pip:

```bash
pip install torch torchvision numpy e2cnn
```

## Dataset Splits

We use eight datasets in our experiments, including canonical, rotated, and augmented variants of standard benchmarks. For consistent and fair evaluation, each dataset is split into training, validation, and test sets as summarized below:

| Dataset                             | Training | Validation | Test   |
| ----------------------------------- | -------- | ---------- | ------ |
| MNIST / Rotated MNIST               | 12,000   | 2,000      | 56,000 |
| FashionMNIST / Rotated FashionMNIST | 55,000   | 5,000      | 10,000 |
| CIFAR-10 / CIFAR-10+                | 40,000   | 10,000     | 10,000 |
| GTSRB / GTSRB+                      | 21,312   | 5,328      | 12,630 |

The training sets are used for model optimization, the validation sets for hyperparameter tuning and early stopping, and the test sets for final performance evaluation. This split setup facilitates reproducibility and consistent comparison across experiments.

For dataset preprocessing, augmentation, and preparation scripts, please refer to the individual dataset folders.

## Results

This repository provides comprehensive experimental results demonstrating the effectiveness of Adaptive Group-Weighted Convolutions (AGWC) across multiple benchmark datasets.

### Summary of Main Results

| Model        | Rotated MNIST    | CIFAR-10         | FashionMNIST     | GTSRB            |
| ------------ | ---------------- | ---------------- | ---------------- | ---------------- |
| Planar       | 96.54 ± 0.93     | 85.34 ± 0.23     | 93.02 ± 0.58     | 97.10 ± 0.29     |
| P4 / P4m     | 97.19 ± 1.18     | 90.75 ± 0.18     | 93.14 ± 0.36     | 97.04 ± 0.78     |
| P4-W / P4m-W | **97.68 ± 0.80** | **91.60 ± 0.37** | **94.05 ± 0.26** | **97.98 ± 0.37** |

- AGWC consistently improves classification accuracy and calibration compared to baseline planar CNNs and classical group-equivariant CNNs.
- Notable improvements include:
  - Up to ~0.9% increase in top-1 accuracy on CIFAR-10 and Fashion-MNIST and other datasets.
  - Over 50% reduction in expected calibration error (ECE) on the GTSRB dataset.
- Results are averaged over 5 random seeds and training runs to ensure statistical significance.

### Access to Logs and Outputs

- Training and evaluation logs for all eight datasets are available in the respective `Outputs/` folders within each dataset directory.

### Visualizations

- The repository includes scripts and output figures for key visualizations such as:
  - Learned importance weight heatmaps across layers and datasets.
  - KL divergence trajectories measuring deviation from uniform symmetry weighting.
  - Calibration plots and reliability diagrams comparing AGWC to baselines.

Refer to the `visualizations/` folder for generated plots that reproduce figures from the paper.

---

For any questions about results or how to reproduce specific experiments, please open an issue or contact the author.

## License

This project is licensed under the Apache License 2.0.  
See the [LICENSE](./LICENSE) file for details.

You are free to use, modify, and distribute this software under the terms of the Apache License 2.0.

## Contact / Support

For any questions, issues, or collaboration inquiries related to this project, please feel free to contact the authors:

- **Dr. Pradeep Singh** – Post-Doctoral
  Researcher and Principal Investigator at the Machine Intelligence Lab, Department of Computer
  Science and Engineering, IIT Roorkee

  Email: pradeep.cs@sric.iitr.ac.in

- **Kanishk Sharma** – M.Tech CSE, IIT Roorkee  
  Email: kanishk_s@cs.iitr.ac.in

- **Dr. Balasubramanian Raman** – Professor (HAG) and Head of the Department of Computer Science & Engineering, IIT Roorkee  
  Email: bala@cs.iitr.ac.in

We welcome feedback, bug reports, and contributions!
