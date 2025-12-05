# Reproducing and Improving a State-of-the-Art CIFAR-10 CNN

**Course:** Neural Networks & Deep Learning  
**Dataset:** CIFAR-10  
**Framework:** PyTorch  

##  Project Overview
This project traverses the entire lifecycle of deep learning model development for image classification. Starting from a simple baseline CNN with limited parameters, the project progressively diagnoses architectural issues, scales dimensions, introduces modern components (Residuals, Batch Norm), reproduces a State-of-the-Art (SOTA) ResNet-18, and finally aims to surpass the SOTA baseline using advanced architectural and training enhancements.

##  Requiremts
The libraries used to build the project are in the requiremnts.txt file

## Project Structure

``` text
├── models/               # Directory to save trained model weights (.pth)
├── scripts/
│   ├── stage1.py         # Baseline CNN implementation (<1M params)
│   ├── stage1_experiments.py # Theoretical analysis: Linear Collapse & Bottlenecks
│   ├── stage2.py         # Scaling: Width vs. Depth comparison
│   ├── stage2_experiments.py # Failure analysis: Vanishing Gradient & Overfitting
│   ├── stage3.py         # Modern CNN: ResBlocks, BatchNorm, Regularization
│   ├── stage4.py         # SOTA Reproduction: ResNet-18 (CIFAR-adapted)
│   ├── stage5.py         # Improved SOTA: SE-ResNet + MixUp + Cutout
│   └── stage5_experiments.py # Ablation studies for Stage 5
└── README.md

## Project Roadmap & Stages

**Stage 1:** Baseline CNN & Measurement Framework

Goal: Implement a lightweight CNN with < 1 Million parameters to establish a performance baseline .

Architecture: Basic stack of Conv2d -> ReLU -> MaxPool layers.

Deliverables: Accuracy curves, confusion matrix, and parameter count analysis .

Theory Checkpoint: Investigation of representational power by removing non-linearities (Linear Collapse Experiment)

**Stage 2:** Architecture Scaling & Diagnostics

Goal: Analyze the impact of scaling network dimensions independently .

Experiments:

Width Scaling: Doubling the number of channels/filters in all layers.

Depth Scaling: Adding additional convolutional blocks without pooling.

Diagnostics: Analysis of overfitting behavior and optimization stability .

Stress Tests: Provoking Vanishing Gradients via extreme depth and Parameter Explosion via extreme width.

**Stage 3:** Modern CNN Components

Goal: Stabilize training and mitigate overfitting using modern deep learning techniques .

Architecture Upgrades: Introduction of Residual Connections and Batch Normalization.

Regularization: Implementation of Label Smoothing, Dropout, and Weight Decay.

Outcome: Significant improvement in convergence speed and generalization.

**Stage 4:** Reproduce a SOTA CIFAR-10 CNN

Goal: Rigorous reproduction of a standard industry benchmark from scratch .

Model: ResNet-18 (adapted for CIFAR-10 input size).

Training Recipe: SGD with Momentum (0.9) and Cosine Annealing Scheduler for 200 epochs .

Target: Achieve >93% accuracy (within 3.0 points of literature).


**Stage 5:** Surpass the SOTA Model

Goal: Implement architectural and training enhancements to beat the Stage 4 baseline .

Enhancements Implemented:

Architecture: Squeeze-and-Excitation (SE) Blocks for dynamic channel feature recalibration.

Data Augmentation: MixUp (training on convex combinations of image pairs).

Regularization: Cutout (Erasure-based augmentation).

Ablation Studies: Systematic removal of components (No-MixUp, No-SE, No-Cutout) to quantify individual contributions .