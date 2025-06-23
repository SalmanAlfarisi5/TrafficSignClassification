# Traffic Sign Classification (Multi-Class CNN Benchmark)

This project focuses on classifying 23 types of traffic signs using various CNN architectures. The dataset consists of ~5000 RGB images with varying resolutions and imbalanced class distributions. The goal is to compare different model performances including pretrained and scratch-built CNNs.

## 🗂 Dataset Overview

- **Total classes**: 23 traffic sign categories  
- **Samples per class**: Ranges from 100 to 500  
- **Image size**: Varies; all images resized to 224×224 during preprocessing  
- **Image format**: RGB `.png` files named as `{Class_Label}_{Sample_Index}.png`

## 📌 Task Objectives

- Stratified split: 80% train / 20% test per class  
- Handle variable image sizes with resizing  
- Train a CNN from scratch  
- Benchmark with pretrained models (AlexNet, ResNet50, VGG16, EfficientNetB0)  
- Apply fine-tuning by unfreezing final layers  
- Report train, validation, and test performance

## 🧪 Model Results

| Model           | Phase         | Best Train | Best Val | Test Acc |
|----------------|---------------|------------|----------|----------|
| **AlexNet**     | Frozen        | 0.9450     | 0.9540   | 0.9635   |
|                | Fine-tuned    | 0.9946     | 0.9898   | 0.9777   |
| **ResNet50**    | Frozen        | 0.9373     | 0.9386   | 0.9209   |
|                | Fine-tuned    | 1.0000     | 1.0000   | 0.9990   |
| **EfficientNetB0** | Frozen     | 0.9508     | 0.9233   | 0.9189   |
|                | Fine-tuned    | 0.9997     | 0.9987   | 0.9990   |
| **VGG16**       | Frozen        | 0.9201     | 0.9118   | 0.9209   |
|                | Fine-tuned    | 0.9974     | 0.9910   | 0.9970   |
| **Scratch CNN** | -             | 0.9888     | 0.9399   | 0.9442   |

## 🧠 Techniques Used

- `torchvision.models` with pretrained weights  
- Transfer learning with classifier replacement  
- Fine-tuning last 20 layers  
- Custom PyTorch Dataset class  
- Data augmentations: rotation, color jitter, flip  
- Accuracy monitoring with `matplotlib` plots

