# Project-Neural-Classification-of-Erythrocyte-Anomalies

This project aims to design, train and validate a Convolutional Neural Network(CNN) model to perform binary classification on single-cell images to distinguish between healthy erythrocytes (red blood cells) and those containing a specific intracellular pathogen.

## The Dataset 
The datasets consists of segmented "patches" (Regions of Interest) extracted from thin blood smear slides stained with Giemsa. Each image contains a single cell. The images have varying resolutions and aspect ratios. 
The `train` set consist of approximately 22,000 labeled images:
- Sub-folder `negative`: Represents Healthy/Control samples.
- Sub-folder `positive`: Represents Infected/Anomalous samples

<p float="left">
  <figure style="display:inline-block; text-align:center; margin-right:20px;">
    <img src="positive_detection_example.png" width="320" alt="positive">
    <figcaption><em>example of positive detection</em></figcaption>
  </figure>
  <figure style="display:inline-block; text-align:center;">
    <img src="negative_detection_example.png" width="320" alt="negative">
    <figcaption><em>example of negative detection</em></figcaption>
  </figure>
</p>

## Design choices
### A. Data preprocessing & augmentation
The pipeline uses transforms.Compose and consist of following methods:
1. Resize to scale image into 256x256 resolution 
2. ToTensor 
3. Normalize with `mean=[0.485, 0.456, 0.406]` and `std=[0.229, 0.224, 0.225]`- in this project pretrained ResNet-18 will be used. Pretrained models expect inputs to follow the same statistical distribution they were trained on.