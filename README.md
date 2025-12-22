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

## 📊 Detailed Report
For complete documentation including model architecture, training details, hyperparameters, results, and visualizations, please refer to [REPORT.md](REPORT.md).



