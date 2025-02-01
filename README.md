# Plant Species Identification and Description Extraction

This project aims to identify and extract descriptions of plant species using a combination of YOLO for detection and CNN models for classification. The application interface is implemented using streamlit.

## Project Overview

The project is divided into the following main components:

1. **YOLO Model**: Detects flowers, leaves, and fruits in the input images.
2. **CNN Models**: Classifies the detected regions into specific plant species.
   - **FruitModel**: ResNet50 CNN model for fruit classification.
   - **FlowerModel**:CNN model for flower classification.
3. **Description Extractor**: Extracts detailed descriptions of the identified plant species.

## Description
---

## Workflow

1. **Detection**: The input image is processed by the main YOLO model to detect flowers, leaves, and fruits.
2. **Classification**: Based on the highest confidence score of the bounding boxes from YOLO, the detected region is fed into the respective CNN model (FruitModel or FlowerModel).
3. **Description Extraction**: The identified plant species are passed to the Description Extractor to derive detailed descriptions.

## Technology Stack

- **Deep Learning**: Tensorflow for implementing the CNN models.
- **Object Detection**: YOLO (You Only Look Once) model.

## Dataset References
1. **Flower Dataset and Fruit Dataset**: self created for object detection
2. **Fruit Dataset**: (Fruits)[https://www.kaggle.com/datasets/moltean/fruits]
3. **Flower Dataset**:(Flowers)[https://www.kaggle.com/datasets/imsparsh/flowers-dataset]
