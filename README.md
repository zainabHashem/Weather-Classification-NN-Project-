# Weather Classification Neural Network Project

This project implements a deep learning solution for weather classification using both custom ResNet architecture and transfer learning approaches with Xception and DenseNet models. The project uses a subset of the BDD100K Weather Classification dataset to classify weather conditions in images.

## Development Environment

- **Platform**: Kaggle Notebooks
- **Hardware**: GPU (P100/T4)
- **Runtime Type**: Python with GPU acceleration
- **Framework**: TensorFlow/Keras

## Dataset

- **Source**: [BDD100K Weather Classification Dataset](https://www.kaggle.com/datasets/marquis03/bdd100k-weather-classification)
- **Sample Size**: 30,000 images
- **Classes**: 
  - Clear
  - Overcast
  - Partly Cloudy
  - Rainy
  - Snowy
  - Unknown

## Model Architectures

### Custom ResNet Implementation
- Built from scratch implementation of ResNet architecture
- Includes residual connections and deep convolutional layers
- Trained end-to-end on the weather classification task

### Transfer Learning Models
1. **Xception**
   - Pre-trained weights from ImageNet
   - Fine-tuned for weather classification
   - Modified top layers for 6-class classification

2. **DenseNet**
   - Pre-trained weights from ImageNet
   - Leverages dense connectivity pattern
   - Adapted for weather classification task

## Performance Metrics

The following metrics are used to evaluate model performance:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC Curves
- AUC Scores

### Visualizations
- Confusion Matrix
- ROC Curves
- Training/Validation Loss & Accuracy Curves

## Kaggle Setup

1. Create a new notebook:
   - Go to "Code" > "New Notebook"
   - Select "GPU" as the accelerator
   - Enable internet access if needed

2. Dataset Integration:
```python
# Add dataset to your notebook
# Using the Kaggle GUI: Add data > Search for "bdd100k-weather-classification"
```

3. Required Package Imports:
```python
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
```

## Project Structure (Kaggle Notebook)

```
📦 weather-classification-notebook
 ┣ 📂 input
 ┃ ┗ 📂 bdd100k-weather-classification
 ┣ 📂 working
 ┃ ┣ 📂 models
 ┃ ┣ 📂 results
 ┃ ┗ 📂 visualizations
 ┗ 📜 weather-classification.ipynb
```

## Implementation Steps

1. Data Preparation:
```python
# Load and preprocess the data
# Split into train/validation/test sets
# Apply necessary augmentation
```

2. Model Training:
```python
# Train custom ResNet
# Fine-tune Xception
# Fine-tune DenseNet
```

3. Evaluation:
```python
# Generate confusion matrices
# Calculate metrics
# Plot ROC curves
```

## Usage Notes

- Make sure to select GPU accelerator in Kaggle notebook settings
- Save intermediate model weights using `model.save()`
- Download important visualizations and results to local machine
- Use Kaggle's version control to track changes

## References

- [BDD100K Dataset Paper](https://arxiv.org/abs/1805.04687)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Xception Paper](https://arxiv.org/abs/1610.02357)
- [DenseNet Paper](https://arxiv.org/abs/1608.06993)

## Acknowledgments

- BDD100K dataset creators
- Kaggle for providing GPU resources
