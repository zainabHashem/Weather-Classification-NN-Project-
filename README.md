# Weather Classification Neural Network Project

This project implements a deep learning solution for weather classification using both custom ResNet architecture and transfer learning approaches with Xception and DenseNet models. The project uses a subset of the BDD100K Weather Classification dataset to classify weather conditions in images.

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

## Requirements

```
# Add your requirements.txt content here
tensorflow>=2.0.0
numpy
pandas
scikit-learn
matplotlib
seaborn
```

## Project Structure

```
weather-classification/
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── models/
│   ├── custom_resnet.py
│   └── transfer_learning.py
├── utils/
│   ├── data_loader.py
│   └── metrics.py
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── model_evaluation.ipynb
└── README.md
```

## Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Weather-Classification-NN-Project.git
cd Weather-Classification-NN-Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
- Download from [Kaggle](https://www.kaggle.com/datasets/marquis03/bdd100k-weather-classification)
- Place the data in the `data/` directory

4. Train the models:
```bash
python train.py --model [resnet|xception|densenet] --epochs 50
```

5. Evaluate the models:
```bash
python evaluate.py --model [resnet|xception|densenet] --weights path/to/weights
```


## Contributing

Feel free to open issues or submit pull requests if you find any bugs or have suggestions for improvements.

