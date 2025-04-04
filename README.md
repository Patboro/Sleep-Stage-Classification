# Sleep Stage Classification

This repository contains a deep learning-based pipeline for classifying sleep stages from physiological data. The system leverages a ResNet architecture implemented in PyTorch to automatically identify different REM and non-REM stages.

## Project Overview

The goal of this project is to develop a robust and efficient model capable of accurately classifying sleep stages based on input data. The pipeline encompasses data preprocessing, model training, evaluation, and visualization components, providing a comprehensive framework for sleep stage classification tasks.

## Features

- **Data Preprocessing**: Efficient handling and transformation of raw physiological data into model-ready inputs.
- **ResNet-based Model**: Utilization of a Residual Network (ResNet) architecture tailored for time-series classification tasks.
- **Training and Evaluation Pipeline**: Automated processes for model training, validation, and performance assessment.
- **Visualization Tools**: Functions for visualizing training progress, model performance, and classification outputs.

## Repository Structure

```
.
├── ResNet/                   # Directory containing ResNet model implementation
│   ├── resnet.py             # Definition of ResNet architecture
│   └── ...
├── data_preprocessing/       # Directory for data preprocessing scripts
│   ├── preprocess.py         # Script for data cleaning and transformation
│   └── ...
├── main.py                   # Main script to run training and evaluation
├── requirements.txt          # List of required Python packages
└── README.md                 # Project documentation
```

## Installation

To set up the environment for this project, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Patboro/Sleep-Stage-Classification.git
   ```

2. **Navigate to the project directory**:

   ```bash
   cd Sleep-Stage-Classification
   ```

3. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

Ensure that your dataset is organized appropriately and that the data preprocessing scripts are configured to locate and process your data correctly. Modify the `preprocess.py` script in the `data_preprocessing` directory as needed to accommodate your dataset's structure and format.

### Training the Model

To train the sleep stage classification model, execute the `main.py` script:

```bash
python main.py
```

This script will initiate the training process, including data loading, model training, and evaluation. Monitor the console output for training progress and performance metrics.

### Evaluation and Visualization

Post-training, utilize the visualization tools provided to assess model performance. These tools can generate plots of training and validation loss, accuracy metrics, and confusion matrices to aid in interpreting the model's effectiveness.


## License

This project is licensed under the MIT License. 

