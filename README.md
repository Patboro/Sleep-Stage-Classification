# Sleep Stage Classification

This repository contains a deep learning-based pipeline for classifying sleep stages from Electroencephalography (EEG) data. The system leverages a ResNet architecture implemented in PyTorch to automatically identify different REM and non-REM stages.

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


## License

This project is licensed under the MIT License. 

