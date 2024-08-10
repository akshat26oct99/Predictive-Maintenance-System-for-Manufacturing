# Predictive Maintenance System for Manufacturing

This machine learning project focuses on reducing operational costs in manufacturing by proactively detecting potential machine failures. The project utilizes the Machine Predictive Maintenance Classification Dataset and compares the performance of Multi-Layer Perceptrons (MLP) implemented in both PyTorch and Matlab.

For a comprehensive understanding of the project, please refer to the summary provided in this repository.

## Overview

Predictive maintenance is revolutionizing the manufacturing industry by enabling the early detection of machinery failures. This project explores the challenge of multivariate time series classification using real-world operational data. The goal is to not only save costs but also extend the lifespan of machinery and enhance safety by identifying potential failures before they lead to accidents.

The study compares the effectiveness of two MLP implementations—one in PyTorch and the other in Matlab—using holdout validation to assess model accuracy. The results reveal that while the Python interpreter introduces significant overhead, requiring more fine-tuning and longer training times, the Matlab implementation is superior in both training speed and validation accuracy.

This project was completed as the capstone project for the Deep Learning course at City, University of London, earning a distinction.

## Implementation Comparison

This project compares the performance of two MLP implementations:

- **Multilayer Perceptron (MLP)**: A feedforward neural network commonly used for classification tasks.

## Dataset

The dataset, sourced from Kaggle and provided by HTW Berlin, is a synthetic representation of real predictive maintenance data from the industry. The full dataset contains 10,000 rows with 8 columns—6 feature columns and 2 label columns for binary and multi-class classification.

### Data Preprocessing Steps

1. **Initial Data Analysis**: Evaluated the distribution of predictors and assessed target variable imbalance.
2. **Feature Selection**: Selected relevant predictors based on domain knowledge.
3. **Encoding Categorical Variables**: Transformed categorical variables into one-hot encoded variables.
4. **Splitting**: Split the data into training and test sets with a 70-30 ratio.
5. **Oversampling**: Applied SMOTE-NC on the training set to balance the target variable. Undersampling was not feasible due to the extreme imbalance (237 to 9,663).
6. **Normalization**: Scaled numerical features to a 0-1 range to mitigate issues related to magnitude sensitivity.

## Models

The optimal model was achieved using an MLP with 2 hidden neurons, trained over 50 epochs with a learning rate of 0.3. ReLU was used as the activation function for the hidden layers, and a sigmoid activation function was applied in the output layer.

## Results

Key findings from the study include:

- **Training Time**: The Matlab MLP implementation significantly outperformed the PyTorch version, with training times of 52ms versus 250ms, respectively—a 5x difference.
- **Validation Accuracy**: The Matlab model achieved a validation accuracy of 0.99999, compared to 0.966 for the PyTorch model, highlighting a significant implementation difference.

## Remarks

- The project was optimized for GPU execution in Google Colab.
- Due to the large size of the dataset, it is not included in this repository. However, it can be downloaded from the link provided and should be placed in a dedicated "data" folder to seamlessly run the code.
