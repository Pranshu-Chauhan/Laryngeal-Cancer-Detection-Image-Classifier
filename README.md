# Laryngeal Cancer Detection using Deep Learning

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Approach 1: Neural Network Approach](#approach-1-neural-network-approach)
  - [Preprocessing](#preprocessing)
  - [Feature Extraction](#feature-extraction)
  - [Model Training](#model-training)
  - [Results](#results)
- [Approach 2: Ensemble Learning Approach](#approach-2-ensemble-learning-approach)
  - [Preprocessing](#preprocessing-1)
  - [Feature Extraction](#feature-extraction-1)
  - [Base Models Training](#base-models-training)
  - [Meta Classifier](#meta-classifier)
  - [Results](#results-1)
- [Additional Experiments](#additional-experiments)
- [Conclusion](#conclusion)
- [References](#references)

## Overview

This repository contains the implementation of three different approaches for detecting laryngeal cancer using deep learning techniques. The dataset used for this project can be accessed [here](https://zenodo.org/records/1003200). The three approaches include:

1. Neural Network Approach
2. Ensemble Learning Approach
3. DenseNet201-based Approach

Each approach includes detailed preprocessing steps, feature extraction, model training, and evaluation metrics.

## Dataset

The dataset used for this project is available on Zenodo: [Laryngeal Cancer Dataset](https://zenodo.org/records/1003200). It includes images of laryngeal tissue classified into four categories: Healthy (He), Hypertrophic Vessels (Hbv), Leukoplakia (Le), and Intrapapillary Capillary Loops (IPCL).

## Approach 1: Neural Network Approach

### Preprocessing

1. Load the dataset.
2. Apply Median Filter followed by Gaussian Filter to remove noise from the images.
3. Display some images before and after filtering.

### Feature Extraction

- Features extracted from four pre-trained CNN models: VGG, Inception, DenseNet, and EfficientNet.
- Total extracted features: 46,976.
- Top 200 features selected for training.

### Model Training

- Neural network with 6 layers (5 dense layers and 1 SoftMax layer).
- Batch normalization and dropout techniques applied.
- Model tuning for optimal performance.

### Results

- Achieved an accuracy of 99.1%.
- Classification summary and confusion matrix are provided.

## Approach 2: Ensemble Learning Approach

### Preprocessing

1. Load the dataset.
2. Apply Median Filter followed by Gaussian Filter to remove noise from the images.
3. Display some images before and after filtering.

### Feature Extraction

- Features extracted from four pre-trained CNN models: VGG, Inception, DenseNet, and EfficientNet.
- Total extracted features: 46,976.

### Base Models Training

- Train Random Forest Classifier, Support Vector Classifier, and K-Neighbors Classifier.
- Stack predictions for the train and test sets.

### Meta Classifier

- Logistic regression used as the meta classifier on the stacked features.

### Results

- Achieved an accuracy of 99.24%.
- Classification summary and confusion matrix are provided.

## Approach 3: DenseNet201

### Preprocessing
- Load the dataset.
- Apply Median Filter followed by Gaussian Filter to remove noise from the images.
- Display some images before and after filtering.

### Base Models Training
- Import the DenseNet201 model and drop the inbuilt last classification layer.
- Add additional layers:
   - Average Max Pooling layer
   - Dense layer with 1024 neurons
   - Fully connected SoftMax layer with 4 neurons for the final 4 classifications
- Unfreeze the last 5 layers for weight updates during training.
- Train the model using StratifiedKFold cross-validation.

### Results
- Achieved an accuracy of 99.42%.
- Classification summary and confusion matrix are provided.

## Additional Experiments

- CLAHE Technique + NN-Approach: Accuracy 90%
- Gamma Correction + NN-Approach: Accuracy 97%
- Image Sharpening + NN-Approach: Accuracy 96%
- NN-Approach without feature selection: Accuracy 98.8%
- Feature Extraction using Local Binary Pattern + STAT features + DenseNet Features: Accuracy 96%
- Recursive Feature Elimination using Random Forest was computationally expensive and not implemented.

## Results

| Approach                         | Accuracy (%) |
|----------------------------------|--------------|
| Neural Network                   | 99.1         |
| Ensemble Learning                | 99.24        |
| DenseNet201 Model                | 99.42        |
| CLAHE Technique + NN Approach    | 90           |
| Gamma Correction + NN Approach   | 97           |
| Image Sharpening Technique + NN Approach | 96   |
| NN Approach without Feature Selection | 98.8  |
| Feature Extraction (LBP + STAT + DenseNet) | 96 |
| Recursive Feature Elimination    | Not implemented |


## Conclusion

This project demonstrates the effectiveness of different deep learning approaches for the detection of laryngeal cancer. The neural network and ensemble learning approaches achieved high accuracy, indicating their potential for clinical applications.

## References

- Moccia, Sara, et al. "Confident texture-based laryngeal tissue classification for early stage diagnosis support." JOURNAL OF MEDICAL IMAGING 4.03 (2017): 1-10.
- This project refers to the research paper: "An improved approach for initial stage detection of laryngeal cancer using effective hybrid features and ensemble learning method.”
- [Dataset](https://zenodo.org/records/1003200)

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for any bugs, feature requests, or improvements.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact
For any questions or feedback, please contact:

Pranshu Chauhan  
[Email](mailto:pranshu26092003@example.com)  
[LinkedIn](https://www.linkedin.com/in/pranshu-chauhan-72921b231)  
[GitHub](https://github.com/Pranshu-Chauhan)

