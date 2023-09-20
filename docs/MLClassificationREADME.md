# Machine learning classification
ML FizzBuzz test to build classification models.

 ## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Description](#data-description)
3. [Classification Model](#classification-model)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Results and Insights](#results-and-insights)
6. [Conclusion](#conclusion)
7. [References](#references)

## 1. Project Overview <a name="project-overview"></a>
This repository provides a machine learning test to classify natural numbers into four classes: “None”, “Fizz”, “Buzz”, and “FizzBuzz”.


## 2. Data Description <a name="data-description"></a>
The dataset was created by generating natural numbers and label them according to an specific criteria. The dataset consists of one file with two columns: one for the number and another one for the label.

## 3. Classification Model <a name="classification-model"></a>
To address the problem, a Support Vector Machine (SVM) model was considered. SVM is a powerful and versatile machine learning model, capable of performing linear or nonlinear classification, seeking the optimal decision boundary for the separation of distinct classes of data points. The core concept of SVMs involves the establishment of a decision boundary that maximizes the margin—the distance between the nearest data points belonging to distinct classes. Consider a dataset comprising two classes that can be divided by a straight line. In this scenario, the best line (referred to as the decision boundary) is determined by SVMs. This line is not only responsible for class separation but also ensures the maximum distance is maintained from the nearest data points of each class. This approach guarantees robust performance when applied to new data.

## 4. Evaluation Metrics <a name="evaluation-metrics"></a>
The model was evaluated using accuracy as metric. The best model was selected based on higher accuracy values.

## 5. Results and Insights <a name="results-and-insights"></a>
Prediction plots for electricity demand forecast from the last year in the data were provided. It was found that purely seasonal models were insufficient, and regression models were necessary to model weather relationships. Challenges were encountered in fitting the demand during summer. Other challenges were discussed related to this problem.

## 6. Conclusion <a name="conclusion"></a>


## 7. References <a name="references"></a>
[Géron, A. (2022). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (3rd ed.). O'Reilly Media, Inc.](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)

