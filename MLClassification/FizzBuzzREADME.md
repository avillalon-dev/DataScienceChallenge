# Machine learning FizzBuzz classification
ML FizzBuzz test to build classification models.

![Pipeline](./fizzbuzz_classification/images/datapipeline.png)

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Description](#data-description)
3. [Data Transformation](#data-transformation)
3. [Classification Models](#classification-models)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Results and Insights](#results-and-insights)
6. [Conclusions](#conclusions)
7. [References](#references)

## 1. Project Overview <a name="project-overview"></a>
This repository provides a machine learning test to classify natural numbers into four classes: “None”, “Fizz”, “Buzz”, and “FizzBuzz”.

**Requirements:**
- Train any classification algorithm (E.g. support vector machine ) to solve the test as a classic supervised classification problem with four classes.
- Build your own dataset of any length, any preprocessing step in the input data is allowed.
- Also, provide the accuracy score obtained by the model with the test data (numbers from 1 - 100).
- Provide a brief theoretical description of the designed model and data pipeline.
- Share the implementation and documentation of the project using GitHub.

**Extra Points:**
- Perform a ten folds cross-validation using different classification algorithms and select the best among them.
- Update the documentation to support the selection (or not) of a new algorithm.
- Publish the trained model as a web service.
- Create unit tests for the implementation with a test coverage >=80%.

*The required theoretical description should be put with the other files from the project, could be in the README or other documentation format.*


## 2. Data Description <a name="data-description"></a>
The dataset was created by generating natural numbers and label them according to an specific criteria. The dataset consists of one file with two columns: one for the number and another one for the label.
- Data generation: Write a program that given the numbers from 1 to 100 print “None” for each number. But for multiples of three print “Fizz” instead of “None” and for the multiples of five print “Buzz”. For numbers which are multiples of both three and five print “FizzBuzz”.


## 3. Data Transformation <a name="data-transformation"></a>
In the data transformation, the primary objective is to ready the dataset for modeling, which involves feature engineering to extract features from the natural numbers taking into account class-related attributes. The generated features are calculated by treating each class as a time series, where the natural number is regarded as the time instant. Consequently, the smallest periods (greater than one) for each time series are computed. To indicate alignment with a specific time series, the modulus of each time instant is calculated concerning all periods derived from all time series. This process results in the generation of "period-based features" from the original data.
Additionally, normalization techniques can be used when necessary to standardize the data, ensuring it aligns with the requirements of the selected models and facilitating robust classification performance.


## 4. Classification Models <a name="classification-models"></a>
To address the problem, various classification models were considered, including:
- Support Vector Machine (SVM): SVM is a powerful and versatile machine learning model, capable of performing linear or nonlinear classification, seeking the optimal decision boundary for the separation of distinct classes of data points. The core concept of SVMs involves the establishment of a decision boundary that maximizes the margin—the distance between the nearest data points belonging to distinct classes. Consider a dataset comprising two classes that can be divided by a straight line. In this scenario, the best line (referred to as the decision boundary) is determined by SVMs. This line is not only responsible for class separation but also ensures the maximum distance is maintained from the nearest data points of each class. This approach guarantees robust performance when applied to new data.
- Decision tree (DT): The DT classifier is a versatile machine learning algorithm widely used in classification problems. At its core, it operates by recursively partitioning the feature space into subsets, making decisions at each node based on feature values. This process continues until leaf nodes represent class labels. Decision trees are valued for their interpretability, as they provide clear, human-readable rules for classification. They have the ability to handle both categorical and numerical features, and adapt to complex decision boundaries.

You have the flexibility to explore and test other classification models in addition to the ones provided in this project. To do so, follow these steps:

- Open the [notebook](./fizzbuzz_notebook.ipynb) associated with this project.
- Navigate to the `Model Architectures` section within the notebook.
- Import the desired classification model that you want to test. For example, the following code was used to import the SVM and Decision Tree classifiers:
```python
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
classifiers = [
    SVC(),                    # Support vector machine classifier
    DecisionTreeClassifier()  # Decision tree classifier
]
```


## 5. Evaluation Metrics <a name="evaluation-metrics"></a>
The models were evaluated using accuracy as metric. The best model was selected based on higher accuracy values.


## 6. Results and Insights <a name="results-and-insights"></a>
Experiments were conducted using two datasets: the first containing the original data with natural numbers, and the second comprising features generated during data transformation. Both classification models were evaluated and compared on both datasets.
The outcomes from the original dataset suggest that the decision tree classifier outperformed the SVM. However, with the second dataset, both models achieved a perfect accuracy score.
The results strongly imply that the feature set obtained after transformation leads to superior performance.

**Results with natural numbers**

![Classifiers in original data](./fizzbuzz_classification/images/originaldata_classifiers_comparison.png)

**Best classifier confusion matrix**
![Best classifier in original data confusion matrix](./fizzbuzz_classification/images/originaldata_bestclassifier_cm.png)

**K-fold cross validation with k = 10 using natural numbers**

![Classifiers cv in original data](./fizzbuzz_classification/images/originaldata_classifiers_cv.png)

**Results with period-based features**

![Classifiers in preprocessed data](./fizzbuzz_classification/images/preprocesseddata_classifiers_comparison.png)

**Best classifier confusion matrix**

![Best classifier in preprocessed data confusion matrix](./fizzbuzz_classification/images/preprocesseddata_bestclassifier_cm.png)

**K-fold cross validation with k = 10 using period-based features**

![Classifiers cv in preprocessed data](./fizzbuzz_classification/images/preprocesseddata_classifiers_cv.png)


## 7. Conclusions <a name="conclusions"></a>
The problem in this project focused on classifying natural numbers into four classes: "None," "Fizz," "Buzz," and "FizzBuzz." Two datasets were employed: one with original natural numbers and the period-based features generated during data transformation. "Period-based features" are derived by calculating the modulus of each natural number concerning the smallest periods obtained from all time series, treating the natural numbers as time instants in the time series represented by the classes. After training and testing the classification model, the results indicated that the DT classifier outperformed the SVM when using the original dataset. However, when utilizing the transformed dataset, both models achieved perfect accuracy scores. This underscores the crucial role of data transformation and feature engineering in enhancing classification performance.


## 8. References <a name="references"></a>
- [Géron, A. (2022). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (3rd ed.). O'Reilly Media, Inc.](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)

