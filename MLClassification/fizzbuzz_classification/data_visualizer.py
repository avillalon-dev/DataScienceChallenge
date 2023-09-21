import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay

def plot_features(data: pd.DataFrame):
    # Plot features using seaborn
    plt.figure()  
    sns.pairplot(data, hue='label')
    
    try:
        # Plot features using scatter plot
        plt.figure()
        encoder = LabelEncoder()
        labels = encoder.fit_transform(data['label'])
        unique_labels = np.unique(labels)
        plt.scatter(data['number'], data['label'], c=labels)
        plt.xlabel('Numbers')
        plt.ylabel('Labels')
        plt.yticks(unique_labels, data['label'].unique())
        plt.title('Numbers vs Labels')
        plt.show()
    except:
        pass
    
def plot_train_results(classifiers_info: Sequence[dict], labels):
    # for info in estimators_info:
    #     confusion_matrix = info['confusion_matrix']
    #     cmD = ConfusionMatrixDisplay(confusion_matrix)
    #     cmD.plot()
    #     plt.title(info['name'])
        
    # Extract classifier names, training accuracies, and testing accuracies
    classifier_names = [clf['name'] for clf in classifiers_info]
    train_accuracies = [clf['train_accuracy'] for clf in classifiers_info]
    test_accuracies = [clf['test_accuracy'] for clf in classifiers_info]

    # Plot the training and testing accuracies for all classifiers in a single bar plot
    x = np.arange(len(classifier_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    train_accuracy_bars = ax.bar(x - width/2, train_accuracies, width, label='Train Accuracy')
    test_accuracy_bars = ax.bar(x + width/2, test_accuracies, width, label='Test Accuracy')

    ax.set_xlabel('Classifiers')
    ax.set_ylabel('Accuracy')
    ax.set_title('Classifier Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(classifier_names, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.show()

    # Create separate figures with subplots for confusion matrices
    for i, clf in enumerate(classifiers_info):
        cm = clf['confusion_matrix']
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

        # Create a separate figure for each classifier
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap=plt.cm.Blues)
        ax.set_title(f'{clf["name"]} Confusion Matrix')

        plt.tight_layout()
        plt.show()