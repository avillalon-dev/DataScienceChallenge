import os
import numpy as np
from typing import Sequence

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

def weights_for_imbalance_classes(labels: np.ndarray) -> np.ndarray:
    unique_labels, count_labels = np.unique(labels, return_counts=True)
    samples_count = labels.shape[0]
    weights = np.zeros_like(labels)
    weights_labels = np.zeros_like(count_labels, dtype=float)
    for i, l in enumerate(unique_labels):
        weights_labels[i] = samples_count / count_labels[i]*2 # Calculate weight for each class
    for i, l in enumerate(unique_labels):
        idx = labels == l
        weights[idx] = weights_labels[i]
        
    return weights

def train_estimator(estimator: BaseEstimator, X, y, params = None, fit_params = None):
    # Set hyperparameters if provided
    if params is not None:
        estimator.set_params(**params)
        
    # Get the current package name
    package_name = __package__
    # Get the type of the estimator as a string
    estimator_name = str(type(estimator).__name__)
    # Use estimator_name to save the estimator
    file_name = os.path.join(os.getcwd(), f"{package_name}/models/{estimator_name}_classifier.pkl")
    
    if not os.path.exists(file_name):
        print('Training model')
        # Fit the classifier to the training data with fit_params
        if fit_params is not None:
            estimator.fit(X, y, **fit_params)
        else:
            estimator.fit(X, y)
        print('Saving model')
        with open(file_name, 'wb') as fout:
            pickle.dump(estimator, fout)  # Save model
    else:
        print('Loading model')
        with open(file_name, 'rb') as fin:
            estimator = pickle.load(fin)  # Load model
            
    return estimator
            
def test_estimator(model: BaseEstimator, X, y):
    # Predict labels on the test set
    print('Testing model')
    y_pred = model.predict(X)
    # Calculate accuracy and print classification report
    print('Evaluate model')
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y, y_pred))
    # Add confusion matrix
    conf_mat = confusion_matrix(y, y_pred)
    return accuracy, conf_mat
    
def tune_hyperparam(estimators: Sequence[BaseEstimator], param_grid: Sequence[dict], X_train, y_train, X_test, y_test, cv, verbose, fit_params = None):
    
    best_params = []
    for e, estimator in enumerate(estimators):
        # Get the current package name
        package_name = __package__
        # Get the type of the estimator as a string
        estimator_name = str(type(estimator).__name__)
        # Use estimator_name to save the estimator
        file_name = os.path.join(os.getcwd(), f"{package_name}/models/{estimator_name}_best_params.pkl")
        
        if not os.path.exists(file_name):
            # Create a GridSearchCV instance
            grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid[e], cv=cv, verbose=verbose)

            # Fit the grid search to the training data with fit_params
            if fit_params is not None:
                grid_search.fit(X_train, y_train, **fit_params)
            else:
                grid_search.fit(X_train, y_train)

            # Print the best hyperparameters
            print("Best Hyperparameters:", grid_search.best_params_)

            # Evaluate the model with the best hyperparameters
            best_svc = grid_search.best_estimator_
            accuracy = best_svc.score(X_test, y_test)
            print("Accuracy with Best Hyperparameters:", accuracy)
            
            # Save the best parameters to a file
            with open(file_name, 'wb') as fout:
                pickle.dump(grid_search.best_params_, fout)
            best_params.append(grid_search.best_params_)
        else:
            with open(file_name, 'rb') as fin:
                best_params.append(pickle.load(fin))
    
    return best_params

def train_and_test_classifier(estimators: Sequence[BaseEstimator], X_train, y_train, X_test=None, y_test=None, params: Sequence[dict]=None, fit_params=None):
    """
    Train and evaluate a Scikit-Learn classifier.

    Parameters:
    - estimator: Scikit-Learn classifier object (e.g., SVC(), RandomForestClassifier(), etc.).
    - X_train: Training features.
    - y_train: Training labels.
    - X_test: Testing features (optional).
    - y_test: Testing labels (optional).
    - params: Dictionary of hyperparameters (optional).

    Returns:
    - trained_estimator: Trained classifier.
    - train_accuracy: Accuracy on the training set.
    - test_accuracy: Accuracy on the testing set (if provided).
    """

    trained_estimators = []
    train_info = []
    for e, estimator in enumerate(estimators):
        estimator_name = str(type(estimator).__name__)
        # Fit the classifier to the training data
        trained_estimator = train_estimator(estimator, X_train, y_train, params[e], fit_params)
        trained_estimators.append(trained_estimator)
        # Calculate training accuracy
        train_accuracy, _ = test_estimator(trained_estimator, X_train, y_train)
        if X_test is not None and y_test is not None:
            # Calculate testing accuracy if testing data is provided
            test_accuracy, confusion_matrix = test_estimator(trained_estimator, X_test, y_test)
            train_info.append({'name': estimator_name, 'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy, 'confusion_matrix': confusion_matrix})
        else:
            train_info.append({'name': estimator_name, 'train_accuracy': train_accuracy})
        
    return trained_estimators, train_info

def get_best_estimator(classifiers_info: Sequence[dict]):
    # Extract testing accuracies
    test_accuracies = [clf['test_accuracy'] for clf in classifiers_info]
    # Find the best classifier based on test accuracy
    best_classifier_index = np.argmax(test_accuracies)
    best_classifier = classifiers_info[best_classifier_index]['name']
    best_test_accuracy = test_accuracies[best_classifier_index]
    
    return (best_classifier, best_classifier_index, best_test_accuracy)