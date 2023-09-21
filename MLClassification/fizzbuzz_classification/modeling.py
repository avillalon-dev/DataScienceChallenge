import os
import numpy as np
from typing import Sequence

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

def weights_for_imbalance_classes(labels: np.ndarray) -> np.ndarray:
    """
    Calculate class weights to address class imbalance.

    Args:
        labels (np.ndarray): Array of class labels.

    Returns:
        np.ndarray: Array of weights for each class.
    """
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
    """
    Train a Scikit-Learn estimator and optionally save the model to a file.

    Args:
        estimator (BaseEstimator): Scikit-Learn classifier object.
        X: Features for training.
        y: Labels for training.
        params (dict): Hyperparameters for the estimator (optional).
        fit_params: Additional fitting parameters (optional).

    Returns:
        BaseEstimator: Trained classifier.
    """
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
    """
    Test a trained classifier and print accuracy and classification report.

    Args:
        model (BaseEstimator): Trained Scikit-Learn classifier.
        X: Features for testing.
        y: True labels for testing.

    Returns:
        float: Accuracy of the classifier.
        np.ndarray: Confusion matrix.
    """
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
    """
    Tune hyperparameters for multiple classifiers using GridSearchCV.

    Args:
        estimators (Sequence[BaseEstimator]): Sequence of Scikit-Learn classifiers.
        param_grid (Sequence[dict]): Sequence of parameter grids for each estimator.
        X_train: Features for training.
        y_train: Labels for training.
        X_test: Features for testing.
        y_test: True labels for testing.
        cv: Number of cross-validation folds.
        verbose: Verbosity level for GridSearchCV.
        fit_params: Additional fitting parameters (optional).

    Returns:
        list: List of best hyperparameters for each classifier.
    """
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
    Train and evaluate multiple Scikit-Learn classifiers.

    Args:
        estimators (Sequence[BaseEstimator]): Sequence of Scikit-Learn classifiers.
        X_train: Features for training.
        y_train: Labels for training.
        X_test: Features for testing (optional).
        y_test: True labels for testing (optional).
        params (Sequence[dict]): Sequence of hyperparameter dictionaries for each classifier (optional).
        fit_params: Additional fitting parameters (optional).

    Returns:
        list: List of trained classifiers.
        list: List of dictionaries containing classifier information.
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

def cross_validate_classifiers(estimators: Sequence[BaseEstimator], X, y, k, fit_params=None):
    """
    Cross-validate multiple classifiers and compute average accuracies.

    Args:
        estimators (Sequence[BaseEstimator]): Sequence of Scikit-Learn classifiers.
        X: Features for cross-validation.
        y: True labels for cross-validation.
        k: Number of cross-validation folds.
        fit_params: Additional fitting parameters (optional).

    Returns:
        list: List of dictionaries containing classifier information and average accuracies.
    """
    cv_info = []
    for e, estimator in enumerate(estimators):
        estimator_name = str(type(estimator).__name__)
        # Cross validate the classifier
        cv_results = cross_validate(estimator, X, y, fit_params=fit_params, return_train_score=True, return_estimator=True)
        cv_info.append({'name': estimator_name, 'train_accuracy': np.mean(cv_results['train_score']), 'test_accuracy': np.mean(cv_results['test_score'])})
        
    return cv_info

def get_best_estimator(classifiers_info: Sequence[dict]):
    """
    Identify the best classifier based on test accuracy.

    Args:
        classifiers_info (Sequence[dict]): Sequence of classifier information dictionaries.

    Returns:
        tuple: Best classifier name, index, and test accuracy.
    """
    # Extract testing accuracies
    test_accuracies = [clf['test_accuracy'] for clf in classifiers_info]
    # Find the best classifier based on test accuracy
    best_classifier_index = np.argmax(test_accuracies)
    best_classifier = classifiers_info[best_classifier_index]['name']
    best_test_accuracy = test_accuracies[best_classifier_index]
    
    return (best_classifier, best_classifier_index, best_test_accuracy)