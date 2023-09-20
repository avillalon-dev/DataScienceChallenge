import os

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import pickle

# Train, save and load classifier
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
            
def test_estimator(model: BaseEstimator, X, y, display = True):
    # Predict labels on the test set
    print('Testing model')
    y_pred = model.predict(X)
    # Calculate accuracy and print classification report
    print('Evaluate model')
    accuracy = accuracy_score(y, y_pred)
    if display:
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:\n", classification_report(y, y_pred))
        # Add confusion matrix
        conf_mat = confusion_matrix(y, y_pred)
        cmD = ConfusionMatrixDisplay(conf_mat)
        cmD.plot()
    return accuracy
    
def tune_hyperparam(estimator: BaseEstimator, X_train, y_train, X_test, y_test, param_grid: dict, cv, verbose, fit_params = None):
    
    # Get the current package name
    package_name = __package__
    # Get the type of the estimator as a string
    estimator_name = str(type(estimator).__name__)
    # Use estimator_name to save the estimator
    file_name = os.path.join(os.getcwd(), f"{package_name}/models/{estimator_name}_best_params.pkl")
    
    if not os.path.exists(file_name):
        # Create a GridSearchCV instance
        grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, verbose=verbose)

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
        return grid_search.best_params_
    else:
        with open(file_name, 'rb') as fin:
            return pickle.load(fin)

def train_and_test_classifier(estimator: BaseEstimator, X_train, y_train, X_test=None, y_test=None, params=None, fit_params=None):
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


    # Fit the classifier to the training data
    trained_estimator = train_estimator(estimator, X_train, y_train, params, fit_params)

    # Calculate training accuracy
    train_accuracy = test_estimator(trained_estimator, X_train, y_train, display=False)

    if X_test is not None and y_test is not None:
        # Calculate testing accuracy if testing data is provided
        test_accuracy = test_estimator(trained_estimator, X_test, y_test)
        return trained_estimator, train_accuracy, test_accuracy
    else:
        return trained_estimator, train_accuracy