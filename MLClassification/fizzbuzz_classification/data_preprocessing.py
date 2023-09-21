import pandas as pd
import numpy as np
import math

from sklearn.base import BaseEstimator, TransformerMixin

class ModulusTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to calculate the modulus of a number based on a divisor.

    This transformer takes an input array and calculates the modulus of each element
    in the array with a specified divisor.

    Parameters:
    divisor (int, optional): The divisor to use for modulus calculation. Default is 1.

    Attributes:
    divisor (int): The divisor used for modulus calculation.

    Methods:
    - fit(self, X, y=None): Perform any required setup or learning (no learning in this case).
    - transform(self, X): Calculate the modulus of each element in X with the specified divisor.
    - inverse_transform(self, X): Not implemented for this transformer.
    - fit_transform(self, X, y=None): Combines the fitting and transformation steps.

    Returns:
    np.ndarray: An array of modulus values based on the input array and divisor.
    """

    def __init__(self, divisor=1):
        """
        Initialize the ModulusTransformer.

        Parameters:
        divisor (int, optional): The divisor to use for modulus calculation. Default is 1.
        """
        self.divisor = divisor
    
    def fit(self, X, y=None):
        """
        Fit the transformer (no learning in this case).

        Parameters:
        X (array-like): Input data.
        y: Ignored.

        Returns:
        self: Returns self.
        """
        return self
    
    def transform(self, X):
        """
        Transform the input array by calculating the modulus.

        Parameters:
        X (array-like): Input data.

        Returns:
        np.ndarray: An array of modulus values.
        """
        return X % self.divisor
    
    def inverse_transform(self, X):
        """
        Inverse transform is not implemented for this transformer.

        Parameters:
        X: Ignored.

        Returns:
        None
        """
        return None
    
    def fit_transform(self, X, y=None):
        """
        Fit the transformer and apply transformation.

        Parameters:
        X (array-like): Input data.
        y: Ignored.

        Returns:
        np.ndarray: An array of modulus values.
        """
        self.fit(X, y)
        return self.transform(X)

def transform_series_to_periods(data: pd.DataFrame, estimators: ModulusTransformer = None, return_estimator = False):
    """
    Transform a series of numbers in a DataFrame into frequency-based features.

    Args:
        data (pd.DataFrame): The input DataFrame containing 'label' and 'number' columns.
        return_estimator (bool, optional): Whether to return transformers. Defaults to False.

    Returns:
        pd.DataFrame: The transformed DataFrame with frequency-based features.
        list: List of transformers if return_estimator is True.
    """
    if estimators is None:
        divisors = []
        for l in np.unique(data['label']):
            if l == 'None': continue
            idx = data['label'] == l
            features_l = data['number'][idx]
            diff = np.diff(features_l, n = 1, axis=0)
            divisors.append(find_common_denominators(diff))
            
        divisors = np.unique(flatten([divisors]))
        estimators = []
        for divisor in divisors:
            estimator = ModulusTransformer(divisor)
            estimators.append(estimator)
    
    new_features = pd.DataFrame(index=data.index)
    for estimator in estimators:
        feature_mod = estimator.fit_transform(data['number'])
        new_features['number_mod' + str(estimator.divisor)] = feature_mod
        
    if return_estimator:
        return new_features, estimators

    return new_features

def flatten(arr):
    """
    Flatten a nested list or array.

    Args:
        arr (list or np.ndarray): The input list or array.

    Returns:
        list: The flattened list.
    """
    result = []
    for item in arr:
        if isinstance(item, np.ndarray) or isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

def prime_factors(n):
    """
    Find the prime factors of a given number.

    Args:
        n (int): The input number.

    Returns:
        set: Set of prime factors.
    """
    factors = set()
    # Divide n by 2 until it is even
    while n % 2 == 0:
        factors.add(2)
        n //= 2
    # After this, n must be odd, so we skip even numbers
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        while n % i == 0:
            factors.add(i)
            n //= i
    # If n is still greater than 1, it's a prime number itself
    if n > 1:
        factors.add(n)
    return factors

def find_common_denominators(numbers):
    """
    Find common unique denominators in a list of numbers.

    Args:
        numbers (list): The input list of numbers.

    Returns:
        list: List of common unique denominators.
    """
    numbers = flatten(numbers)
    if len(numbers) == 0:
        return []

    # Find the GCD of all numbers in the list
    gcd_result = numbers[0]
    for num in numbers[1:]:
        gcd_result = math.gcd(gcd_result, num)

    # Find the prime factors of the GCD
    gcd_factors = prime_factors(gcd_result)

    return list(gcd_factors)