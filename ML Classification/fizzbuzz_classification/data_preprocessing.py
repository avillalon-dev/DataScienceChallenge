import pandas as pd
import numpy as np
import math

from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer to calculate the modulus of a number based on a divisor
class ModulusTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, divisor=1):
        self.divisor = divisor
    
    def fit(self, X, y=None):
        # The fit method is used for any required setup or learning.
        # In this case, there's no learning, so we just return self.
        return self
    
    def transform(self, X):
        return [x % self.divisor for x in X]
    
    def inverse_transform(self, X):
        return None
    
    def fit_transform(self, X, y=None):
        # Combines the fitting and transformation steps
        self.fit(X, y)
        return self.transform(X)

def transform_series_to_frequency(data: pd.DataFrame, return_estimator = False):
    divisors = []
    for l in np.unique(data['label']):
        if l == 'None': continue
        idx = data['label'] == l
        features_l = data['number'][idx]
        diff = np.diff(features_l, n = 1, axis=0)
        divisors.append(find_common_denominators(diff))
        
    divisors = np.unique(flatten([divisors]))
    new_features = pd.DataFrame(index=data.index)
    estimators = []
    for divisor in divisors:
        estimator = ModulusTransformer(divisor)
        feature_mod = estimator.fit_transform(data['number'])
        estimators.append(estimator)
        new_features['number_mod' + str(divisor)] = feature_mod
        
    if return_estimator:
        return new_features, estimators

    return new_features

def flatten(arr):
    result = []
    for item in arr:
        if isinstance(item, np.ndarray) or isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

# Function to find prime factors of a number
def prime_factors(n):
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

# Function to find common unique denominators in a list of numbers
def find_common_denominators(numbers):
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