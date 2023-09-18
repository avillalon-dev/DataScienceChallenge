import os
from typing import Literal

import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self):
        self.data_path = os.getcwd() + '/fizzbuzz_classification/data/'
        if not os.path.isdir(self.data_path):
            os.mkdir(self.data_path)
        self.data_filename = 'fizzbuzz_data'
        
    def load_data(self, set: Literal['original', 'preprocessed'], set_length: int = 100, return_df = True) -> tuple[np.ndarray, np.ndarray]:
        
        # Specify file name
        filename = str.join('_', [self.data_filename, str(set_length), set]) + '.csv'
        # Verify existence of data file
        assert os.path.isfile(self.data_path + filename), 'Data file does not exist.'
        
        # Read data from file
        df_data = pd.read_csv(self.data_path + filename)
        df_data['label'] = df_data['label'].astype(str)
        if return_df:
            return  df_data
        
        # Convert dataframe to numpy arrays for compatibility with classification algorithms
        labels = np.array(df_data.pop('label'))
        features = np.array(df_data)
        
        return features, labels
    
    def generate_data(self, set_length: int = 100) -> None:
        
        # Specify file name
        filename = str.join('_', [self.data_filename, str(set_length), 'original']) + '.csv'
        if os.path.isfile(self.data_path + filename):
            return # Return if data already exists

        # Initialize empty lists to store features and labels
        features = []
        labels = []

        # Define the range of numbers (e.g., 1 to set_length)
        for num in range(1, set_length + 1):
            # Initialize feature vector for each number
            features_vector = []

            # Add values to the features vector
            features_vector.append(num) # Number
            # features_vector.append(num % 3) # If multiple of 3
            # features_vector.append(num % 5) # If multiple of 5

            # Append the features vector to the list of features
            features.append(features_vector)

            # Determine the label based on the FizzBuzz rules
            if num % 3 == 0 and num % 5 == 0:
                labels.append("FizzBuzz")
            elif num % 3 == 0:
                labels.append("Fizz")
            elif num % 5 == 0:
                labels.append("Buzz")
            else:
                labels.append("None")
        
        # Convert lists to numpy arrays for compatibility with classification algorithms
        features = np.array(features).reshape(-1, 1)
        labels = np.array(labels).reshape(-1, 1)
        
        # Save data to csv files
        df_data = pd.DataFrame(np.concatenate([features, labels], axis=1), columns=['number', 'label'])
        df_data.to_csv(os.path.join(self.data_path, str.join('_', [self.data_filename, str(set_length), 'original']) + '.csv'), index=False)