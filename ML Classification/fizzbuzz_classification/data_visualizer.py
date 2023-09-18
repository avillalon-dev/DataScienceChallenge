import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def plot_features(data: pd.DataFrame):
    # Plot features using seaborn
    plt.figure()  
    sns.pairplot(data, hue='label')
    
    # Plot features using scatter plot
    plt.figure()
    encoder = LabelEncoder()
    data['label'] = encoder.fit_transform(data['label'])
    plt.scatter(data['number'], data['label'], c=data['label'])
    plt.xlabel('Numbers')
    plt.ylabel('Labels')
    plt.title('Numbers vs Labels')
    plt.show()