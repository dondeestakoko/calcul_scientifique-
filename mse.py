import matplotlib.pyplot as plt
import numpy as np 
import csv
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from itertools import product
import pandas as pd


time = [0]
lapin = [1]
renard = [2]

param_grid = { 
    'alpha' : [1/3,  2/3, 1, 4/3],
    'beta' : [1/3,  2/3, 1, 4/3],
    'delta' : [1/3,  2/3, 1, 4/3],
    'gamma' : [1/3,  2/3, 1, 4/3]
}

step = 0.001

'''def mse(real_data, predicted_data):
    sum_errors = 0
    for index in range(1, len(real_data)):
        difference = real_data[index] - predicted_data[index]
        sum_errors =+ difference
    return sum_errors'''

def mse(real_data, predicted_data):
    # Check if both 'lapin' and 'renard' columns exist in both DataFrames
    required_columns = ['lapin', 'renard']
    for col in required_columns:
        if col not in real_data.columns or col not in predicted_data.columns:
            raise ValueError(f"Column '{col}' not found in both datasets.")
    
    # Initialize sum of errors for each column
    sum_errors = {'lapin': 0, 'renard': 0}
    
    # Iterate through the rows
    for index in range(len(real_data)):
        for col in required_columns:
            try:
                # Convert values to float and compute the difference
                real_value = float(real_data[col].iloc[index])
                predicted_value = float(predicted_data[col].iloc[index])
                sum_errors[col] += real_value - predicted_value
            except (ValueError, TypeError):
                # Skip non-numeric values and print a message
                print(f"Skipping non-numeric data in column '{col}' at index {index}: "
                      f"{real_data[col].iloc[index]}, {predicted_data[col].iloc[index]}")
    
    return sum_errors



def optimization(alpha, beta, delta, gamma):
    for _ in range(1, 100_0):
        time_update = time[-1] + step
        lapin_update = (lapin[-1]*(alpha - beta * renard[-1])) * step + lapin[-1]
        renard_update = (renard[-1]*(delta*lapin[-1] - gamma)) * step  + renard[-1]

        lapin.append(lapin_update)
        renard.append(renard_update)
        time.append(time_update)
    df = pd.DataFrame({'lapin': lapin*1000,'renard': renard*1000})
    return df

print(optimization(1,1,1,1))

best_score = float('inf')  # Initialize to a very large value
best_params = None



real_data = pd.read_csv('math\calcul_scientifique-\populations_lapins_renards.csv')



for params in product(*param_grid.values()):
    # Map parameter values to their corresponding keys
    param_dict = dict(zip(param_grid.keys(), params))
    
    # Evaluate the objective function
    predicted = optimization(**param_dict)
    score = mse(real_data, predicted)  # Returns a dictionary {'lapin': ..., 'renard': ...}
    total_score = sum(score.values())  # Sum of errors for lapin and renard
    
    print(f"Score: {score}, Total Score: {total_score}")
    
    # Update best score and parameters
    if total_score < best_score:
        best_score = total_score
        best_params = param_dict

print(f"Best Parameters: {best_params}")
print(f"Best Score: {best_score}")