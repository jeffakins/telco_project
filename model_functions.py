# Models Used for the Telco Project

#---------------Imports---------------------------------

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore")

#---------------Functions-------------------------------
# Decision Tree:
def decision_tree(X_train, y_train, X_validate, y_validate, threshold=0.05, max_dep=25):
    threshold = threshold   # Set our threshold for how overfit we'll tolerate

    models = []             # Initiate models list
    metrics = []            # Initiate metrics list

    for i in range(2, max_dep):
        # Make the model
        tree = DecisionTreeClassifier(max_depth=i, random_state=123)

        # Fit the model (on train and only train)
        tree = tree.fit(X_train, y_train)

        # Use the model
        # We'll evaluate the model's performance on train, first
        in_sample_accuracy = tree.score(X_train, y_train)   
        out_of_sample_accuracy = tree.score(X_validate, y_validate)

        # Calculate the difference
        difference = in_sample_accuracy - out_of_sample_accuracy
        
        # Add a conditional to check vs. the threshold
        if difference > threshold:
            break
        
        # Formulate the output for each model's performance on train and validate
        output = {
            "max_depth": i,
            "train_accuracy": in_sample_accuracy,
            "validate_accuracy": out_of_sample_accuracy,
            "difference": difference
        }
        
        # Add the metrics dictionary to the list, so we can make a dataframe
        metrics.append(output)
        
        # Add the specific tree to a list of trained models
        models.append(tree)

    # make a dataframe
    results = pd.DataFrame(metrics)
    print(results)

    # plot the data
    results[['max_depth', 'train_accuracy', 'validate_accuracy']].set_index('max_depth').plot(figsize = (16,9), linewidth=2)
    plt.ylim(0.50, 1)
    plt.xlabel("Max Depth", fontsize = 16)
    plt.ylabel('Accuracy', fontsize = 18)
    plt.xticks(np.arange(1, i+1, 1))
    plt.grid(b=True)

    return results

#-------------------------------------------------------
# Random Forest
