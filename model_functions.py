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

    models = []             # Initiate models list for outputs
    metrics = []            # Initiate metrics list for outputs

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
    # print(results)

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
def rand_forest(X_train, y_train, X_validate, y_validate, threshold=0.05, max_dep=7):
    models = []                 # For output
    metrics = []                # For output
    for i in range(2, max_dep): # Max Depth
        for n in range(2, max_dep): # Min sample leaf
            # Make the model
            rf = RandomForestClassifier(bootstrap=True, 
                                    class_weight=None, 
                                    criterion='gini',
                                    min_samples_leaf=n,
                                    n_estimators=100,
                                    max_depth=i, 
                                    random_state=123)

            # Fit the model (on train and only train)
            rf = rf.fit(X_train, y_train)

            # We'll evaluate the model's performance on train and validate
            in_sample_accuracy = rf.score(X_train, y_train)   
            out_of_sample_accuracy = rf.score(X_validate, y_validate)

            # Calculate the difference
            difference = in_sample_accuracy - out_of_sample_accuracy

            # Add a conditional to check vs. the threshold
            if difference > threshold:
                break

            # Formulate the output for each model's performance on train and validate
            output = {
                "max_depth": i,
                "min_samples_leaf": n,
                "train_accuracy": in_sample_accuracy,
                "validate_accuracy": out_of_sample_accuracy,
                "difference": difference
            }

            # Add the metrics dictionary to the list, so we can make a dataframe
            metrics.append(output)

            # Add the specific tree to a list of trained models
            models.append(rf)

    df = pd.DataFrame(metrics)
    df

    # make a dataframe
    results = pd.DataFrame(metrics)
    # print(results)

    results[['max_depth', 'train_accuracy', 'validate_accuracy']].set_index('max_depth').plot(figsize = (16,9), linewidth=2)
    plt.ylim(0.50, 1)
    plt.xlabel("Max Depth", fontsize = 16)
    plt.ylabel('Accuracy', fontsize = 18)
    plt.xticks(np.arange(1, i+1, 1))
    plt.grid(b=True)

    return results

#-------------------------------------------------------
# KNN
def knn(X_train, y_train, X_validate, y_validate, max_k = 26):
    metrics = []        # For output

    # loop through different values of k
    for k in range(1, max_k):
                
        # define the thing
        knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
        
        # fit the thing (remmeber only fit on training data)
        knn.fit(X_train, y_train)
        
        # use the thing (calculate accuracy)
        train_accuracy = knn.score(X_train, y_train)
        validate_accuracy = knn.score(X_validate, y_validate)
        difference = train_accuracy - validate_accuracy
        
        output = {
            "k": k,
            "train_accuracy": train_accuracy,
            "validate_accuracy": validate_accuracy,
            "difference": difference
        }
        
        metrics.append(output)

    # make a dataframe
    results = pd.DataFrame(metrics)
    # print(results)

    # plot the data
    results[['k', 'train_accuracy', 'validate_accuracy']].set_index('k').plot(figsize = (16,9), linewidth=2)
    plt.ylim(0.50, 1)
    plt.xlabel("k", fontsize = 16)
    plt.ylabel('Accuracy', fontsize = 18)
    plt.xticks(np.arange(1, k+1, 1))
    plt.grid(b=True)

    return results

#-------------------------------------------------------
# Logistic Regression
def log_regression(X_train, y_train, X_validate, y_validate):
    # Train Data
    logit = LogisticRegression(C=1, random_state=123)   # Create the model
    logit.fit(X_train, y_train)                         # Fit the model with Train Data
    print('Coefficient: \n', logit.coef_)               # Print coeffecients
    print('Intercept: \n', logit.intercept_)            # Print the intercept
    
    y_pred = logit.predict(X_train)                     # y prediction
    y_pred_proba = logit.predict_proba(X_train)         # y prob
    print("Train Confusion Matrix:")                          
    print(confusion_matrix(y_train, y_pred))            # Confusion Matrix
    print("")
    print("Train Data:")
    train_class_report = pd.DataFrame(classification_report(y_train, y_pred, output_dict=True))
    print(train_class_report)                           # Print accuracy report on Train Data

    # Validate Data
    y_pred = logit.predict(X_validate)                  # y prediction
    y_pred_proba = logit.predict_proba(X_validate)      # y prob
    print("Validate Confusion Matrix:") 
    print(confusion_matrix(y_validate, y_pred))         # Confusion Matrix
    print("")
    print("Validate Data:")
    val_class_report = pd.DataFrame(classification_report(y_validate, y_pred, output_dict=True))
    print(val_class_report)                             # Print accuracy report on Validate Data

    return train_class_report, val_class_report