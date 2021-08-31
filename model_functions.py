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
    '''
    This function uses the sklearn DecisionTreeClassifier to create a Decision Tree
    '''
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
    plt.title('Decision Tree', fontsize = 20)
    plt.xlabel("Max Depth", fontsize = 16)
    plt.ylabel('Accuracy', fontsize = 18)
    plt.xticks(np.arange(1, i+1, 1))
    plt.grid(b=True)

    return results

#-------------------------------------------------------
# Random Forest
def rand_forest(X_train, y_train, X_validate, y_validate, threshold=0.05, max_dep=7):
    '''
    This function uses the sklearn RandomForestClassifier 
    to create a random forrest model
    '''
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
    plt.title('Random Forest', fontsize = 20)
    plt.xlabel("Max Depth", fontsize = 16)
    plt.ylabel('Accuracy', fontsize = 18)
    plt.xticks(np.arange(1, i+1, 1))
    plt.grid(b=True)

    return results

#-------------------------------------------------------
# KNN
def knn(X_train, y_train, X_validate, y_validate, max_k = 26):
    '''
    This function uses the sklearn KNeighborsClassifier 
    to create a k neraest neighbors model
    '''
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
    plt.title('KNN', fontsize = 20)
    plt.xlabel("k", fontsize = 16)
    plt.ylabel('Accuracy', fontsize = 18)
    plt.xticks(np.arange(1, k+1, 1))
    plt.grid(b=True)

    return results

#-------------------------------------------------------
# Logistic Regression
def log_regression(X_train, y_train):
    '''
    This function uses the sklearn LogisticRegression 
    to create a logistic regression model for the train data
    '''
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
    #print(train_class_report)                           # Print accuracy report on Train Data

    return train_class_report

def log_regression_val(X_train, y_train, X_validate, y_validate):
    '''
    This function uses the sklearn LogisticRegression 
    to create a logistic regression model for the train data
    '''
    # Validate Data
    logit = LogisticRegression(C=1, random_state=123)   # Create the model
    logit.fit(X_train, y_train)                         # Fit the model with Train Data
    print('Coefficient: \n', logit.coef_)               # Print coeffecients
    print('Intercept: \n', logit.intercept_)            # Print the intercept

    y_pred = logit.predict(X_validate)                  # y prediction
    y_pred_proba = logit.predict_proba(X_validate)      # y prob
    print("Validate Confusion Matrix:") 
    print(confusion_matrix(y_validate, y_pred))         # Confusion Matrix
    print("")
    print("Validate Data:")
    val_class_report = pd.DataFrame(classification_report(y_validate, y_pred, output_dict=True))
    #print(val_class_report)                             # Print accuracy report on Validate Data

    return val_class_report


# Model Comparison Report

def model_report_all_data():
    '''
    This is a function to output the best models based on train accuracy and minimizing oversampling
    Utilizing All Data features from Telco
    '''
    report1 = {
        'Model': ['DT', 'RF', 'KNN', 'LR'],
        'Parameters' : ['Max Depth = 3', 'Max Depth = 6 & Min Sample Leaf = 2', 'KNN = 19', 'Default'],
        'Train' : [0.792, 0.818, 0.796, 0.805],
        'Validate' : [0.794, 0.808, 0.789, 0.792]
    } 
    report = pd.DataFrame(report1)
    report['Difference'] = report.Train - report.Validate
    return report

def model_report_select_data():
    '''
    This is a function to output the best models based on train accuracy and minimizing oversampling
    Utilizing select features from Telco
    '''
    report1 = {
        'Model': ['DT', 'RF', 'KNN', 'LR'],
        'Parameters' : ['Max Depth = 5', 'Max Depth = 6 & Min Sample Leaf = 4', 'KNN = 14', 'Default'],
        'Train' : [0.798, 0.812, 0.813, 0.807],
        'Validate' : [0.794, 0.802, 0.800, 0.801]
    } 
    report = pd.DataFrame(report1)
    report['Difference'] = report.Train - report.Validate
    return report


# Best Model to run on test data

def best_rf(X_train, y_train, y, X):
    '''This function outputs a classification report for the best TELCO model'''                                                
    # Create the model
    rf = RandomForestClassifier(bootstrap=True, 
                        class_weight=None, 
                        criterion='gini',
                        min_samples_leaf=2,
                        n_estimators=100,
                        max_depth=6, 
                        random_state=123)

    # Fit the model (on train and only train)
    rf = rf.fit(X_train, y_train)
    y_pred = rf.predict(X)
    
    # Create the report
    report = pd.DataFrame(classification_report(y, y_pred, output_dict=True))
    return report


# Prediction on Test
def best_model_churn_prediction(X_train, y_train, y, X):
    '''a CSV file with customer_id, probability of churn, and prediction of churn. 
    (1=churn, 0=not_churn). These predictions should be from your best performing 
    model ran on X_test. Note that the order of the y_pred and y_proba are numpy 
    arrays coming from running the model on X_test. The order of those values will 
    match the order of the rows in X_test, so you can obtain the customer_id from 
    X_test and concatenate these values together into a dataframe to write to CSV.'''
    rf = RandomForestClassifier(bootstrap=True, 
                    class_weight=None, 
                    criterion='gini',
                    min_samples_leaf=2,
                    n_estimators=100,
                    max_depth=6, 
                    random_state=123)
    # Fit the model (on train and only train)

    rf = rf.fit(X_train, y_train)
    y_pred = rf.predict(X)  
    return y_pred


    