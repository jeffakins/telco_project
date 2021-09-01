![TELCO Logo](https://upload.wikimedia.org/wikipedia/commons/6/6d/Telco_System_Logo.jpg)

# TELCO Classification Project 
### - by Jeff Akins

## Project Summary

### Project Objectives
- Document code, process (data acquistion, preparation, exploratory data analysis and statistical testing, modeling, and model evaluation), findings, and key takeaways in a Jupyter Notebook report.
- Create modules (acquire.py, prepare.py) that make your process repeateable.
- Construct a model to predict customer churn using classification techniques.
- Deliver a 5 minute presentation consisting of a high-level notebook walkthrough using your Jupyter Notebook from above; your presentation should be appropriate for your target audience.
- Answer panel questions about your code, process, findings and key takeaways, and model.

### Business Goals
- Find drivers for customer churn at Telco. Why are customers churning?
- Construct a ML classification model that accurately predicts customer churn.
- Document your process well enough to be presented or read like a report.

### Audience
- Your target audience for your notebook walkthrough is the Codeup Data Science team. 

### Deliverables
- Jupyter Notebook Report 
- README.md file containing the project information
- CSV file with customer_id, probability of churn, and prediction of churn
- Acquire and prepare files
- Notebook walkthrough presentation 

## Executive Summary
#### Why are our customers churning?
- They are on a monthly contract
- They have fiber internet	
- They pay with electronic check	
- Higher monthly charges	
- Utilize paperless billing	
- They are a senior citizen	
- They stream tv	
- They stream movies

#### Why our customers are staying?
- They use our online backup service	
- They pay with a mailed check	
- They pay using bank transfer
- They pay with credit card	
- They have dsl internet	
- They have a partner and/or dependents	
- They use our tech support	
- They use online security	
- They have a one or two year contract	
- They do not use our internet services		
 	
### Can we predict churn?
- #### Based on these factors we can predict with 80% accuracy whether or not a customer will churn


### Data dictionary
Target |   Description |    Data Type
--|--|--
churn |   indicates whether or not a customer churned |   int64

Categorical Features |  Description |    Data Type
--|--|--
senior_citizen |    indicates if the customer is a senior citizen |  int64
partner |    indicates if the customer has a partner | int64
dependents |        indicates if the customer has dependents |   int64
phone_service |    indicates if the customer has phone service with Telco    |  int64
multiple_lines |    indicates if the customer with phone service has multiple lines    |    int64
online_security |    indicates if the customer has online security services |   int 64
online_backup |    indicates if the customer has online backup services |   int64
device_protection |     indicates if the customer has device protection services | int64
tech_support |  indicates if the customer has tech support services |    int64
streaming_tv |    indicates if the customer has tv streaming services |    int64
streaming_movies |    indicates if the customer has movie streaming services |    int64
payment_type | indicates the type of payment method a customer is using | int64
internet_service_type |    indicates which internet service (if any) the customer has |    int64
gender |   indicates the the customers' gender identity |    uint8
contract_type |     indicates the type of contract the customer has with Telco |   int64

Continuous Features | Description | Data Type
--|--|--
monthly_charges | how much a customer pays per month in dollars|    float64
total_charges   | how much a customer has paid over the course of their tenure |    float64
tenure          | how many months the customer has been with the company|   int64

Other   | Description   | Data Type
--|--|--
customer_id |   customer id number  | object

## Project Specifications

### Plan:
- Initially model all data
- Refine features based on explore results

### Acquire:
- Acquire.py function brings in TELCO data from Codeup's MySQL server
- 7043 Rows (customers)
- 24 Columns (features)

### Prepare:
- Used the prep_telco.py function to clean the Telco Data, whcih included:
- Dropping Duplicates
- Removing white space
- Replacing 'total_charges' empty cells with 0 due to tenure = 0
- Converting 'total_charges' from obj to float
- Encoding (Changing Yes to 1 and No to 0)
- Creating dummy variables for 'gender', 'contract', 'internet', 'payment_type' 
- Concatenating the dummy variables
- Dropping 7 redundant columns
- Renaming 12 columns to shorten the names
- Results:
  - 7043 Rows
  - 29 Columns

### Explore:
#### Univariate
##### Findings: 
- Gained a better understanding of the ratio of the customers per category
#### Bivariate
##### Statistical Findings: Answering the question, is there a relationship between churn and all other customer features?
- Significance Level, alpha = 0.05
- Target = churn
###### 24 Chi^2 Tests - Churn vs Categorical Variables
- H<sub>0</sub>: churn is independent of each categorical variable
- H<sub>a</sub>: churn has a dependent relationship with each categorical variable
- Rejected H<sub>0</sub> in all cases due to p<0.05, except: 'male', 'female', 'phone_service'
###### 2 Mann-Whitney Test - Churn vs Continuous Variables
- H<sub>0</sub>: there is no relationship between tenure or monthly charges and churn 
- H<sub>a</sub>: churn has a linear relationship with each continuous variable
- Rejected H<sub>0</sub> due to p<0.05 in both cases
#### Multivariate
- Grouped by category
- Showed clusters of churn within all features
#### Correlation Heatmap
##### Correlations Findings:
- total_charges and tenure 
- monthly charges and fiber
- monthly charges and no internet
- male and female
- dls and fiber

### Model & Evaluate
- Decision Tree
- Random Forest
- KNN
- Logistical Regression
##### Best model was a Random Forrest using all features

## Conclusion 
### We can predict with roughly 80% accuracy whether a customer will churn. 
#### We can also conclude that customers with these TELCO features (services) increase their probability of churn:
- Customers on a monthly contract 
- Customers using fiber internet	
- Customers that pay with an electronic check	
- Higher monthly charges increase the probability of churn	
- Customers that use paperless billing	
- Customers that are a senior citizen	
- Customers that stream tv	
- Customers that stream movies
#### These features help to reduce churn:
- Customers that use TELCO's online backup service	
- Customers that pay with a mailed check	
- Customers that pay using bank transfer
- Customers that pay with credit card	
- Customers using dsl internet	
- Customers that have a partner and/or dependents	
- Customers that use our tech support	
- Customers that use online security	
- Customers that have a one or two year contract	
- Customers that do not use our internet services	
- Higher tenure as a TELCO customer decreases the probability of churn