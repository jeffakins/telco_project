# TELCO Classification Project 
### - by Jeff Akins
![TELCO Logo](https://upload.wikimedia.org/wikipedia/commons/6/6d/Telco_System_Logo.jpg)

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

### Project Specifications
- Why are our customers churning?