# Telco Churn Data Acquisition 

#---------------------------Imports---------------------------------------------------

import env
import pandas as pd
import numpy as np
import os

#---------------------------Connection Info--------------------------------------------

# Connection information for the mySQL Server

def get_connection(db, user=env.username, host=env.hostname, password=env.password):
    connection_info = f'mysql+pymysql://{user}:{password}@{host}/{db}'
    return connection_info

#---------------------------Data Base Functions----------------------------------------

# Function to retrieve the TELCO Data Set from CODEUP's mySQL Server 
def get_telco_data():
    if os.path.isfile('telco_churn.csv'):
        df = pd.read_csv('telco_churn.csv', index_col=0)  # If csv file exists read in data from csv file.
    else:
        sql = '''SELECT * 
                FROM customers
                JOIN contract_types USING(contract_type_id)
                JOIN internet_service_types USING(internet_service_type_id)
                JOIN payment_types USING(payment_type_id);''' 
                                                    # SQL query
        db = 'telco_churn'                          # Database name
        df = pd.read_sql(sql, get_connection(db))   # Pandas DataFrame
        df.to_csv('telco_churn.csv')                # Cache Data
    return df



