# Script to Prep the TELCO Data

#--------Imports---------
import pandas as pd
import acquire as aq

#--------Functions-------

def prep_telco():
    '''
    This function will walk through all of the cleaning and data prep 
    process needed to explore and model the Telco data set
    '''
    df = aq.get_telco_data()                            # Using Acquire function to bring in telco data
    
    df = df.drop_duplicates()                           # Dropping Duplicates
    
    df.total_charges = df.total_charges.str.strip()     # Removing white space
    df.total_charges = df.total_charges.replace('', 0)  # Replacing total_charges empty cells with 0 due to tenure = 0
    df.total_charges = df.total_charges.astype('float64') # Convert from obj to 

    to_replace={'Yes': 1, 'No': 0, 
                'No internet service': 0, 
                'No phone service': 0}                  # Encoding (Changing Yes to 1 and No to 0)
    df = df.replace(to_replace)                         # Encoding
    
    columns_to_rename = {'contract_type': 'contract',
                   'internet_service_type': 'internet'} # Renaming columns
    df = df.rename(columns=columns_to_rename)           # Renaming

    dummy_df = pd.get_dummies(df[['gender', 'contract','internet', 'payment_type']])
    df = pd.concat([df, dummy_df], axis=1)              # Creating dummy variables and concating

    columns_to_drop = ['payment_type_id', 'internet_service_type_id', 
                        'contract_type_id', 'gender', 'contract', 
                        'internet', 'payment_type']
    df = df.drop(columns=columns_to_drop)               # Dropping seven columns

    columns_to_rename = {'gender_Female': 'female',
                   'gender_Male': 'male',
                    'contract_Month-to-month': 'monthly_contract',
                    'contract_One year': 'one_yr_contract',
                    'contract_Two year': 'two_yr_contract',
                    'internet_DSL': 'dsl',
                    'internet_Fiber optic': 'fiber',
                    'internet_None': 'no_internet',
                    'payment_type_Bank transfer (automatic)': 'bank_transfer',
                    'payment_type_Credit card (automatic)': 'credit_card',
                    'payment_type_Electronic check': 'electronic_check',
                    'payment_type_Mailed check': 'mailed_check'}
    df = df.rename(columns=columns_to_rename)           # More columns to rename
    return df