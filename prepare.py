# Script to Clean the Iris Data

#--------Imports---------
import pandas as pd
import acquire as aq

#--------Functions-------
# Function to clean the Iris Data
def prep_telco():
    df = aq.get_telco_data()                            # Using Acquire function to bring in telco data
    df = df.drop_duplicates()                           # Dropping Duplicates
    df.total_charges = df.total_charges.str.strip()     # Removing white space
    df.total_charges = df.total_charges.replace('', 0)  # Replacing total_charges empty cells with 0 due to tenure = 0
    cols_to_drop = ['species_id', 'measurement_id']
    iris_df = iris_df.drop(columns=cols_to_drop)
    dummy_df = pd.get_dummies(iris_df[['species_name']], dummy_na=False)
    iris_df = pd.concat([iris_df, dummy_df], axis=1)
    return iris_df