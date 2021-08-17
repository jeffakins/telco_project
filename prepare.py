# Script to Clean the Iris Data

#--------Imports---------
import pandas as pd
import acquire as aq

#--------Functions-------
# Function to clean the Iris Data
def prep_iris():
    iris_df = aq.get_iris_data()
    iris_df = iris_df.drop_duplicates()
    cols_to_drop = ['species_id', 'measurement_id']
    iris_df = iris_df.drop(columns=cols_to_drop)
    dummy_df = pd.get_dummies(iris_df[['species_name']], dummy_na=False)
    iris_df = pd.concat([iris_df, dummy_df], axis=1)
    return iris_df