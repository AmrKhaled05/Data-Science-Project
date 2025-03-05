import numpy as np
import pandas as pd
from pandas import read_csv

dataset=pd.read_csv('Obesity.csv')

dataset = dataset.convert_dtypes()

#==== (a) Apply a filter to select rows based on a specific condition of your choice (e.g., select records where a value exceeds a certain threshold)
filtered_data = dataset[dataset["Age"] > 27]
print(filtered_data)
print("\n===============\n")
#====

#==== (b) Identify records where a chosen attribute starts with a specific letter and count how many records match this condition
filtered_data = dataset[dataset["Gender"].str.startswith("F")]
count = len(filtered_data)
print(filtered_data)
print(f"Number of records where Gender starts with 'F': {count}")
print("\n===============\n")
#====

#==== (c) Determine the total number of duplicate rows and remove them if found
duplicate_count = dataset.duplicated().sum()
print(f"Total number of duplicate rows: {duplicate_count}")
dataset_cleaned = dataset.drop_duplicates()
#print(dataset_cleaned)
print("\n===============\n")
#====

#==== (d) Convert the data type of a numerical column from integer to string
dataset["Age"] = dataset["Age"].astype("string")
print(f"Age was Integer now It is : {dataset["Age"].dtype}\n")
print(dataset.head())
print("\n===============\n")
#====

#==== (e) Group the dataset based on two selected categorical features and analyze the results
grouped_data = dataset.groupby(["Gender", "NObeyesdad"]).size().reset_index(name="Count")
print(grouped_data)
print("\n===============\n")
#====

#==== (f) Check for the existence of missing values within the dataset
missing_values = dataset.isnull().sum()
print("Missing values in each column:")
print(missing_values)
total_missing = missing_values.sum()
print(f"\nTotal number of missing values in the dataset: {total_missing}")
print("\n===============\n")
#====

#==== (g) If any missing values are found, replace them with the median or mode as appropriate
missing_values = dataset.isnull().sum()
total_missing = missing_values.sum()
print("Missing values in each column before handling:")
print(missing_values)
print(f"\nTotal missing values: {total_missing}\n")
if total_missing > 0:
    for column in dataset.columns:
        if dataset[column].dtype in ["int64", "float64"]:
            dataset[column].fillna(dataset[column].median(), inplace=True)
        else:
            dataset[column].fillna(dataset[column].mode()[0], inplace=True)
    missing_values_after = dataset.isnull().sum()
    print("\nMissing values in each column after handling:")
    print(missing_values_after)
    print("\nAll missing values have been handled successfully!")
else:
    print("\nNo missing values found in the dataset.")
print("\n===============\n")
#====

#==== (h) Divide a chosen numerical column into 5 equal-width bins and count the number of records in each bin
dataset["Age_Binned"] = pd.cut(dataset["Age"].astype(float), bins=5)
bin_counts = dataset["Age_Binned"].value_counts().sort_index()
print(f"Number of records in each Age bin:\n")
print(bin_counts)
print("\n===============\n")
#====