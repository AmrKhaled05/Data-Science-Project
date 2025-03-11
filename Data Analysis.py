import numpy as np
import pandas as pd
from pandas import read_csv
dataset=pd.read_csv('Obesity.csv')
dataset = dataset.convert_dtypes()
#print(dataset)
#first part of the requirement tasks (Data Analysis task)
#a)Display the first 12 rows of the dataset(Obesity)
print(dataset.head(12))
print("========================================================================================")
#a)Display the last 12 rows of the dataset(Obesity)
print(dataset.tail(12))
print("========================================================================================")
#b)Identify and print the total number of rows and columns present.
print(dataset.shape)
print("========================================================================================")
#c)List all column names along with their corresponding data types
print(dataset.dtypes)
print("========================================================================================")
#d)Print the name of the first column.
print(dataset.columns[0])
print("========================================================================================")
#e) Generate a summary of the dataset, including non-null counts and data types.
print(dataset.info())
print("========================================================================================")
#f) Choose a categorical attribute and display the distinct values it contains.
print(dataset['NObeyesdad'].unique())
print("========================================================================================")
#g) Identify the most frequently occurring value in the chosen categorical attribute.
print(dataset['NObeyesdad'].mode())
print("========================================================================================")
#h) Calculate and present the mean, median, standard deviation, and percentiles (20)
resulth = dataset.describe(percentiles=[0.2])
meanvalues = resulth.loc["mean"]
stdvalues = resulth.loc["std"]
medianvalues = resulth.loc["50%"]
percentile20th = resulth.loc["20%"]
print("Mean:\n", meanvalues)
print("Standard Deviation:\n", stdvalues)
print("Median:\n", medianvalues)
print("20th Percentile:\n", percentile20th)
