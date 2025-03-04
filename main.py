import numpy as np
import pandas as pd
from pandas import read_csv
dataset=read_csv('Obesity.csv',header=None)
#print(dataset)

#first part of the requirement tasks (Data Analysis task)
#a)Display the first 12 rows of the dataset(Obesity)
print(dataset.head(12))
#a)Display the last 12 rows of the dataset(Obesity)
print(dataset.tail(12))
