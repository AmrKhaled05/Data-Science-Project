import numpy as np
import pandas as pd
from pandas import read_csv
dataset=pd.read_csv('Obesity.csv')
dataset = dataset.convert_dtypes()
#run it first before writing any code to ensure the dataset is printed out
#print(dataset)