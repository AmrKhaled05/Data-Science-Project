import numpy as np
import pandas as pd
import seaborn as sns
from pandas import read_csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
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


print("========================================================================================")
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
#==== (i)  Identify and print the row corresponding to the maximum value of a selected numerical feature.
max_weight_row = dataset[dataset['Weight'] == dataset['Weight'].max()]
print("Max weight row\n")
print(max_weight_row)
print("\n===============\n")
#====

#==== (j) Construct a boxplot for an attribute you consider significant and justify the selection
#FCVC Frequency of vegetable consumption (scale from 1 to 3).
df = pd.DataFrame(dataset)
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['FCVC'])
plt.title("Boxplot of Frequency of Vegetable Consumption (FCVC)", fontsize=16)
plt.xlabel("FCVC (1=Low, 2=Medium, 3=High)", fontsize=12)
plt.show()
#====


#==== (k) Generate a histogram for a chosen attribute and provide an explanation for its relevance.
df = pd.DataFrame(dataset)
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.histplot(df['NCP'], bins=4, kde=False, color='skyblue', edgecolor='black')
plt.title("Histogram of Number of Main Meals Per Day (NCP)", fontsize=16)
plt.xlabel("Number of Meals", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(range(int(min(df['NCP'])), int(max(df['NCP'])) + 1, 1))  # Set ticks in steps of 1
plt.show()

# Explanation for why NCP was chosen for the histogram:
"""
We chose the 'NCP' (Number of Main Meals Per Day) attribute for the histogram because it provides 
insight into dietary habits. Understanding how many main meals people consume in a day can help with 
various analyses related to health, nutrition, and lifestyle.

Histograms are particularly useful for visualizing the distribution of discrete data, and NCP fits this 
category as it is a count variable (integer values).

This histogram allows us to:
1. Identify the most common number of main meals people have per day.
2. Check the distribution of meals and whether people tend to have more or fewer meals on average.
3. Help us understand if there is any skewness in the data toward people with fewer or more meals.
4. Inform decisions regarding dietary recommendations, public health initiatives, or understanding 
   lifestyle habits in different populations.
"""

#===

#=== (l) Create a scatterplot using two attributes and interpret the relationship observed.
df = pd.DataFrame(dataset)

# Set the style for the plot
sns.set(style="whitegrid")

# Create the scatterplot for AGE vs. Weight
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Age', y='Weight', color='purple')

# Add a title and labels
plt.title("Scatterplot of AGE vs. Weight", fontsize=16)
plt.xlabel("Age", fontsize=12)
plt.ylabel("Weight", fontsize=12)

# Show the plot
plt.show()

# Interpretation of the scatterplot:
"""
The scatterplot visualizes the relationship between AGE and Weight.

Each point on the plot represents an individual. The x-axis represents their AGE, 
and the y-axis represents their Weight.

By observing the scatterplot, we can interpret:
- If there is a positive or negative correlation between AGE and Weight.
- Whether the points are scattered randomly or if there's a trend, such as older individuals having higher or lower weights.
- Whether there are any outliers in the data, such as individuals who are significantly heavier or lighter than expected for their age.

For example, if the points tend to rise from left to right, it might indicate a positive correlation between AGE and Weight, where older individuals tend to weigh more.
Conversely, if the points tend to decrease from left to right, it might suggest a negative correlation.
"""

#===

#=== (m)
# Create DataFrame
df = pd.DataFrame(dataset)

# Initialize the StandardScaler
scaler = StandardScaler()

# Select numerical columns that you want to standardize
numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP']

# Apply StandardScaler to the selected columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Display the standardized data
print(df)
print("pca operation")
#n) Perform PCA (Principal Component Analysis) to reduce dimensionality to two components, andvisualize the dataset before and after applying PCA.
pca = PCA(n_components=2)
X = dataset.iloc[:, :-1]
X = pd.get_dummies(X)
X = X.dropna()
X = X.values
X = X.astype(float)
X = (X - X.mean()) / X.std()
pca_result = pca.fit_transform(X)
pca_result = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
print("PCA Result:")
print(pca_result)
# Visualize the dataset before PCA
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.title('Dataset before PCA')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Visualize the dataset after PCA
plt.figure(figsize=(8, 6))
plt.scatter(pca_result["PC1"], pca_result["PC2"], alpha=0.5)
plt.title('Dataset after PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
print("\n===============\n")

#o) Analyze the correlation between numerical features using a heatmap.
# Select only numeric columns
numeric_dataset = dataset.select_dtypes(include=[float, int])

# Compute the correlation matrix
correlation_matrix = numeric_dataset.corr()
plt.figure(figsize=(10, 8))
plt.title("Correlation Heatmap")
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.show()
print("\n===============\n")

#b) Use Python to find the class distribution of a selected categorical feature and analyze the results.
class_distribution = dataset["NObeyesdad"].value_counts()
print("Class Distribution of 'NObeyesdad':")
print(class_distribution)
print("\n===============\n")
#c)Apply Python techniques to create new features from existing ones (feature engineering) and explain the significance of the new features.
# Create a new feature 'BMI' by calculating Body Mass Index using
# the formula: BMI = Weight (kg) / (Height (m) ^ 2)
dataset["Height"] = dataset["Height"] / 100
dataset["BMI"] = dataset["Weight"] / (dataset["Height"] ** 2)
print("New dataset with BMI feature:")
print(dataset)
print("\n===============\n")

