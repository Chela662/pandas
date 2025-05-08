# pandas
# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset

try:
    # Load Iris dataset from sklearn
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print("Dataset loaded successfully.\n")
    print("First 5 rows of the dataset:")
    print(df.head())

    # Check data types and missing values
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing values:")
    print(df.isnull().sum())

except Exception as e:
    print("Error loading the dataset:", e)

# Task 2: Basic Data Analysis

# Basic statistics
print("\nDescriptive statistics:")
print(df.describe())

# Group by species and calculate mean
grouped = df.groupby('species').mean()
print("\nAverage measurements per species:")
print(grouped)

# Observations
print("\nObservation:")
print("Setosa has the shortest petals and versicolor has intermediate values between setosa and virginica.")

# Task 3: Data Visualization

# 1. Line chart - showing average petal length over species (not time, but to meet requirement)
plt.figure(figsize=(8, 5))
plt.plot(grouped.index, grouped['petal length (cm)'], marker='o')
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Bar chart - average sepal width per species
grouped['sepal width (cm)'].plot(kind='bar', color='skyblue')
plt.title('Average Sepal Width by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Width (cm)')
plt.tight_layout()
plt.show()

# 3. Histogram - distribution of petal length
plt.hist(df['petal length (cm)'], bins=10, color='lightgreen', edgecolor='black')
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 4. Scatter plot - sepal length vs. petal length
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal Length vs Petal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.show()
