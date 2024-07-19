import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
url = "https://www.kaggle.com/datasets/yasserh/titanic-dataset"
titanic_df = pd.read_csv(url)

# Data Cleaning
# Handle missing values (e.g., impute missing age values)
titanic_df["Age"].fillna(titanic_df["Age"].median(), inplace=True)

# Remove irrelevant columns (e.g., passenger ID)
titanic_df.drop(columns=["PassengerId"], inplace=True)

# Check for duplicates
duplicates = titanic_df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Exploratory Data Analysis (EDA)
# Explore the distribution of features
sns.histplot(titanic_df["Age"], bins=20, kde=True)
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Distribution of Passenger Ages")
plt.show()

# Visualize survival rates by different variables
sns.catplot(x="Sex", hue="Survived", kind="count", data=titanic_df)
plt.xlabel("Sex")
plt.ylabel("Count")
plt.title("Survival by Gender")
plt.show()

# Identify outliers and anomalies
sns.boxplot(x="Fare", data=titanic_df)
plt.xlabel("Fare")
plt.title("Fare Distribution")
plt.show()

# Relationship Between Variables
# Investigate correlations
correlation_matrix = titanic_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Patterns and Trends
# Analyze survival patterns
sns.catplot(x="Pclass", hue="Survived", kind="count", data=titanic_df)
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.title("Survival by Passenger Class")
plt.show()
