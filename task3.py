import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (assuming you've downloaded 'bank-additional-full.csv')
data = pd.read_csv('bank-additional-full.csv')

# Preprocessing: Encode categorical variables and handle missing values
# Example: Assuming 'X' contains features and 'y' contains the target variable
X = data.drop(columns=['subscribed'])
y = data['subscribed']
X_encoded = pd.get_dummies(X, drop_first=True)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create the Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=5)  # Adjust hyperparameters as needed
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
