import matplotlib.pyplot as plt

# Example data (replace with your actual data)
data = {'Male': 30, 'Female': 20, 'Non-binary': 5}

# Extract category names and values
categories = list(data.keys())
values = list(data.values())

# Create a bar chart
plt.bar(categories, values)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Distribution of Gender')
plt.show()
