import pandas as pd
import matplotlib.pyplot as plt

# Load your social media data (replace with your actual data)
data = pd.read_csv('social_media_data.csv')

# Assuming you have a 'sentiment_score' column (range: -1 to 1)
plt.figure(figsize=(8, 6))
plt.hist(data['sentiment_score'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title('Sentiment Distribution')
plt.show()
