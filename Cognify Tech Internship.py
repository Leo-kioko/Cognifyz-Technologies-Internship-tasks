#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# In[4]:


df = pd.read_csv(r"C:\Users\Noel\Downloads\Indian Internship\Dataset .csv")
df


# In[7]:


pd.set_option('display.max_rows',None)


# In[9]:


df = pd.read_csv(r"C:\Users\Noel\Downloads\Indian Internship\Dataset .csv")
df


# In[10]:


# Assume the column with cuisines is named 'cuisine'
# Count the occurrences of each cuisine
cuisine_counts = df['Cuisines'].value_counts()

# Get the top 3 most common cuisines
top_three_cuisines = cuisine_counts.head(3)

print("Top 3 most common cuisines:")
print(top_three_cuisines)


# In[11]:


# Step 2: Calculate the count of each cuisine
cuisine_counts = df['Cuisines'].value_counts()

# Step 3: Identify the top three cuisines
top_cuisines = cuisine_counts.head(3)

# Step 4: Calculate the percentage of restaurants serving each of the top cuisines
total_restaurants = len(df)
top_cuisines_percentage = (top_cuisines / total_restaurants) * 100

# Print the results
print("Top 3 Cuisines:")
print(top_cuisines)

print("\nPercentage of Restaurants Serving Each Top Cuisine:")
print(top_cuisines_percentage)


# In[12]:


# Step 2: Identify the city with the highest number of restaurants
city_counts = df['City'].value_counts()
city_with_most_restaurants = city_counts.idxmax()

# Step 3: Calculate the average rating for restaurants in each city
average_ratings_per_city = df.groupby('City')['Aggregate rating'].mean()

# Step 4: Determine the city with the highest average rating
city_with_highest_average_rating = average_ratings_per_city.idxmax()
highest_average_rating = average_ratings_per_city.max()

# Print the results
print(f"City with the highest number of restaurants: {city_with_most_restaurants} ({city_counts.max()} Restaurant Name)")

print("\nAverage rating for restaurants in each city:")
print(average_ratings_per_city)

print(f"\nCity with the highest average rating: {city_with_highest_average_rating} ({highest_average_rating:.2f} Aggregate rating)")


# In[13]:


# Step 1: Calculate the count of restaurants in each price range category
price_range_counts = df['Price range'].value_counts()

# Calculate the percentage of restaurants in each price range category
total_restaurants = len(df)
price_range_percentage = (price_range_counts / total_restaurants) * 100

# Print the counts and percentages
print("Counts of Restaurants in Each Price Range Category:")
print(price_range_counts)

print("\nPercentage of Restaurants in Each Price Range Category:")
print(price_range_percentage)

# Step 3: Visualize the distribution using a bar chart
plt.figure(figsize=(10, 6))
price_range_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Price Ranges Among Restaurants')
plt.xlabel('Price Range')
plt.ylabel('Number of Restaurants')
plt.xticks(rotation=45)
plt.show()


# In[16]:


# Step 1: Calculate the percentage of restaurants that offer online delivery
online_delivery_counts = df['Has Online delivery'].value_counts(normalize="Yes") * 100
percentage_online_delivery = online_delivery_counts.get("Yes", 0)

# Step 2: Compare the average ratings of restaurants with and without online delivery
average_rating_with_delivery = df[df['Has Online delivery'] == "Yes"]['Aggregate rating'].mean()
average_rating_without_delivery = df[df['Has Online delivery'] == "No"]['Aggregate rating'].mean()

# Print the results
print(f"Percentage of restaurants that offer online delivery: {percentage_online_delivery:.2f}%")
print(f"Average rating of restaurants with online delivery: {average_rating_with_delivery:.2f}")
print(f"Average rating of restaurants without online delivery: {average_rating_without_delivery:.2f}")


# In[18]:


# Step 1: Analyze the distribution of aggregate ratings
plt.figure(figsize=(10, 6))
df['Aggregate rating'].plot(kind='hist', bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Aggregate Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Step 2: Determine the most common rating range
rating_counts = pd.cut(df['Aggregate rating'], bins=[0, 1, 2, 3, 4, 5], include_lowest=True).value_counts().sort_index()
most_common_rating_range = rating_counts.idxmax()

# Step 3: Calculate the average number of votes received by restaurants
average_votes = df['Votes'].mean()

# Print the results
print(f"Most common rating range: {most_common_rating_range}")
print(f"Average number of votes received by restaurants: {average_votes:.2f}")


# In[23]:


# Step 1: Identify the most common combinations of cuisines
# Split the 'Cuisines' column into individual cuisines, sort them, and rejoin to handle combinations consistently
# Also, handle cases where the 'Cuisines' column might not be a string
def process_cuisine_combinations(cuisine):
    if isinstance(cuisine, str):
        return ','.join(sorted([c.strip() for c in cuisine.split(',')]))
    return cuisine

df['cuisine_combination'] = df['Cuisines'].apply(process_cuisine_combinations)

# Count the occurrences of each cuisine combination
cuisine_combination_counts = df['cuisine_combination'].value_counts()

# Print the most common cuisine combinations
print("Most common cuisine combinations:")
print(cuisine_combination_counts.head(10))

# Step 3: Determine if certain cuisine combinations tend to have higher ratings
# Calculate the average rating for each cuisine combination
average_ratings_by_combination = df.groupby('cuisine_combination')['Aggregate rating'].mean()

# Print the cuisine combinations with the highest average ratings
print("\nCuisine combinations with the highest average ratings:")
print(average_ratings_by_combination.sort_values(ascending=False).head(10))


# In[30]:


pip install matplotlib cartopy


# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
from scipy import stats


# In[33]:


df = pd.read_csv(r"C:\Users\Noel\Downloads\Indian Internship\Dataset .csv")

# Step 2: Plot the locations on a map
plt.figure(figsize=(12, 8))

# Define the map projection and create a subplot
ax = plt.axes(projection=ccrs.Mercator())

# Add features to the map
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAKES, edgecolor='black')
ax.add_feature(cfeature.RIVERS)

# Plot restaurant locations
ax.scatter(df['Longitude'], df['Latitude'], color='red', s=10, transform=ccrs.Geodetic(), label='Restaurant Name')

# Set plot title and labels
plt.title('Geographic Distribution of Restaurants')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Add a legend
plt.legend()

# Show the plot
plt.show()


# In[34]:


# Step 1: Identify restaurant chains
# For simplicity, assume a restaurant chain is identified by having the same name appearing multiple times
# Count occurrences of each restaurant name
chain_counts = df['Restaurant Name'].value_counts()

# Filter for names that appear more than once
chains = chain_counts[chain_counts > 1].index

# Step 2: Analyze ratings and popularity of restaurant chains
# Filter the dataset to include only the identified chains
chain_df = df[df['Restaurant Name'].isin(chains)]

# Calculate average ratings and total votes for each chain
chain_summary = chain_df.groupby('Restaurant Name').agg({
    'Aggregate rating': 'mean',
    'Votes': 'sum'
}).reset_index()

# Sort by average rating and total votes
sorted_by_rating = chain_summary.sort_values(by='Aggregate rating', ascending=False)
sorted_by_votes = chain_summary.sort_values(by='Votes', ascending=False)

# Print the results
print("Restaurant Chains Sorted by Average Rating:")
print(sorted_by_rating)

print("\nRestaurant Chains Sorted by Total Votes:")
print(sorted_by_votes)


# In[1]:


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


# Step 1: Read the dataset
df = pd.read_csv(r"C:\Users\Noel\Downloads\Indian Internship\Dataset .csv")

# Step 2: Clean and preprocess the reviews
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove non-alphabetic characters and convert to lower case
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
    text = text.lower()
    text = text.strip()
    
    # Tokenize the text
    tokens = text.split()
    
    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    return filtered_tokens

df['processed_reviews'] = df['Rating text'].apply(preprocess_text)

# Step 3: Identify common positive and negative keywords
# Define positive and negative keywords (sample lists)
positive_keywords = ['good', 'very good', 'excellent', 'average']
negative_keywords = ['poor', 'not rated']

# Count occurrences of positive and negative keywords
positive_counts = Counter()
negative_counts = Counter()

for review in df['processed_reviews']:
    positive_counts.update(token for token in review if token in positive_keywords)
    negative_counts.update(token for token in review if token in negative_keywords)

print("Most common positive keywords:")
print(positive_counts.most_common())

print("\nMost common negative keywords:")
print(negative_counts.most_common())

# Step 4: Calculate the average length of reviews
df['review_length'] = df['Rating text'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
average_review_length = df['review_length'].mean()
print(f"\nAverage review length: {average_review_length:.2f} words")

# Step 5: Explore the relationship between review length and rating
# Map 'Rating color' to numeric values if it's categorical
rating_mapping = {
    'Dark Green': 5,
    'Green': 4,
    'Yellow': 3,
    'Orange': 2,
    'Red': 1,
    'White': 0
}

df['numeric_rating'] = df['Rating color'].map(rating_mapping)

# Drop rows with missing values in 'numeric_rating' or 'review_length'
df.dropna(subset=['numeric_rating', 'review_length'], inplace=True)

# Calculate average review length for each rating
average_review_length_per_rating = df.groupby('numeric_rating')['review_length'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='numeric_rating', y='review_length', data=average_review_length_per_rating)
plt.title('Average Review Length vs. Rating')
plt.xlabel('Rating')
plt.ylabel('Average Review Length (words)')
plt.show()

# Calculate correlation between review length and numeric rating
correlation = df['numeric_rating'].corr(df['review_length'])
print(f"Correlation between review length and rating: {correlation:.2f}")

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Read the dataset
df = pd.read_csv(r"C:\Users\Noel\Downloads\Indian Internship\Dataset .csv")

# Step 2: Identify the restaurants with the highest and lowest number of votes
highest_votes_restaurant = df.loc[df['Votes'].idxmax()]
lowest_votes_restaurant = df.loc[df['Votes'].idxmin()]

print("Restaurant with the highest number of votes:")
print(highest_votes_restaurant[['Restaurant Name', 'Votes', 'Aggregate rating']])

print("\nRestaurant with the lowest number of votes:")
print(lowest_votes_restaurant[['Restaurant Name', 'Votes', 'Aggregate rating']])

# Step 3: Analyze correlation between the number of votes and the rating
# Map 'Rating color' to numeric values if it's categorical
rating_mapping = {
    'Dark Green': 5,
    'Green': 4,
    'Yellow': 3,
    'Orange': 2,
    'Red': 1,
    'White': 0
}

df['numeric_rating'] = df['Rating color'].map(rating_mapping)

# Drop rows with missing values in 'numeric_rating' or 'Votes'
df.dropna(subset=['numeric_rating', 'Votes'], inplace=True)

# Calculate correlation between the number of votes and the rating
correlation = df['Votes'].corr(df['numeric_rating'])
print(f"\nCorrelation between the number of votes and rating: {correlation:.2f}")

# Visualize the relationship between votes and rating
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Votes', y='numeric_rating', data=df)
plt.title('Votes vs. Rating')
plt.xlabel('Number of Votes')
plt.ylabel('Rating')
plt.show()

# In[2]:


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Read the dataset
df = pd.read_csv(r"C:\Users\Noel\Downloads\Indian Internship\Dataset .csv")

# Step 2: Identify the restaurants with the highest and lowest number of votes
highest_votes_restaurant = df.loc[df['Votes'].idxmax()]
lowest_votes_restaurant = df.loc[df['Votes'].idxmin()]

print("Restaurant with the highest number of votes:")
print(highest_votes_restaurant[['Restaurant Name', 'Votes', 'Aggregate rating']])

print("\nRestaurant with the lowest number of votes:")
print(lowest_votes_restaurant[['Restaurant Name', 'Votes', 'Aggregate rating']])

# Step 3: Analyze correlation between the number of votes and the rating
# Map 'Rating color' to numeric values if it's categorical
rating_mapping = {
    'Dark Green': 5,
    'Green': 4,
    'Yellow': 3,
    'Orange': 2,
    'Red': 1,
    'White': 0
}

df['numeric_rating'] = df['Rating color'].map(rating_mapping)

# Drop rows with missing values in 'numeric_rating' or 'Votes'
df.dropna(subset=['numeric_rating', 'Votes'], inplace=True)

# Calculate correlation between the number of votes and the rating
correlation = df['Votes'].corr(df['numeric_rating'])
print(f"\nCorrelation between the number of votes and rating: {correlation:.2f}")

# Visualize the relationship between votes and rating
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Votes', y='numeric_rating', data=df)
plt.title('Votes vs. Rating')
plt.xlabel('Number of Votes')
plt.ylabel('Rating')
plt.show()


# In[3]:


# Step 1: Clean and preprocess data
# Ensure columns 'Price range', 'Has Online delivery', and 'Has Table booking' are correctly typed
df['Price range'] = df['Price range'].astype('category')
df['Has Online delivery'] = df['Has Online delivery'].map({'Yes': 1, 'No': 0})
df['Has Table booking'] = df['Has Table booking'].map({'Yes': 1, 'No': 0})

# Step 2: Calculate the percentage of restaurants offering online delivery and table booking within each price range
price_range_online_delivery = df.groupby('Price range')['Has Online delivery'].mean() * 100
price_range_table_booking = df.groupby('Price range')['Has Table booking'].mean() * 100

# Step 3: Visualize the distribution using bar charts
plt.figure(figsize=(14, 6))

# Bar chart for online delivery
plt.subplot(1, 2, 1)
sns.barplot(x=price_range_online_delivery.index, y=price_range_online_delivery.values)
plt.title('Percentage of Restaurants Offering Online Delivery by Price Range')
plt.xlabel('Price Range')
plt.ylabel('Percentage')

# Bar chart for table booking
plt.subplot(1, 2, 2)
sns.barplot(x=price_range_table_booking.index, y=price_range_table_booking.values)
plt.title('Percentage of Restaurants Offering Table Booking by Price Range')
plt.xlabel('Price Range')
plt.ylabel('Percentage')

plt.tight_layout()
plt.show()

# Step 5: Determine if higher-priced restaurants are more likely to offer these services
# Analysis summary
print("Percentage of restaurants offering online delivery by price range:")
print(price_range_online_delivery)

print("\nPercentage of restaurants offering table booking by price range:")
print(price_range_table_booking)


# In[ ]:




