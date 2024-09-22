#!/usr/bin/env python
# coding: utf-8

# # Kağan Tek
# ### kagantek2003@hotmail.com

# # Task 1 - Exploratory Data Analysis
# ### First of all I will try to do some EDA and try to gain insights regarding this dataset without any processing. After which I will be processing the data and continuing other steps of visualization.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


# In[2]:


file_path = "side_effect_data.xlsx"
data = pd.read_excel(file_path, sheet_name='Sheet1')
data.head()


# In[3]:


data.info()


# In[4]:


data.describe()


# ### I check the counts of the unique parameter types of different parameters to decide which correlations I can visualize.

# In[5]:


chronical_diseases = data["Kronik Hastaliklarim"].str.split(", ").explode()
print(chronical_diseases.unique())
print("Count of unique diseases: ", len(chronical_diseases.unique()))


# In[6]:


print(data['Alerjilerim'].unique())
print("\nUnique Allergy Counts: ", len(data['Alerjilerim'].unique()))


# In[7]:


print(data['Yan_Etki'].unique())
print("\nUnique Side Effects Count: ", len(data['Yan_Etki'].unique()))


# In[8]:


data["Ilac_Adi"].unique()


# ### From this I can say that there are some problems with the dataset that will need processing in the next task such as:
#     - Some side effects are identical in their effects such as "Mide Bulantisi" and "Bulanti"
#     - There are null values in certain categories like allergies, side effects, chronic diseases etc.
# 
# ### I proceed by adding an 'Age' column to the dataframe to make age distributive analysis

# In[9]:


data['Age'] = (pd.to_datetime('today') - data['Dogum_Tarihi']).dt.days // 365


# In[10]:


plt.figure(figsize=(20, 6))

# Age distribution
plt.subplot(1, 3, 1)
sns.histplot(data['Age'], bins=40, kde=True, color='blue')
plt.xlabel("Age")
plt.ylabel("Count")
plt.title('Age Distribution')

# Weight distribution
plt.subplot(1, 3, 2)
sns.histplot(data['Kilo'], bins=20, kde=True, color='green')
plt.title('Weight Distribution')

plt.subplot(1, 3, 3)
sns.histplot(data['Boy'], bins=20, kde=True, color='red')
plt.title('Height Distribution')

# Gender distribution
plt.figure(figsize=(20, 6))
sns.countplot(x='Cinsiyet', data=data)
plt.title('Gender Distribution')

plt.tight_layout()
plt.show()


# ### From the plots above we can understand that:
#  - Our dataset is populated in mostly between 25-35 years old and 45-55 years old people.
#  - Aside from that the average weight goes between 80-100 kilos
#  - For the height we can say that it is between 175-190.
# 
# ### We can continue to create plots that give us more insights within these parameters' relationship and gender as the hue

# In[11]:


# Create scatter plots for age, weight, and height with gender as the hue
plt.figure(figsize=(16, 6))

# Scatter plot for Age vs Weight
sns.lmplot(x='Age', y='Kilo', hue='Cinsiyet', data=data)

# Scatter plot Age vs Height by Gender
sns.lmplot(x='Age', y='Boy', hue='Cinsiyet', data=data)

# Scatter plot for Weight vs Height
sns.lmplot(x='Kilo', y='Boy', hue='Cinsiyet', data=data)


# ### The plots above show us the relationships between age, height and weight which are categorized by gender. Now it is time for me to move on to display the distribution of chronic ilnesses

# In[12]:


# Plot the counts of chronic illnesses
ch_disease = data["Kronik Hastaliklarim"].str.split(", ").explode().str.strip()
ch_illness_count = ch_disease.value_counts()

plt.figure(figsize=(10,6))
ch_illness_count.plot(kind='bar', color='skyblue')

plt.title('Top 10 Chronic Illnesses Count')
plt.xlabel('Chronic Illness')
plt.ylabel('Number of People')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()


# ### Since some people have multiple chronic illnesses I needed to split them by commas and visualize them by their count. However in order to gain insights about them with their relationships to other parameter I need to process the data to get their counts singularly within the same dataframe.

# In[13]:


sns.countplot(x='Alerjilerim', hue='Cinsiyet', data=data)
plt.title('Allergies by Gender')
plt.xlabel('Allergies')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set global parameters for font size and line thickness
plt.rcParams.update({'font.size': 12, 'lines.linewidth': 2})

# Plotting the relationship between allergies and side effects
plt.figure(figsize=(14, 6))
sns.countplot(x='Alerjilerim', hue='Yan_Etki', data=data)

# Title and axis labels
plt.title('Relationship Between Allergies and Side Effects', fontsize=16)
plt.xlabel('Allergies', fontsize=14)
plt.ylabel('Count', fontsize=14)

# Rotate x-axis labels for better readability
plt.xticks(rotation=90)

# Adjust the layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()


# ### I wanted to visualize the relationship between allergies and side effects from the medication however as it's seen above it isn't a very user friendly approach to visualize them in a single plot, therefore I utilized cross tabulation functions to determine percentages and visualize them one by one 

# In[15]:


import pandas as pd

# Create a crosstab to calculate the count of each side effect for each allergy
ef_al_cross = pd.crosstab(data['Yan_Etki'], data['Alerjilerim'])

# Calculate the percentage for each allergy's part of the side effects
ef_al_percent = ef_al_cross.div(ef_al_cross.sum(axis=1), axis=0) * 100

# Display the percentage distribution of side effects by allergy
print(ef_al_percent)


# In[16]:


import matplotlib.pyplot as plt

# Iterate over each allergy and plot the percentage distribution of side effects
for allergy in ef_al_percent.columns:
    plt.figure(figsize=(10, 6))
    ef_al_percent[allergy].dropna().sort_values(ascending=False).plot(kind='bar'
                                                                      , color='skyblue')
    plt.title(f'Side Effects Distribution for {allergy}')
    plt.xlabel('Side Effects')
    plt.ylabel(f'Percentage of Side Effects Attributed to {allergy}')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


# ### From the visualizations above it can be inferred that certain allergies and side effects are seen more together. 

# In[17]:


# Count side effects in each age group
side_effects_by_age = data.groupby('Age')['Yan_Etki'].value_counts(normalize=True).unstack()

# Plot the heatmap for side effects distribution by age group
plt.figure(figsize=(12, 8))
sns.heatmap(side_effects_by_age, cmap='Greens', annot=False)
plt.title('Side Effects Distribution by Age Group')
plt.show()


# In[18]:


# Plotting the side effect count distributions with gender as the hue
plt.figure(figsize=(14, 6))
sns.countplot(x='Yan_Etki', hue='Cinsiyet', data=data)
plt.title('Side Effect Count Distributions by Gender')
plt.xlabel('Side Effect')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# ### Analysing the above visualizations we can see the correlations of certain ages and side effects such as older individuals experience blood pressure problems as side effects and younger people have correlations with fatigue.
# 
# ### We can also determine that the type of side effects experienced may differ due to gender such as female individuals experience "Agizda Farkli Bir Tat" side effect more then men etc.
# 
# I want to do more data analysis however with the current state of the dataset the insights would not be as reliable and visualizing certain categorical parameters would prove to be hard, therefore I am moving on with data processing and then going to continue with certain visualizations.

# # Task 2 - Data Pre-Processing
# ### We can start with a heatmap to display missing values.

# In[19]:


plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap")
plt.show()


# ### We can see that this is very problematic as there are significant amounts off null data, I will be following certain strategies to solve this problem:
#     - First of all I will be using SimpleImputer to fill the empty columns of data.
#     - I will be filling the numerical values with the mean value
#     - I will be filling the categorical values with the most_frequent value

# In[20]:


from sklearn.impute import SimpleImputer

# For categorical columns the null columns will be filled with the 
# most frequent value
categoric_imputer = SimpleImputer(strategy='most_frequent')

# For numerical columns the null columns will be filled with the 
# mean value

numeric_imputer = SimpleImputer(strategy='mean')

# List of columns to impute
categoric_col = ['Cinsiyet', 'Il', 'Kan Grubu', 'Alerjilerim', 
                    'Kronik Hastaliklarim', 'Baba Kronik Hastaliklari', 
                    'Anne Kronik Hastaliklari', 
                    'Kiz Kardes Kronik Hastaliklari', 
                    'Erkek Kardes Kronik Hastaliklari']
numeric_col = ['Kilo', 'Boy'] 

# Apply imputers to the columns
data[categoric_col] = categoric_imputer.fit_transform(data[categoric_col])
data[numeric_col] = numeric_imputer.fit_transform(data[numeric_col])

# Check missing values
remaining_missing = data.isnull().sum()
remaining_missing


# In[21]:


#Checking the heatmap once again to see the difference and making sure the null values are taken care of.
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap")
plt.show()


# ### Now that the missing values are taken care of, it is time to encode the categorical parameter in order to visualize our data better. I will be using OneHotEncoder to encode the categorical variables.

# In[22]:


from sklearn.preprocessing import OneHotEncoder
# List of columns that contain chronic illnesses
chronic_illness_cols = ['Kronik Hastaliklarim', 'Baba Kronik Hastaliklari', 
                        'Anne Kronik Hastaliklari', 'Kiz Kardes Kronik Hastaliklari', 
                        'Erkek Kardes Kronik Hastaliklari']

# Create a set to store all unique illnesses
all_illnesses = set()

# Split the illnesses in each column and collect them
for col in chronic_illness_cols:
    data[col] = data[col].apply(lambda x: x.split(',') if isinstance(x, str) else [])
    for illnesses in data[col]:
        all_illnesses.update([illness.strip() for illness in illnesses])

# Create a one-hot encoding for each unique illness
for illness in all_illnesses:
    data[illness] = data[chronic_illness_cols].apply(lambda x: int(any(illness in illnesses for illnesses in x)), axis=1)

# Identify remaining categorical columns excluding chronic illness columns
remaining_categorical_columns = ['Cinsiyet', 'Il', 'Uyruk', 'Yan_Etki', 'Kan Grubu', 'Alerjilerim']

# Apply One-Hot Encoding to all remaining categorical columns
one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)  # Updated sparse_output
encoded_columns = pd.DataFrame(one_hot_encoder.fit_transform(data[remaining_categorical_columns]), 
                               columns=one_hot_encoder.get_feature_names_out(remaining_categorical_columns))

# Concatenate the encoded columns with the original dataframe
df_encoded = pd.concat([data.drop(columns=remaining_categorical_columns + chronic_illness_cols), 
                        encoded_columns], axis=1)

# Display the first few rows of the processed dataframe
print(df_encoded.head())


# ### Now it is time to standardize the data in order for the parameters to have the same variance and mean values, therefore making them more viable for machine learning algorithms.

# In[23]:


from sklearn.preprocessing import StandardScaler

# List of numerical columns to standardize
numerical_cols = ['Kilo', 'Boy']

# StandardScaler initialization
scaler = StandardScaler()

# Apply Z-score normalization to the numerical columns
df_encoded[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Display standardized data
print(df_encoded[numerical_cols].head())


# In[24]:


df_encoded


# ### Now that the data is processed we can do a few visulizations regarding the chronic ilnesses.

# In[25]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df_encoded is your one-hot encoded dataframe
# Recreate a simple 'Gender' column for easier plotting
df_encoded['Cinsiyet'] = df_encoded['Cinsiyet_Male'].apply(lambda x: 'Male' if x == 1 else 'Female')

# List of all chronic illness columns (replace these with the actual one-hot encoded chronic illness column names)
chronic_illness_cols = ['Hipertansiyon', 'Kan Hastaliklari', 'Kalp Hastaliklari', 'Diyabet', 'Diger', 
                        'KOAH', 'Astim', 'Kemik Erimesi', 'Kanser', 'Alzheimer', 'Guatr']  

# 1. Count Plot for Each Chronic Illness Per Gender
for illness in chronic_illness_cols:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df_encoded, x='Cinsiyet', hue=illness)
    plt.title(f'Distribution of {illness} by Gender')
    plt.show()

# 2. Heatmap of Chronic Illnesses per Gender
# Group the data by gender and calculate the mean (proportion) of chronic illnesses
gender_illness_summary = df_encoded.groupby('Cinsiyet')[chronic_illness_cols].mean()

# Plot a heatmap for better visualization
plt.figure(figsize=(10, 6))
sns.heatmap(gender_illness_summary.T, annot=True, cmap='Blues', cbar_kws={'label': 'Proportion of Gender with Illness'})
plt.title('Proportion of Chronic Illnesses per Gender')
plt.ylabel('Chronic Illness')
plt.xlabel('Gender')
plt.show()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# List of all chronic illnesses
chronic_illness_cols = ['Hipertansiyon', 'Kan Hastaliklari', 'Kalp Hastaliklari', 'Diyabet', 'Diger', 
                        'KOAH', 'Astim', 'Kemik Erimesi', 'Kanser', 'Alzheimer', 'Guatr']  

# List of all one-hot encoded side effect columns (replace these with actual side effect column names)
side_effect_cols = ['Yan_Etki_Kabizlik', 'Yan_Etki_Yorgunluk', 'Yan_Etki_Carpinti', 
                    'Yan_Etki_Sinirlilik', 'Yan_Etki_Deride Morarma', 'Yan_Etki_Bas Agrisi', 
                    'Yan_Etki_Gormede Bulaniklik', 'Yan_Etki_Gucsuzluk', 'Yan_Etki_Az Uyuma',
                    'Yan_Etki_Huzursuzluk', 'Yan_Etki_Mide Bulantisi', 'Yan_Etki_Kas Agrisi', 
                    'Yan_Etki_Istah Artisi', 'Yan_Etki_Terleme', 'Yan_Etki_Karin Agrisi', 
                    'Yan_Etki_Tansiyon Yukselme', ' Yan_Etki_Gec Bosalma', 'Yan_Etki_Ishal', 
                    'Yan_Etki_Bulanti', 'Yan_Etki_Tansiyon Dusuklugu', 'Yan_Etki_Uykululuk Hali']

# Chronic illnesses vs. Side Effects
for illness in chronic_illness_cols:
    for side_effect in side_effect_cols:
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df_encoded, x=side_effect, hue=illness)
        plt.title(f'Relationship Between {illness} and Side Effect ({side_effect})')
        plt.xticks(rotation=45, ha="right")
        plt.show()


# ### The above series' of visualization show us the relationships between chronic ilnesses and side effects. For example people that have "Kan Hastaliklari" make up for almost %50 of the people that experience Bas Agrisi and Kas Agrisi. 

# In[ ]:




