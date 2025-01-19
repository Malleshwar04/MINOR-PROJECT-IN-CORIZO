#!/usr/bin/env python
# coding: utf-8
Introduction:
           This report presents an analysis of body measurement data for males and females, based on datasets from the National Health and Nutrition Examination Survey (NHANES). The goal is to explore and compare the distributions, relationships, and derived metrics such as Body Mass Index (BMI), waist-to-height ratios, and waist-to-hip ratios.
           The analysis is structured into multiple sections, including data preparation, visualization, and statistical evaluation.Methodology:
   Libraries and Tools:
          The following Python libraries were used:
              NumPy - For numerical data handling and computations.
              Pandas - For data manipulation and organization.
              Matplotlib and Seaborn - For data visualization.

   Datasets:
          Two CSV files containing body measurements for males and females were used:
                nhanes_adult_male_bmx_2020.csv
                nhanes_adult_female_bmx_2020.csv
          Each dataset contains the following columns:
                Weight (kg)
                Standing Height (cm)
                Upper Arm Length (cm)
                Upper Leg Length (cm)
                Arm Circumference (cm)
                Hip Circumference (cm)
                Waist Circumference (cm) 
# In[2]:


#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#datasets
male_file = "nhanes_adult_male_bmx_2020_x.csv"
female_file = "nhanes_adult_female_bmx_2020_x.csv"
male = np.loadtxt(male_file, delimiter=',', skiprows=1)
female = np.loadtxt(female_file, delimiter=',', skiprows=1)

Create histograms for male and female weights:
       This initializes a figure for plotting histograms to visualize the distribution of weights for males and females.
       We draw two separate histograms:
                Top plot for Female weights.
                Bottom plot for Male weights. The x-axis range is made identical for both to allow easy comparison of weight distributions.
# In[3]:


# Histograms for male and female weights
plt.figure(figsize=(10, 8))

# Female weights

#Creates a subplot for the histogram of female weights in the top section of the figure.
plt.subplot(2, 1, 1)

plt.hist(female[:, 0], bins=20, color='pink', alpha=0.7, label='Female Weights')
plt.title('Female Weights Distribution')
plt.xlabel('Weight (kg)')
plt.ylabel('Frequency')
plt.legend()

# Male weights

#Creates a subplot for the histogram of male weights in the bottom section of the figure.
plt.subplot(2, 1, 2)

plt.hist(male[:, 0], bins=20, color='blue', alpha=0.7, label='Male Weights')
plt.title('Male Weights Distribution')
plt.xlabel('Weight (kg)')
plt.ylabel('Frequency')
plt.legend()

# x-axis limits are the same for both
plt.subplots_adjust(hspace=0.5)
plt.show()

Boxplot comparison of male and female weights:
            A boxplot is created to summarize male and female weight distributions side by side.
# In[4]:


# Boxplot comparison of male and female weights
plt.figure(figsize=(8, 6))

#This section creates a boxplot to compare the weight distributions between males and females
#Highlighting differences in spread and central tendencies.
plt.boxplot([female[:, 0], male[:, 0]], labels=['Female', 'Male'], patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))

plt.title('Boxplot of Male and Female Weights')
plt.ylabel('Weight (kg)')
plt.show()

Basic numerical aggregates:
        We calculate key statistics for weights:
                Mean - The average weight.
                Median - The middle value when weights are sorted.
                Standard deviation - How much weights vary around the average.
                Skewness - Whether the distribution is symmetrical or skewed to one side.
# In[11]:


# Basic numerical aggregates
# Defines a function
def summarize_weights(weights, gender):
    
    mean = np.mean(weights)
    median = np.median(weights)
    std_dev = np.std(weights)
    skewness = pd.Series(weights).skew()
    print(f"{gender} Weights Summary:")
    print(f"\n Mean: {mean:.2f},\n Median: {median:.2f},\n Std Dev: {std_dev:.2f},\n Skewness: {skewness:.2f}\n")

summarize_weights(female[:, 0], "Female")
summarize_weights(male[:, 0], "Male")

Add BMI to female dataset:
        We calculate the Body Mass Index (BMI) for females using the formula:

           𝐵𝑀𝐼 =  Weight (kg) / (Height (m))^2
 
The calculated BMI is added as a new column to the female dataset.
# In[13]:


# Add BMI to female dataset
bmi = female[:, 0] / ((female[:, 1] / 100) ** 2) 
female = np.column_stack((female, bmi))

Standardize the female dataset:
        All columns in the female dataset are converted into z-scores (standardized values):

       𝑧 = Value − Mean / Standard Deviation
       
This makes different measurements (e.g., weight and height) comparable.
# In[14]:


# This standardizes the female dataset by converting all columns to z-scores 
zfemale = (female - np.mean(female, axis=0)) / np.std(female, axis=0)

Scatterplot matrix:
             We create a scatterplot matrix for standardized variables like weight, height, waist, hip, and BMI to visualize relationships. 
             It also computes:
                       Pearson correlation - Linear relationships.
                       Spearman correlation - Ranked relationships.
# In[15]:


# Scatterplot matrix
# weight, height, waist, hip, BMI
variables = [0, 1, 6, 5, 7]

# This creates a scatterplot matrix for selected standardized variables to visualize relationships and correlations
sns.pairplot(pd.DataFrame(zfemale[:, variables], columns=['Weight', 'Height', 'Waist', 'Hip', 'BMI']),
             diag_kind='kde', corner=True)

plt.suptitle('Scatterplot Matrix of Standardized Female Variables', y=1.02)
plt.show()

# Compute correlations
# Computes Pearson correlation coefficients to measure linear relationships between variables.
pearson_corr = np.corrcoef(zfemale[:, variables], rowvar=False)

spearman_corr = pd.DataFrame(zfemale[:, variables]).corr(method='spearman')
print("Pearson Correlation:\n", pearson_corr)
print("Spearman Correlation:\n", spearman_corr)

Waist-to-height and waist-to-hip ratios:
          Two new ratios are calculated:
                Waist-to-height ratio - How waist circumference compares to height.
                Waist-to-hip ratio - How waist circumference compares to hip circumference. These ratios are added as new columns for both male and female datasets.
# In[17]:


# Waist-to-height and waist-to-hip ratios
# This computes waist-to-height and waist-to-hip ratios for male participants, later used for comparison
male_ratios = male[:, [6, 5]] / male[:, [1, 5]]  
female_ratios = female[:, [6, 5]] / female[:, [1, 5]]

# Append these columns to the datasets
male = np.column_stack((male, male_ratios))
female = np.column_stack((female, female_ratios))

Boxplot for ratios:
        We create a boxplot to compare the ratios:
            Waist-to-height ratio for males and females.
            Waist-to-hip ratio for males and females. This shows the central tendency and spread for these ratios.
# In[18]:


# Creates a boxplot comparing waist-to-height and waist-to-hip ratios for males and females.
plt.figure(figsize=(10, 6))
plt.boxplot([female[:, -2], female[:, -1], male[:, -2], male[:, -1]],
            labels=['Female W/Ht', 'Female W/Hip', 'Male W/Ht', 'Male W/Hip'],
            patch_artist=True, boxprops=dict(facecolor='lightgreen', color='green'),
            medianprops=dict(color='red'))
plt.title('Comparison of Ratios')
plt.ylabel('Ratio')
plt.show()

Standardized BMI extremes:
        We identify and display:
                The 5 individuals with the lowest BMI and The 5 individuals with the highest BMI. This provides insights into extreme cases in the dataset.
# In[19]:


# This identifies individuals with the lowest and highest BMI by sorting the standardized BMI values.
sorted_bmi = np.argsort(zfemale[:, 7])
lowest_bmi = zfemale[sorted_bmi[:5]]
highest_bmi = zfemale[sorted_bmi[-5:]]
print("Lowest BMI Individuals:\n", lowest_bmi)
print("Highest BMI Individuals:\n", highest_bmi)


# In[ ]:




