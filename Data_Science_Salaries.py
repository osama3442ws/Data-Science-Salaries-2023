import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import plotly.express as px
import pycountry
import numpy as np
import missingno as msno
from sklearn.impute import SimpleImputer
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, ConfusionMatrixDisplay

from sklearn import tree
#************************************************************************************
from sklearn import preprocessing

#************************************************************************************

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cluster      import KMeans
from sklearn.tree         import DecisionTreeRegressor
from sklearn.tree         import DecisionTreeClassifier
from sklearn.tree         import plot_tree
from sklearn.ensemble     import RandomForestClassifier
from sklearn.neighbors    import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#************************************************************************************
#Importing the dataset

data = pd.read_csv('H:\\Programming AI\\مجلد للتطبيق و التجريب\\Data Science Salaries 2023\\ds_salaries.csv')

print(data)
print(data.info())
print(data.describe())
print(data.nunique())
#************************************************************************************
# Transformation of the codes of the categorical variables
data['experience_level'] = data['experience_level'].replace({'SE': 'Expert', 'MI': 'Intermediate', 'EN': 'Junior', 'EX': 'Director'})

data['employment_type'] = data['employment_type'].replace({'FT': 'Full-time', 'CT': 'Contract', 'FL': 'Freelance', 'PT': 'Part-time'})

def country_name(country_code):
    try:
        return pycountry.countries.get(alpha_2=country_code).name
    except:
        return 'other'
    
data['company_location'] = data['company_location'].apply(country_name)
data['employee_residence'] = data['employee_residence'].apply(country_name)

#************************************************************************************
# Categorical variables

for column in ['work_year','experience_level','employment_type','company_size','remote_ratio','job_title','company_location']:
    print(data[column].unique())

#************************************************************************************
# Extract the "job title" column
job_titles = data['job_title']

# Calculate the frequency of each job title
title_counts = job_titles.value_counts()

# Extract the top 20 most frequent job titles
top_20_titles = title_counts.head(20)

# Create a DataFrame for the top 20 titles
top_20_df = pd.DataFrame({'Job Title': top_20_titles.index, 'Count': top_20_titles.values})

# Plotting the count plot
plt.figure(figsize=(12, 6))
sns.set(style="darkgrid")
ax = sns.barplot(data=top_20_df, x='Count', y='Job Title', palette='cubehelix')
plt.xlabel('Count')
plt.ylabel('Job Titles')
plt.title('Top 20 Most Frequent Job Titles')

# Add count labels to the bars
for i, v in enumerate(top_20_df['Count']):
    ax.text(v + 0.2, i, str(v), color='black', va='center')

plt.tight_layout()

# Calculate the number of individuals in each experience level
level_counts = data['experience_level'].value_counts()

# Create a pie chart
plt.figure(figsize=(7,12),dpi=80)
plt.pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%')
plt.title('Experience Level Distribution')

# Create a cross-tabulation of the two columns
cross_tab = pd.crosstab(data['experience_level'], data['company_size'])

# Create a heatmap using the cross-tabulation data
plt.figure(figsize=(10, 8))
sns.heatmap(cross_tab, annot=True, fmt="d", cmap='Reds')

plt.xlabel('Company Size')
plt.ylabel('Experience Level')
plt.title('Relationship between Experience Level and Company Size')

# Create bar chart
average_salary = data.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False)
top_ten_salaries = average_salary.head(10)

plt.figure(figsize=(15,10),dpi=80)
plt.bar(top_ten_salaries.index, top_ten_salaries)

# Add labels to the chart
plt.xlabel('Job')
plt.ylabel('Salary $')
plt.title('Average of the ten highest salaries by Job Titles')
plt.xticks(rotation=35, ha='right')
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
#************************************************************************************
common_jobs = ['Data Engineer', 'Data Scientist', 'Data Analyst', 'Machine Learning Engineer', 'Analytics Engineer','Research Scientist', 'Data Science Manager', 'Applied Scientist']
common_jobs = data[data['job_title'].isin(common_jobs)]

salary_common_jobs = common_jobs.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False)
remote_common_jobs = common_jobs.groupby('job_title')['remote_ratio'].mean().sort_values(ascending=False)
salary_common_country = common_jobs.groupby('company_location')['salary_in_usd'].mean().sort_values(ascending=False)

# Create bar chart
salary_common_jobs = common_jobs.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False)

plt.figure(figsize=(15,10),dpi=80)
plt.bar(salary_common_jobs.index, salary_common_jobs)

# Add labels to the chart
plt.xlabel('Job')
plt.ylabel('Salary $')
plt.title('Average salary for common Job Titles')
plt.xticks(rotation=20, ha='right')
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
#************************************************************************************
# Create bar chart
remote_common_jobs = common_jobs.groupby('job_title')['remote_ratio'].mean().sort_values(ascending=False)

plt.figure(figsize=(15,10),dpi=80)
plt.bar(remote_common_jobs.index, remote_common_jobs)

# Add labels to the chart
plt.xlabel('Job')
plt.ylabel('% remote')
plt.title('Remote rate by Job Titles')
plt.xticks(rotation=20, ha='right')
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

#************************************************************************************
# Create bar chart
salary_common_country = common_jobs.groupby('company_location')['salary_in_usd'].mean().sort_values(ascending=False)

plt.figure(figsize=(15,10),dpi=80)
plt.bar(salary_common_country.head(10).index, salary_common_country.head(10))

# Add labels to the chart
plt.xlabel('Country')
plt.ylabel('Salary $')
plt.title('Average of the 10 highest salaries of common jobs by country')
plt.xticks(rotation=20, ha='right')
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

plt.show()
plt.show()