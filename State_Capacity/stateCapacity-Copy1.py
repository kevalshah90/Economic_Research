#!/usr/bin/env python
# coding: utf-8

# The general assumption is that with strong state capacity and economic development, complex rules will be enforced. Epstein argues that this complexity imposed creates private, public and social costs. 
# 
# 'Simple rules for the developing world' paper argues that this insights holds true even for developing nations. Prematurely adopting complex rules with limited state capacity can actually reduce the ability of weak states to develop greater state capacity. 
# 
# Empirical Task:
# 
# > Countries with weak / poor state capacity and low regulatory complexity have better outcomes 
# 
# vs
# 
# > Countries with weak / poor state capacity and high regulatory complexity have poor outcomes. 
# 
# We use the following measures: 
# 
# 1. Chong - Letter Grading Government Efficiency for State Capacity
# 
# 2. Courts data - formalism index 
# 
#     The index measures substantive and procedural statutory intervention in judicial cases at lower-level civil trial courts, and
#     is formed by adding up the following indices: (i) professionals vs. laymen, (ii) written vs. oral elements, (iii) legal justification, (iv) statutory regulation of evidence, (v) control of superior review, (vi) engagement formalities, and (vii) independent procedural actions. The index ranges from 0 to 7, where 7 means a higher level of control or intervention in the judicial process.
#     
# 
# 3. World Bank's ease of doing business as measure for regulatory complexity. Additional measures also include: Ease of Starting a business, construction permits, Registering a property 
# 
# 4. Outcome is measured as GDP Per Capita 
# 
# Other variables to consider (possible control variables): 
# 
# 1. Income per capita
# 2. Democracy Index
# 3. Communist 
# 4. Population (predictor of complexity) 

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.compat import lzip


# In[3]:


# Read .dta Stata file - Regulations
df_reg = pd.io.stata.read_stata('regulation/population_regulation.dta')


# In[4]:


df_reg.columns


# In[5]:


# Read .dta State file - Courts
df_courts = pd.io.stata.read_stata('courts/courts_database_july06.dta')
df_courts.sample(5)


# In[6]:


# Read Country codes data

df_country = pd.read_excel('data/country_codes.xlsx')


# In[7]:


# Country to lowercase 

df_courts['country'] = df_courts['country'].str.lower()

df_country['Country'] = df_country['Country'].str.lower()

df_courts_v1 = pd.merge(df_courts,
                        df_country[['Country','Code3']],
                        left_on = 'country',
                        right_on = 'Country')

df_courts_v1 = df_courts_v1.drop('Country', 1)

df_courts_v1.sample(5)


# #### Measures for Regulatory Complexity, State Capacity (Postal mail), World Bank's Governance Indicators - Government Effectiveness, Income per Capita, GDP per Capita, Voice and accountability
# 
# Create a combined pandas dataframe with all of these measures. 

# In[8]:


df = pd.read_excel('data/combinedData.xlsx', None)
df.keys()


# #### State Capacity Mail Data
# 
# Chong, Alberto, Rafael LaPorta, Florencio Lopez-de-Silanes, and Andrei Shleifer. 2014. “Letter Grading Government Efficiency.” Journal of European Economic Association 12 (2): 277-299. 
# 
# Source: https://scholar.harvard.edu/shleifer/publications/letter-grading-government-efficiency

# In[9]:


df_sc2 = pd.read_excel('data/combinedData.xlsx', sheet_name = "State Capacity - Postal Mail")
df_sc2.sample(5)


# #### GDP Per Capita

# In[10]:


df_gdp = pd.read_excel('data/combinedData.xlsx', sheet_name="GDP Per Capita")

# Calculate percent change in GDP Per Capita between 2011 and 2016
df_gdp['gdp_percent_change_2011_2016'] = round(((df_gdp[2016] - df_gdp[2011])/df_gdp[2011]),2)*100


# #### Income per Capita

# In[25]:


df_incpc = pd.read_excel('data/combinedData.xlsx', sheet_name="Income Per Capita")

# Calculate percent change in Income Per Capita between 2011 and 2016
df_incpc['incpc_percent_change_2011_2016'] = round(((df_incpc[2016] - df_incpc[2011])/df_incpc[2011]),2)*100


# #### Labor Force 

# In[46]:


df_labor = pd.read_excel('data/combinedData.xlsx', sheet_name="Labor Force")

# Calculate percent change in labor force between 2011 and 2016
df_labor['labor_percent_change_2011_2016'] = round(((df_labor[2016] - df_labor[2011])/df_labor[2011]),2)*100


# #### Voice, Accountability, Participation in government election, freedom of expression etc. 
# 
# Source: https://info.worldbank.org/governance/wgi/Home/Documents 

# In[27]:


df_voice = pd.read_excel('data/combinedData.xlsx', sheet_name = "Voice")
df_voice.sample(5)


# #### World Bank Data Irregularities 
# 
# https://www.worldbank.org/en/news/statement/2021/09/16/statement-on-release-of-investigation-into-data-irregularities-in-doing-business-2018-and-2020 
# 
# In light of world bank data irregularities, we will alter our analysis and model and use courts data for complexity measures. 
# 

# In[56]:


df_full = pd.merge(df_sc2,
                   df_courts_v1[['country','Code3','all_indexn_c','consistent']],
                   left_on = ['Code'],
                   right_on = ['Code3'])

df_full = df_full.drop(['country','Code3'],1)

# Rename columns
df_full = df_full.rename(columns={'Got the letter back': 'got_letter_back',
                                  'Got the letter back in 90 days': 'letter_back_90_days',
                                  'Avg. number of days to get back the letter': 'avg_days_get_letter_back'}
                        )

# Merge df with gdp data
df_full_v1 = pd.merge(df_full,
                      df_gdp[['Country Code','gdp_percent_change_2011_2016',2012]],
                      left_on = ['Code'],
                      right_on = ['Country Code']
                      )

df_full_v1 = df_full_v1.rename(columns = {2012: '2012_gdp'})

df_full_v1 = df_full_v1.drop('Country Code', 1)

# Merge df with Income data
df_full_v2 = pd.merge(df_full_v1,
                      df_incpc[['Country Code','incpc_percent_change_2011_2016',2012]],
                      left_on = ['Code'],
                      right_on = ['Country Code'])

df_full_v2 = df_full_v2.rename(columns = {2012: '2012_incpc'})

df_full_v2 = df_full_v2.drop('Country Code', 1)

# Merge df with Voice / Democracy data

df_full_v3 = pd.merge(df_full_v2,
                      df_voice[['Code',2012]],
                      left_on = ['Code'],
                      right_on = ['Code'])

df_full_v3 = df_full_v3.rename(columns = {2012 : '2012_voice'})

# Merge df with Labor
df_full_v4 = pd.merge(df_full_v3,
                      df_labor[['Country Code', 'labor_percent_change_2011_2016', 2012]],
                      left_on = ['Code'],
                      right_on = ['Country Code'])

df_full_v4 = df_full_v4.rename(columns = {2012 : '2012_labor'})


# #### Scatterplot of regulatory complexity from court data and state capacity

# In[57]:


# Plot scatter of ease of courts - formalism index and state capacity 

plt.figure(figsize=(15,8))
ax = sns.scatterplot(x='all_indexn_c', y='letter_back_90_days', data = df_full_v4)

for line in range(0,df_full_v4.shape[0]):
    
     ax.text(df_full_v4.all_indexn_c[line]+0.05, df_full_v4.letter_back_90_days[line], 
             df_full_v4.Code[line], 
             horizontalalignment='left', 
             rotation=65,
             size='medium', 
             color='black', 
             weight='normal')
        

# Add Quadrants
ax.axhline(0.5, color="blue", linestyle="--")
ax.axvline(3.5, color="blue", linestyle="--")

# # Add Annotations
style = dict(size=12, color='darkgray')

ax.text(1.5, 0.3, "Low Complexity, Low State Capacity", **style)
ax.text(1.5, 0.8, "Low Complexity, High State Capacity", **style)
ax.text(4.5, 0.3, "High Complexity, Low State Capacity", **style)
ax.text(4.5, 0.8, "High Complexity, High State Capacity", **style)

plt.title('Plot Countries across Complexity and State Capacity measures')
# Set x-axis label
plt.xlabel('Complexity - Formalism Index')
# Set y-axis label
plt.ylabel('State Capacity - postal return effectiveness')

#adjust_text(ax.text, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
plt.show()


# The above visual reaffirms that weak states have complex rules as discussed in the paper, likely due the fact that weak states try to imitate the complex regulatory structure of states with strong state capacity. 

# ##### Let's fit a regression with Percentage change in GDP Per Capita in 5 years since State Capacity Observation as Dependent variable, 
# 
# > control variables include measure for GDP Per Capita Income, Democracy 

# In[61]:


# Fit regression model

model = sm.formula.ols(formula="gdp_percent_change_2011_2016 ~ all_indexn_c * letter_back_90_days + consistent + incpc_percent_change_2011_2016", data=df_full_v4).fit()

print(model.summary(yname='gdp_change'))


# In[354]:


# Regression Diagnostics, check for linearity and equal variance

pred_val = model.fittedvalues.copy()
true_val = df6['gdp_percent_change_2011_2016'].values.copy()
residual = true_val - pred_val

fig, ax = plt.subplots(figsize=(10,6))
_ = ax.scatter(pred_val, residual)

plt.xlabel('Fitted')
plt.ylabel('Residuals')
plt.title('Fitted vs Residual Plot')
plt.show()


# In[339]:


# Plot
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(model, fig=fig)


# In[ ]:




