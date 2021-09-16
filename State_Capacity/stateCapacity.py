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
# 2. World Bank's ease of doing business as measure for regulatory complexity. Additional measures also include: Ease of Starting a business, construction permits, Registering a property 
# 
# 3. Outcome is measured as GDP Per Capita 
# 
# Other variables to consider (possible control variables): 
# 
# 1. Income per capita
# 2. Democracy Index
# 3. Communist 
# 4. Population (predictor of complexity) 

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


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


# #### Measures for Regulatory Complexity, State Capacity (Postal mail), World Bank's Governance Indicators - Government Effectiveness, Income per Capita, GDP per Capita, Voice and accountability
# 
# Create a combined pandas dataframe with all of these measures. 

# In[6]:


df = pd.read_excel('data/combinedData.xlsx', None)
df.keys()


# #### Regulatory Complexity - Low Income and lower middle Income as defined by World Bank <sup>1</sup>
# 
# Source: https://www.doingbusiness.org/en/doingbusiness
# 
# 1 https://datahelpdesk.worldbank.org/knowledgebase/articles/906519-world-bank-country-and-lending-groups 

# In[7]:


df_complexity = pd.read_excel('data/combinedData.xlsx', sheet_name = "Complexity")
df_complexity.sample(5)


# In[8]:


df_complexity_v1 = df_complexity[df_complexity['Year'] == 'DB2011']
df_complexity_v1.shape


# In[9]:


df_comp_sample = df_complexity_v1[['Economy','Country Code','Year','score_start_business']]
df_comp_sample.dropna()
df_comp_sample.reset_index(inplace=True, drop=True)


# #### Government Effectiveness
# 
# Source: https://info.worldbank.org/governance/wgi/

# In[10]:


df_sc1 = pd.read_excel('data/combinedData.xlsx', sheet_name = "Government Effectiveness")
df_sc1.sample(5)


# #### State Capacity Mail Data
# 
# Chong, Alberto, Rafael LaPorta, Florencio Lopez-de-Silanes, and Andrei Shleifer. 2014. “Letter Grading Government Efficiency.” Journal of European Economic Association 12 (2): 277-299. 
# 
# Source: https://scholar.harvard.edu/shleifer/publications/letter-grading-government-efficiency

# In[11]:


df_sc2 = pd.read_excel('data/combinedData.xlsx', sheet_name = "State Capacity - Postal Mail")
df_sc2.sample(5)


# #### GDP Per Capita

# In[12]:


df_gdp = pd.read_excel('data/combinedData.xlsx', sheet_name="GDP Per Capita")

# Calculate percent change in GDP Per Capita between 2011 and 2016
df_gdp['gdp_percent_change_2011_2016'] = round(((df_gdp[2016] - df_gdp[2011])/df_gdp[2011]),2)*100


# #### Income per Capita

# In[13]:


df_incpc = pd.read_excel('data/combinedData.xlsx', sheet_name="Income Per Capita")

# Calculate percent change in Income Per Capita between 2011 and 2016
df_incpc['incpc_percent_change_2011_2016'] = round(((df_incpc[2016] - df_incpc[2011])/df_incpc[2011]),2)*100


# #### Labor Force 

# In[14]:


df_labor = pd.read_excel('data/combinedData.xlsx', sheet_name="Labor Force")

# Calculate percent change in labor force between 2011 and 2016
df_labor['labor_percent_change_2011_2016'] = round(((df_labor[2016] - df_labor[2011])/df_labor[2011]),2)*100


# #### Voice, Accountability, Participation in government election, freedom of expression etc. 
# 
# Source: https://info.worldbank.org/governance/wgi/Home/Documents 

# In[15]:


df_voice = pd.read_excel('data/combinedData.xlsx', sheet_name = "Voice")
df_voice.sample(5)


# ### Generate a complete dataset with pandas join / merge

# In[16]:


# State Capacity - mail and complexity (business)
df1 = pd.merge(df_sc2, 
               df_comp_sample[['Country Code','Year','score_start_business']], 
               left_on = ['Code'],
               right_on = ['Country Code'])

df1 = df1.drop('Country Code', 1)

# Rename columns
df1 = df1.rename(columns={'Got the letter back': 'got_letter_back',
                          'Got the letter back in 90 days': 'letter_back_90_days',
                          'Avg. number of days to get back the letter': 'avg_days_get_letter_back'}
                )

# State Capacity - government effectivness
df2 = pd.merge(df1,
               df_sc1[['Code',2011]],
               left_on = ['Code'],
               right_on = ['Code'])

# Rename columns
df2 = df2.rename(columns={2011: 'government_effectiveness_2011'})

# Join gdp percentage change
df3 = pd.merge(df2,
               df_gdp[['Country Code','gdp_percent_change_2011_2016']],
               left_on = ['Code'],
               right_on = ['Country Code'])

df3 = df3.drop('Country Code', 1)

# Join democracy, voice indicator
df4 = pd.merge(df3,
               df_voice[['Code',2011]],
               left_on = ['Code'],
               right_on = ['Code'])

# Rename columns
df4 = df4.rename(columns={2011: 'democracy_index'})

# Join Per Capita Income
df5 = pd.merge(df4,
               df_incpc[['Country Code','incpc_percent_change_2011_2016']],
               left_on = ['Code'],
               right_on = ['Country Code'])

# Rename columns
df5 = df5.rename(columns={2011: 'income_per_capita'})

df5 = df5.drop('Country Code', 1)

# Join Labor Force
df6 = pd.merge(df5,
               df_labor[['Country Code','labor_percent_change_2011_2016']],
               left_on = ['Code'],
               right_on = ['Country Code'])

df6 = df6.drop('Country Code', 1)


# #### Scatterplot of ease of doing business and state capacity

# In[17]:


# Plot scatter of ease of doing business and state capacity 

plt.figure(figsize=(15,8))
ax = sns.scatterplot(x='score_start_business', y='letter_back_90_days', data = df1)

for line in range(0,df1.shape[0]):
     ax.text(df1.score_start_business[line]+0.5, df1.letter_back_90_days[line], 
             df1.Code[line], horizontalalignment='left', rotation=65,
             size='medium', color='black', weight='semibold')
        
# get labels
    
# texts = [ax.text(df1['score_start_business'], 
#                  df1['letter_back_90_days'], 
#                  df1['Code']) for i in range(len(df1))]
# adjust_text(texts)

# Add Quadrants
ax.axhline(0.5, color="blue", linestyle="--")
ax.axvline(55, color="blue", linestyle="--")

# Add Annotations
style = dict(size=12, color='darkgray')

ax.text(20, 0.3, "Low Complexity, Low State Capacity", **style)
ax.text(30, 0.8, "Low Complexity, High State Capacity", **style)
ax.text(65, 0.4, "High Complexity, Low State Capacity", **style)
ax.text(60, 0.7, "High Complexity, High State Capacity", **style)


plt.title('Plot Countries across Complexity and State Capacity measures')
# Set x-axis label
plt.xlabel('Complexity - measure for starting a business')
# Set y-axis label
plt.ylabel('State Capacity - postal return effectiveness')

#adjust_text(ax.text, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
plt.show()


# The above visual reaffirms that weak states have complex rules as discussed in the paper, likely due the fact that weak states try to imitate the complex regulatory structure of states with strong state capacity. 

# In[18]:


# Plot scatter of gdp growth and state capacity 

plt.figure(figsize=(15,8))
ax = sns.scatterplot(x='gdp_percent_change_2011_2016', y='letter_back_90_days', data = df4)

for line in range(0,df4.shape[0]):
     ax.text(df4.gdp_percent_change_2011_2016[line]+0.5, df4.letter_back_90_days[line], 
             df4.Code[line], horizontalalignment='left', rotation=65,
             size='medium', color='black', weight='semibold')
        
# get labels
    
# texts = [ax.text(df1['score_start_business'], 
#                  df1['letter_back_90_days'], 
#                  df1['Code']) for i in range(len(df1))]
# adjust_text(texts)

# Add Quadrants
#ax.axhline(0.5, color="blue", linestyle="--")
#ax.axvline(55, color="blue", linestyle="--")

# Add Annotations
#style = dict(size=12, color='darkgray')

#ax.text(20, 0.3, "Low Complexity, Low State Capacity", **style)
#ax.text(30, 0.8, "Low Complexity, High State Capacity", **style)
#ax.text(65, 0.4, "High Complexity, Low State Capacity", **style)
#ax.text(60, 0.7, "High Complexity, High State Capacity", **style)


plt.title('Plot Countries across Per Capita GDP Growth outcomes and State Capacity measures')
# Set x-axis label
plt.xlabel('Per Capita GDP Growth in 5 years, 2011 - 2016')
# Set y-axis label
plt.ylabel('State Capacity - postal return effectiveness')

#adjust_text(ax.text, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
plt.show()


# In[19]:


# Plot scatter of gdp growth and state capacity 

plt.figure(figsize=(15,8))
ax = sns.scatterplot(x='gdp_percent_change_2011_2016', y='score_start_business', data = df4)

for line in range(0,df4.shape[0]):
     ax.text(df4.gdp_percent_change_2011_2016[line]+0.5, df4.score_start_business[line], 
             df4.Code[line], horizontalalignment='left', rotation=65,
             size='medium', color='black', weight='semibold')
        
# get labels
    
# texts = [ax.text(df1['score_start_business'], 
#                  df1['letter_back_90_days'], 
#                  df1['Code']) for i in range(len(df1))]
# adjust_text(texts)

# Add Quadrants
#ax.axhline(0.5, color="blue", linestyle="--")
#ax.axvline(55, color="blue", linestyle="--")

# Add Annotations
#style = dict(size=12, color='darkgray')

#ax.text(20, 0.3, "Low Complexity, Low State Capacity", **style)
#ax.text(30, 0.8, "Low Complexity, High State Capacity", **style)
#ax.text(65, 0.4, "High Complexity, Low State Capacity", **style)
#ax.text(60, 0.7, "High Complexity, High State Capacity", **style)


plt.title('Plot Countries across Per Capita GDP Growth outcomes and Regulatory complexity measures')
# Set x-axis label
plt.xlabel('Per Capita GDP Growth in 5 years, 2011 - 2016')
# Set y-axis label
plt.ylabel('Regulatory Complexity')

#adjust_text(ax.text, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
plt.show()


# #### Read and visualize complexity time series data

# In[20]:


df_complexity_ts = pd.read_excel('data/combinedData.xlsx', sheet_name = "Ease Biz Time Series")
df_complexity_ts = df_complexity_ts.rename(columns={'Score-Starting a business': 'Score-Start-business'})
df_complexity_ts = df_complexity_ts[['Economy','Year','Score-Start-business']]
df_complexity_ts.dropna(inplace=True)
df_complexity_ts.reset_index(inplace=True, drop=True)
df_complexity_ts['Year'] = df_complexity_ts['Year'].str.replace('DB','')
df_complexity_ts['Year'] = df_complexity_ts['Year'].astype(int)
df_complexity_ts.sample(5)


# In[21]:


# Filter by List of Countries - for simplicity and plot clarity
countries = ['Afghanistan','Bangladesh - Dhaka', 'Egypt, Arab Rep.', 'Ethiopia', 'India - Mumbai', 'Indonesia - Jakarta', 'Kenya', 'Myanmar', 'Nigeria', 'Philippines', 'Ukraine', 'Vietnam']
df2 = df_complexity_ts[df_complexity_ts['Economy'].isin(countries)]

plt.figure(figsize=(15,8))

ax = sns.lineplot(data=df2, x="Year", y="Score-Start-business", hue="Economy", style="Economy")

plt.title("Time Series Plot of Regulatory Complexity (as measured by ease of starting a business)")
plt.ylabel('Measure of Complexity - Starting a Business')
plt.show()


# Complexity has increased over the years in developing, low and lower middle income countries as indicated in this plot.

# ##### Let's fit a regression with Percentage change in GDP in 5 years since State Capacity Observation as Dependent variable, 
# > Ratio of complexity and state capacity as independent variable 
# 
# > control variables include measure for GDP Per Capita Income, Democracy 

# In[347]:


df6 = df6.dropna()

model = sm.formula.ols(formula="gdp_percent_change_2011_2016 ~ government_effectiveness_2011 * score_start_business + democracy_index + labor_percent_change_2011_2016", data=df6).fit()

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




