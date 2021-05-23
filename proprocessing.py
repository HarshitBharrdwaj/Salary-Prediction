#%% Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from warnings import filterwarnings
filterwarnings('ignore')
#import datetime as dt
#import sweetviz as sv

#%%Reading the train dataset

df = pd.read_excel('Train.xlsx')

#%%Dropping Irrelavant Columns

df.columns

df.drop(['Unnamed: 0', 'ID', 'CollegeID', 'CollegeCityID','10board', '12board'], axis = 1, inplace = True)

#%% Feature Extraction 'SubjectCount' which has the count of subject student opts

df['SubjectCount'] = df[df.iloc[:,20:28]!=-1].count(axis=1)

#%% Feature extraction Total Elective Marks which has total marks of only elective subjects

# As we can see other than domain other features of elective subjects has very big values, thus we need to 
# scale down the values.

subjects = pd.DataFrame(df.iloc[:,21:28][df.iloc[:,21:28]!=-1].sum(axis=1))

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()

subjects[0] = mms.fit_transform(subjects)

# Replacing -1 from the elective subject features with 0 for ease of calculation.

for i in (df.iloc[:,20:28].columns):
    df[i].replace(-1, 0, inplace = True)

df['SubjectTotal'] = subjects[0] + df['Domain']

#%% Encoding the Elective Subject Features by replacing the respective marks with 1.

def enc_Elective_Subjects(a):
    if a>0: 
        return 1
    else:
        return 0

for i in (df.iloc[:, 20:28].columns):
    df[i] = list(map(enc_Elective_Subjects, df[i]))

#%%Age

df['DOB'] = pd.to_datetime(df['DOB'])
df['today'] = pd.to_datetime('2015-12-31')

df['Age'] = (((df['today']-df['DOB']).dt.days)/365).round(1)

df.drop(['today', 'DOB'], axis=1, inplace = True)

#%%Handling DOL & DOL to YOE

df['DOL'].replace('present',np.nan,inplace=True)
df['DOL']= pd.to_datetime(df['DOL'])
df['DOL'] =df['DOL'].fillna(pd.to_datetime('2015-12-31'))
df['YOE']= (((df['DOL']-df['DOJ']).dt.days)/365).round(1)

#Checking for incorrect records as some of the values in YOE are negative values.

df[df['YOE']<0]

(df[df['YOE']<0].shape[0]/df.shape[0])*100

# There are 40 rows of data having incorrect values.
# After looking at the data we can say that the wrong values are entered for the respective columns.
# Since the number of records are very low, i.e., aound 1% of total observations we can drop the rows.

df.drop(df[df['YOE']<0].index, inplace = True)

df.drop(['DOL','DOJ'], axis = 1, inplace = True)

#%%Resetting the index for further use.

df = df.reset_index(drop = True)

#%% Handeling datatypes

df.dtypes

df['12graduation'] = df['12graduation'].astype('object')
df['GraduationYear'] = df['GraduationYear'].astype('object')

#%%Separating Numerical and Categorical features

numdf = df.select_dtypes(include = np.number)

catdf = df.select_dtypes(exclude = np.number)

#%% Finding missing values numerical features

numdf.isnull().sum()

#No null values

#%% Data cleaning of categorical values.

#%% Designation Column Cleaning

#z = pd.Series(catdf['Designation'].unique()).sort_values()

catdf['Designation'].replace('assistant systems engineer', 'assistant system engineer', inplace = True)

catdf['Designation'].replace('assistant system engineer - trainee', 'assistant system engineer trainee', inplace = True)

catdf['Designation'].replace('associate software engg', 'associate software engineer', inplace = True)

catdf['Designation'].replace('asst. manager', 'assistant manager', inplace = True)

catdf['Designation'].replace('business development managerde', 'business development manager', inplace = True)

catdf['Designation'].replace('business systems analyst', 'business system analyst', inplace = True)

catdf['Designation'].replace('dotnet developer', '.net developer', inplace = True)

catdf['Designation'].replace('engineer- customer support', 'customer support engineer', inplace = True)

catdf['Designation'].replace('executive engg', 'executive engineer', inplace = True)

catdf['Designation'].replace('graduate trainee engineer', 'graduate engineer trainee', inplace = True)

catdf['Designation'].replace('executive hr', 'hr executive', inplace = True)

catdf['Designation'].replace('human resource assistant', 'hr assistant', inplace = True)

catdf['Designation'].replace('junior software developer', 'jr. software developer', inplace = True)

catdf['Designation'].replace('junior software engineer', 'jr. software engineer', inplace = True)

catdf['Designation'].replace('operation engineer', 'operations engineer', inplace = True)

catdf['Designation'].replace('operation executive', 'operations executive', inplace = True)

catdf['Designation'].replace('operational executive', 'operations executive', inplace = True)

catdf['Designation'].replace('qa analyst', 'quality assurance analyst', inplace = True)

catdf['Designation'].replace('qa engineer', 'quality assurance engineer', inplace = True)

catdf['Designation'].replace('sales & service engineer', 'sales and service engineer', inplace = True)

catdf['Designation'].replace('service and sales engineer', 'sales and service engineer', inplace = True)

catdf['Designation'].replace('software devloper', 'software developer', inplace = True)

catdf['Designation'].replace('software eng', 'software engineer', inplace = True)

catdf['Designation'].replace('software engg', 'software engineer', inplace = True)

catdf['Designation'].replace('software engineere', 'software engineer', inplace = True)

catdf['Designation'].replace('software enginner', 'software engineer', inplace = True)

catdf['Designation'].replace('software engineering associate', 'software engineer associate', inplace = True)

catdf['Designation'].replace('software test engineerte', 'software test engineer', inplace = True)

catdf['Designation'].replace('software trainee engineer', 'software engineer trainee', inplace = True)

catdf['Designation'].replace('trainee software engineer', 'software engineer trainee', inplace = True)

catdf['Designation'].replace('sr. engineer', 'senior engineer', inplace = True)

catdf['Designation'].replace('systems administrator', 'system administrator', inplace = True)

catdf['Designation'].replace('systems analyst', 'system analyst', inplace = True)

catdf['Designation'].replace('systems engineer', 'system engineer', inplace = True)

catdf['Designation'].replace('team leader', 'team lead', inplace = True)

catdf['Designation'].replace('telecommunication engineer', 'telecom engineer', inplace = True)

catdf['Designation'].replace('testing engineer', 'test engineer', inplace = True)

catdf['Designation'].replace('engineer trainee', 'trainee engineer', inplace = True)

#%%Replacing the bogus value with np.nan.

catdf['JobCity'].replace(-1,np.nan, inplace=True)

#%% JobCity Data Cleaning

catdf.JobCity.unique()

catdf.JobCity = catdf.JobCity.str.capitalize()

catdf.JobCity = catdf.JobCity.str.strip()

#z = pd.Series(catdf.JobCity.unique()).sort_values()

catdf['JobCity'].nunique()

catdf['JobCity']=catdf['JobCity'].replace({'haryana':'Haryana','KOTA':'Kota','manesar':'Manesar','MEERUT':'Meerut','Asifabadbanglore':'Bengaluru'})

catdf['JobCity']=catdf['JobCity'].replace({'karnal':'Karnal','Banglore ':'Bengaluru','manesar':'Manesar','Bangalore':'Bengaluru','Banglore':'Bengaluru'})

catdf['JobCity']=catdf['JobCity'].replace({'shahibabad':'Sahibabad','KANPUR':'Kanpur','pondy':'Puducherry','mohali':'Mohali','noida':'Noida','delhi':'Delhi','HYDERABAD':'Hyderabad','mysore':'Mysore','latur (Maharashtra )':'Latur'})

catdf['JobCity']=catdf['JobCity'].replace({'Ambala City':'Ambala','RAE BARELI':'Raebareli','jAipur':'Jaipur','sampla':'Sampla','NEW DELHI':'Delhi'})
# 232 unique values
catdf['JobCity'].nunique()

catdf['JobCity']=catdf['JobCity'].replace({'noida ':'Noida','ranchi':'Ranchi','PUNE':'Pune',' Pune':'Pune','orissa':'Odisha','kala amb ':'Kala Amb','chennai ':'Chennai'})

catdf['JobCity']=catdf['JobCity'].replace({'ghaziabad':'Ghaziabad','Panchkula ':'Panchkula','Mettur, Tamil Nadu ':'Mettur','Baddi HP':'Baddi','Pune ':'Pune','Greater NOIDA':'Greater Noida','hyderabad ':'Hyderabad','chandigarh':'Chandigarh','BHUBANESWAR':'Bhubaneswar','Navi mumbai':'Navi Mumbai','hyderabad(bhadurpally)':'Hyderabad','GREATER NOIDA':'Greater Noida'})

catdf['JobCity'].nunique()


catdf['JobCity']=catdf['JobCity'].replace({'Punr':'Pune','Latur (maharashtra )':'Latur',' delhi':'Delhi',' pune':'Pune','Sadulpur,rajgarh,distt-churu,rajasthan':'Sadulpur','Hyderabad(bhadurpally)':'Hyderabad'})


catdf['JobCity']=catdf['JobCity'].replace({'Kochi/cochin, chennai and coimbatore':'Chennai','New delhi - jaisalmer':'Delhi','Chennai, bangalore':'Chennai','Navi mumbai , hyderabad':'Hyderabad','A-64,sec-64,noida':'Noida','Rayagada, odisha':'Odisha','Orissa':'Orissa','pune':'Pune','bangalore':'Bengaluru','New delhi':'Delhi','Banaglore':'Bengaluru','Delhi/ncr':'Delhi'})

catdf['JobCity'].nunique()
#220 unique values

catdf['JobCity']=catdf['JobCity'].replace({'Mettur, tamil nadu':'Mettur','Hderabad':'Hyderabad','New dehli':'Delhi','Bellary':'Ballari'})

catdf['JobCity']=catdf['JobCity'].replace({'Technopark, trivandrum':'Thiruvananthapuram','Trivandrum':'Thiruvananthapuram','ariyalur':'Ariyalur','Kochi/cochin':'Kochi','Indirapuram, ghaziabad':'Ghaziabad','Gaziabaad':'Ghaziabad','Bhubneshwar':'Bhubaneswar'})

catdf['JobCity']=catdf['JobCity'].replace({'Banagalore':'Bengaluru','Bhubneshwar':'Bhubaneswar','Gajiabaad':'Ghaziabad','Kudankulam ,tarapur':'Koodankulam','Ncr':'Delhi','mumbai':'Mumbai'})

catdf['JobCity'].nunique()
#208 unique values

catdf['JobCity']=catdf['JobCity'].replace({'Guragaon':'Gurugram','Gurgaon':'Gurugram','Gurgoan':'Gurugram','Nouda':'Noida','chennai':'Chennai','Kolkata`':'Kolkata','Orissa':'Odisha','Guragaon':'Gurugram','Kalmar, sweden':'Sweden','Chennai & mumbai':'Chennai'})

catdf['JobCity'].nunique()
#201 unique

catdf['JobCity']=catdf['JobCity'].replace({'Al jubail,saudi arabia':'Saudi Arabia','Jeddah saudi arabia':'Saudi Arabia','Nasikcity':'Nashik'})


catdf['JobCity']=catdf['JobCity'].replace({'Bhubaneswar':'Bhubaneshwar','Shahibabad':'Sahibabad','Baddi hp':'Baddi','Burdwan':'Bardhaman'})

#197 unique values
catdf['JobCity'].nunique()

#461 np.nan - 11%
catdf['JobCity'].isnull().sum()/len(catdf)*100

catdf['JobCity'].unique()

catdf['JobCity']=catdf['JobCity'].replace({'Sonepat':'Sonipat','Gandhi nagar':'Gandhinagar','Muzzafarpur':'Muzaffarpur',})
#194 unique values

catdf['JobCity']=catdf['JobCity'].replace({'Vsakhapttnam':'Visakhapatnam','Vizag':'Visakhapatnam','Hospete':'Hosapete','Trichy':'Tiruchirappalli','Tirunelveli':'Tirunelvelli','Keral':'Kerala'})

catdf.JobCity.nunique()

metropolitian=['Bangalore','Bengaluru','Hyderabad','Pune','Ahmedabad','Surat','Jaipur','Lucknow','Kanpur','Nagpur','Indore','Thane','Bhopal','Agra','Nashik','Visakhapatnam','Patna','Kannur','Ludhiana','Rajkot','Chandigarh','Amritsar','Meerut','Thiruvananthapuram','Vadodara','Ranchi','Varanasi','Kota','Raipur','Madurai','Jodhpur','Vijayawada','Gwalior','Jabalpur','Coimbatore','Howrah','Allahabad','Dhanbad','Aurangabad','Srinagar','Vassai-Virar','Kalyan-Dombivali','Mumbai','Delhi','Chennai','Kolkata','New Delhi','Navi Mumbai','Ghaziabad','Faridabad','Gurugram']

len(metropolitian)

abroad=['Australia','Dubai','Saudi Arabia','Sweden','Johannesburg','London','Ras al khaimah','Dammam']

#%% JobCity Classification and feature Extraction.

def jc_Class(i):
    if i in metropolitian:
        return 'Metropolitian'
    elif i in abroad:
        return 'Abroad'
    elif type(i) == str:
        return 'Non-Metropolitian'
    else:
        return np.nan

catdf['City_Type'] = list(map(jc_Class, catdf['JobCity']))

#%% Specialization Column

#z = pd.Series(catdf['Specialization'].unique()).sort_values()

catdf['Specialization'].replace('electronics & instrumentation eng', 'electronics and instrumentation engineering', inplace = True)

#%% Encoding the Categorical features which are important for model building.

#Since all the features have different number of unique labels in them, different encoding techniques will be applied.

# Since there is no ordinality in some of the features we will go with Label Encoding.

col = ['Designation', 'Gender', 'Degree', 'Specialization', 'CollegeState']

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for i in col:
    catdf['Enc_'+i]= le.fit_transform(catdf[i])

# 12graduation and GraduationYear are irrelavant with respect to model building thus not encoding it.

#%% Concating the Numerical and Encoded Categorical features to get Final DataFrame for Clustering and 
#classification of Salary feature.

df_clust = pd.concat([numdf, catdf.iloc[:,9:]], axis = 1)

#%% Clustering of Data - Finding optimal value of k.

from sklearn.cluster import KMeans

wcv = []

for i in range(1, 11):
    km = KMeans(n_clusters = i)
    km.fit(df_clust)
    wcv.append(km.inertia_)
    
plt.plot(range(1, 11), wcv)
plt.xlabel('K (Number of Centroids)')
plt.ylabel('WCV (Within Cluster Variation)')
plt.title('Optimal Number of Clusters using K-Means')
plt.show()

#After k=4, the graph becomes linear. Thus, optimal number of clusters according to WCA is 4,5 and 6.

from sklearn.metrics import silhouette_score

silhouette = []

for i in range(2, 11):
    km = KMeans(n_clusters = i)
    km.fit(df_clust)
    silhouette.append(silhouette_score(df_clust, km.labels_))
    
plt.plot(range(2, 11), silhouette)
plt.xlabel('K (Number of Centroids)')
plt.ylabel('Silhouette Score')
plt.title('Optimal Number of Clusters using K-Means')
plt.show()

ss_score = pd.DataFrame()
ss_score['K-Value'] = range(2,11)
ss_score['Silhouette_Score'] = silhouette
ss_score.sort_values(by='Silhouette_Score', ascending=False, inplace=True)

#%% Making clusters based on wcv graph and ss_score. The optimal value of k is 5.

km = KMeans(n_clusters = 4)
km.fit(df_clust)
df_clust['Cluster'] = km.predict(df_clust)

#%% Analysing the clusters

clust0 = df_clust[df_clust['Cluster'] == 0]
clust1 = df_clust[df_clust['Cluster'] == 1]
clust2 = df_clust[df_clust['Cluster'] == 2]
clust3 = df_clust[df_clust['Cluster'] == 3]

# After analysing the clusters we found that the clusters are made based on Salary feature.

#%% Checking and understanding the range of values of Salary for each clusters. Then finding the order of the clusters
#according to the Salary bracket in order to prepare dataset for Classification.

check_clust = []

for i in range(0,4):
    check_clust.append((i, df_clust[df_clust['Cluster'] == i].iloc[:,0]
    .min(), df_clust[df_clust['Cluster'] == i].iloc[:,0].max()))

#%% Salary Classification

#sns.distplot(df['Salary'])
#plt.show()

def sal_Class(i):
    if i>=35000 and i<=260000:
        return'Low'
    elif i>=265000 and i<=485000:
        return 'Good'
    elif i>=490000 and i<=1500000:
        return 'Average'
    elif i>=1745000 and i<=4000000:
        return 'High'
        
df_clust['Salary_Class'] = list(map(sal_Class, df['Salary'])) #Got new feature Salary_Class

#%% Making separate dataset for missing value imputation in City_Type

df_imp_city = df_clust.iloc[:,1:-2]

#%% Encoding required columns and preparing df_imp_city to apply imputation algorithm.

#Encoding Salary_Class and concating the ecoded column in df_imp_city

from sklearn.preprocessing import OrdinalEncoder

ordenc = OrdinalEncoder(categories=[['Low', 'Good', 'Average', 'High']])

df_imp_city['enc_Salary_Class'] = ordenc.fit_transform(df_clust.Salary_Class.values.reshape(-1,1))

# Encoding City_Type, since there is ordinality in City_Types, we should go with Ordinal Encoding but there 
# are nan values too, so, we have to encode manually.

def jc_Class_Enc(i):
    if i == 'Non-Metropolitian':
        return 0
    elif i == 'Metropolitian':
        return 1
    elif i == 'Abroad':
        return 2
    
catdf['enc_City_Type'] = list(map(jc_Class_Enc, catdf['City_Type']))

catdf.drop('JobCity', axis=1, inplace=True)

#%% Cncating enc_City_Type from catdf to impute missing values in it.

df_imp_city = pd.concat([df_imp_city, catdf.enc_City_Type], axis = 1)

#%%Imputing missing values in enc_City_Type with fancyimpute

imp_columns = df_imp_city.columns

from fancyimpute import KNN

df_imp_city = KNN(k=3).fit_transform(df_imp_city)

df_imp_city = pd.DataFrame(df_imp_city, columns = imp_columns)

#%%Checking for null values after imputation

df_imp_city.isnull().sum()

#%%Operating on values to make it perfect 0,1 or 2 instead of decimal.

for i in df_imp_city.index:
    if df_imp_city.loc[i, 'enc_City_Type']<=0.5:
        df_imp_city.loc[i, 'enc_City_Type'] = 0
    elif df_imp_city.loc[i, 'enc_City_Type']>0.5 and df_imp_city.loc[i, 'enc_City_Type']<= 1.0:
        df_imp_city.loc[i, 'enc_City_Type'] = 1
    elif df_imp_city.loc[i, 'enc_City_Type']>1:
        df_imp_city.loc[i, 'enc_City_Type'] = 2

#%% Replacing encCityType in catdf with new imputed encCitytype feature from df_imp_city. Also, imputing value in non-encoded City_Type feature for future purposes.

catdf.enc_City_Type = df_imp_city.loc[:, 'enc_City_Type']

#Now imputing values in City_Type based on enc_City_Type.

def city_Class_Impute(i):
    if i == 1:
        return 'Metropolitian'
    elif i == 2:
        return 'Abroad'
    elif i == 0:
        return 'Non-Metropolitian'
    
catdf['City_Type'] = list(map(city_Class_Impute, catdf['enc_City_Type']))

#%% Final df to work on model building

df_final = pd.concat([numdf.iloc[:,1:], catdf.iloc[:, 8:], df_clust.loc[:,'Salary_Class']], axis = 1)



