# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Code starts here

data = pd.read_csv(path)
data['Rating'].hist()
data = data[data['Rating']<=5]
data[data['Rating']<=5].hist()
#Code ends here


# --------------
# code starts here
total_null = data.isnull().sum()
print(total_null)
percent_null = (total_null/data.isnull().count())
print(percent_null)
missing_data =pd.concat([total_null, percent_null], keys=['Total', 'Percent'],axis = 1)
data.dropna(inplace=True)
total_null_1 = data.isnull().sum()
percent_null_1 = total_null_1/data.isnull().count()
missing_data_1 = pd.concat([total_null_1, percent_null_1], keys=['Total', 'Percent'],axis = 1)
# code ends here


# --------------

#Code starts here

sns.catplot(x="Category", y="Rating", data=data,kind="box",height = 10)
import matplotlib.pyplot as plt
plt.xticks(rotation = 90)


#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here

data["Installs"]= data["Installs"].str.replace('+' , " ") 
data["Installs"]= data["Installs"].str.replace(',' , "")
data["Installs"]= data["Installs"].astype(int)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Installs'] = le.fit_transform(data['Installs'])
data['Installs']
sns.regplot(x="Installs",y="Rating",data = data)
plt.title('Rating vs Installs [RegPlot]')

#Code ends here



# --------------
#Code starts here
data['Price'] = data['Price'].str.replace('$','')
data['Price'].value_counts()
data["Price"]= data["Price"].astype(float)

sns.regplot(x="Price",y="Rating",data = data)
plt.title('Rating vs Price [RegPlot]')

#Code ends here


# --------------

#Code starts here
data['Genres'].value_counts()
data['Genres'].unique()
data["Genres"]  = data["Genres"].str.split(';',expand = True)
gr_mean = data[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean()
gr_mean.describe()
gr_mean = gr_mean.sort_values(by='Rating',ascending=True)
print("-----2----",gr_mean.iloc[14])
print("-----3----",gr_mean.iloc[18])

#Code ends here


# --------------

#Code starts here
data['Last Updated'] = pd.to_datetime(data['Last Updated'])
max_date = data['Last Updated'].max()
data['Last Updated Days']=(max_date-data['Last Updated']).dt.days
sns.regplot(x="Last Updated Days",y="Rating",data = data)
plt.title('Rating vs Price [RegPlot]')
#var = data.groupby('Genres',as_index=False)[['Rating']].mean()
#var.unstack().plot(kind='bar',  color=['red','blue'], grid=False)
#Code ends here


