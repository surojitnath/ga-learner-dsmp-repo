# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(path)
#path of the data file- path
data['Gender'].replace("-","Agender",inplace=True)
gender_count=data['Gender'].value_counts()
gender_count.plot(kind='bar')
#Code starts here 




# --------------
#Code starts here
alignment=data['Alignment'].value_counts()
plt.pie(alignment)


# --------------
#Code starts here
sc_df= data[['Strength','Combat']]
sc_covariance = sc_df.cov().iloc[0,1]
sc_covariance
sc_strength = data['Strength'].std()
sc_combat=data['Combat'].std()
sc_pearson = sc_covariance/(sc_strength*sc_combat)
print("Pearson's Correlation Coefficient : ", sc_pearson)
ic_df= data[['Intelligence','Combat']]
ic_covariance = ic_df.cov().iloc[0,1]
ic_intelligence = data['Intelligence'].std()
ic_combat=data['Combat'].std()
ic_pearson = ic_covariance/(ic_intelligence*ic_combat)
print("Pearson's Correlation Coefficient : ", ic_pearson)


# --------------
#Code starts here
total_high = data['Total'].quantile(0.99)
super_best= data[data['Total']>total_high]
super_best_names =list(super_best['Name'])


# --------------
#Code starts here
fig,(ax_1,ax_2,ax_3) = plt.subplots(3,1,figsize=(10,10))
ax_1.boxplot(data['Intelligence'])
ax_1.set_title('Intelligence')
ax_2.boxplot(data['Speed'])
ax_2.set_title('Speed')
ax_3.boxplot(data['Power'])
ax_3.set_title('Power')


