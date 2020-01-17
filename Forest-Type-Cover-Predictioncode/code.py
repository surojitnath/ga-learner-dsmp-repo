# --------------
import pandas as pd
from sklearn import preprocessing

#path : File path

# Code starts here


# read the dataset

dataset = pd.read_csv(path)

# look at the first five columns
print(dataset.head(5))

# Check if there's any column which is not useful and remove it like the column id
dataset.drop(['Id'],1,inplace=True)
# check the statistical description



# --------------
# We will visualize all the attributes using Violin Plot - a combination of box and density plots
import seaborn as sns
from matplotlib import pyplot as plt

#names of all the attributes 
cols=dataset.columns

#number of attributes (exclude target)
size=len(cols)-1


#x-axis has target attribute to distinguish between classes
x=dataset['Cover_Type']


#y-axis shows values of an attribute
y=dataset.drop(['Cover_Type'],axis=1)


#Plot violin for all attributes
for col in cols:
    sns.violinplot(x=dataset[col])



# --------------
import numpy
upper_threshold = 0.5
lower_threshold = -0.5


import numpy
import seaborn as sns
upper_threshold = 0.5
lower_threshold = -0.5


# Code Starts Here
subset_train=dataset.iloc[:,:10]
data_corr=subset_train.corr()
sns.heatmap(data_corr)
correlation=data_corr.unstack().sort_values(kind='quicksort')
corr_var_list=[]
for i in correlation:
    if i<lower_threshold:
        corr_var_list.append(i)
    if (i>upper_threshold) & (i !=1):
        corr_var_list.append(i)
print(corr_var_list)



# --------------
#Import libraries 
from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Identify the unnecessary columns and remove it 
dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)



# Scales are not the same for all variables. Hence, rescaling and standardization may be necessary for some algorithm to be applied on it.



#Standardized
#Apply transform only for continuous data


#Concatenate scaled continuous data and categorical
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=0)


num_feat_cols = ['Elevation', 'Aspect', 'Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon', 'Hillshade_3pm','Horizontal_Distance_To_Fire_Points']

cat_feat_cols = list(set(X_train.columns) - set(num_feat_cols))
scaler=StandardScaler()

X_train_temp=X_train[num_feat_cols].copy()

X_test_temp=X_test[num_feat_cols].copy()

X_train_temp[num_feat_cols]=scaler.fit_transform(X_train_temp[num_feat_cols])

X_test_temp[num_feat_cols]=scaler.fit_transform(X_test_temp[num_feat_cols])


X_train1=pd.concat([X_train_temp,X_train.loc[:,cat_feat_cols]],axis=1)

print(X_train1.head())

X_test1=pd.concat([X_test_temp,X_test.loc[:,cat_feat_cols]],axis=1)

print(X_test1.head())

scaled_features_train_df=X_train1
#Standardized
#Apply transform only for non-categorical data
scaled_features_test_df=X_test1



# --------------
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif


# Write your solution here:

skb=SelectPercentile(score_func=f_classif,percentile=90)
predictors=skb.fit_transform(X_train1,Y_train)
scores=skb.scores_
Features=X_train.columns
dataframe=pd.DataFrame({'Features':Features,'scores':scores})
dataframe.sort_values(by=['scores'],ascending=False,inplace=True)
top_k_index  = skb.get_support(indices=True)
top_k_predictors = list(dataframe['Features'][:predictors.shape[1]])
print(top_k_predictors)


# --------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score


clf = OneVsRestClassifier(LogisticRegression())
clf1=OneVsRestClassifier(LogisticRegression())
model_fit_all_features =clf.fit(X_train , Y_train)
predictions_all_features=clf.predict(X_test)

score_all_features= accuracy_score(Y_test,predictions_all_features )

print(scaled_features_train_df.columns[skb.get_support()])

X_new = scaled_features_train_df.loc[:,skb.get_support()]
X_test_new=scaled_features_test_df.loc[:,skb.get_support()]

model_fit_top_features  =clf1.fit(X_new , Y_train)
predictions_top_features=clf1.predict(X_test_new)

score_top_features= accuracy_score(Y_test,predictions_top_features )


