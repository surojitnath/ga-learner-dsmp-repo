# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df = pd.read_csv(path)
Count_700=len(df[df['fico']>700])
p_a=Count_700/len(df['fico'])
debt_con=len(df[df['purpose']=='debt_consolidation'])
p_b=debt_con/len(df['fico'])
df1=df[df['purpose']=='debt_consolidation']
p_a_b=(p_a*p_b)/p_a
if(p_a_b==p_a):
    result = True
else:
    result = False    
# code ends here


# --------------
# code starts here

paid_back=len(df[df['paid.back.loan']=='Yes'])
prob_lp =paid_back/len(df['paid.back.loan'])
prob_cs=len(df[df['credit.policy']=='Yes'])/len(df['paid.back.loan'])
new_df =df[df['paid.back.loan']=='Yes']
prob_pd_cs = new_df[new_df['credit.policy'] == 'Yes'].shape[0] / new_df.shape[0]
bayes =(prob_pd_cs*prob_lp)/prob_cs


# code ends here


# --------------
# code starts here
purpose=df['purpose'].value_counts()
purpose.plot(kind='bar')
df1=df[df['paid.back.loan']=='No']
purpose1=df1['purpose'].value_counts()
purpose1.plot(kind='bar')
# code ends here


# --------------
# code starts here

inst_median= df['installment'].median()
inst_mean=df['installment'].mean()
df['installment'].hist()
df['log.annual.inc'].hist()
# code ends here


