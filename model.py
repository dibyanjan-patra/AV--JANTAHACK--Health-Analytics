# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 20:58:53 2020

@author: dibya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("D:\\Data Science\\Projects\\AV- Health Analytics\\Train.csv")
test = pd.read_csv("D:\\Data Science\\Projects\\AV- Health Analytics\\test_l0Auv8Q.csv")
submit = pd.read_csv("D:\\Data Science\\Projects\\AV- Health Analytics\\sample_submmission.csv")

health_camp = pd.read_csv("D:\\Data Science\\Projects\\AV- Health Analytics\\Health_Camp_Detail.csv")
pp = pd.read_csv("D:\\Data Science\\Projects\\AV- Health Analytics\\Patient_Profile.csv")
first_health_camp = pd.read_csv("D:\\Data Science\\Projects\\AV- Health Analytics\\First_Health_Camp_Attended.csv")
second_health_camp = pd.read_csv("D:\\Data Science\\Projects\\AV- Health Analytics\\Second_Health_Camp_Attended.csv")
third_health_camp = pd.read_csv("D:\\Data Science\\Projects\\AV- Health Analytics\\Third_Health_Camp_Attended.csv")

pp['online_Activity_Score']=pp['Online_Follower']+pp['LinkedIn_Shared']+pp['Twitter_Shared']+pp['Facebook_Shared']
del pp['Online_Follower']
del pp['LinkedIn_Shared']
del pp['Twitter_Shared']
del pp['Facebook_Shared']
"""
col_names = [['Patient_ID','Health_Camp_ID','Outcome']]
first_camp = first_health_camp[['Patient_ID','Health_Camp_ID','Health_Score']]
first_camp.columns = col_names
second_camp = second_health_camp[['Patient_ID','Health_Camp_ID','Health Score']]
second_camp.columns = col_names
third_camp = third_health_camp[['Patient_ID','Health_Camp_ID','Number_of_stall_visited']]
third_camp = third_camp[third_camp['Number_of_stall_visited']>0]
third_camp.columns = col_names
print (third_camp.shape)

all_camps = pd.concat([first_camp, second_camp, third_camp])
all_camps['Outcome'].isnull().sum()
all_camps['Outcome'] = 1
print (all_camps.shape)"""

#Pre-process data for patient profile
pp.info()
pp.isnull().sum()
#replacing none value with nan
pp[['Income', 'Education_Score', 'Age']] = pp[['Income', 'Education_Score', 'Age']].apply(lambda x: x.str.replace('None', 'NaN').astype('float'))

# label-encoding categorical value
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in ['City_Type','Employer_Category']:
    pp[col]=  pp[col].astype('str')
    pp[col]= le.fit_transform(pp[col])

pp['first_interaction_year'] = pd.to_datetime(pp['First_Interaction']).dt.year
pp['first_interaction_month'] = pd.to_datetime(pp['First_Interaction']).dt.month
pp['first_interaction_day'] = pd.to_datetime(pp['First_Interaction']).dt.day

#feature engg of train data
train['registration_year'] = pd.to_datetime(train['Registration_Date']).dt.year
train['registration_month'] = pd.to_datetime(train['Registration_Date']).dt.month
train['registration_day'] = pd.to_datetime(train['Registration_Date']).dt.day

train['Registration_Date']=pd.to_datetime(train['Registration_Date'])
train.sort_values(by=['Patient_ID','Registration_Date'],inplace=True)
train.info()

train['days_since_last_registration'] = train.groupby('Patient_ID')['Registration_Date'].diff().apply(lambda x: x.days)
train['days_since_next_registration'] = train.groupby('Patient_ID')['Registration_Date'].diff(-1) * (-1) / np.timedelta64(1, 'D')       
train.reset_index(drop=True,inplace=True)

train['Unique_Health_Camp_per_patient']=train.groupby(['Patient_ID'])['Health_Camp_ID'].transform('nunique')
train['Unique_Patient_per_HealthCamp']=train.groupby(['Health_Camp_ID'])['Patient_ID'].transform('nunique')
train['Unique_year_per_patient']=train.groupby(['Patient_ID'])['registration_year'].transform('nunique')
train['Unique_months_per_patient']=train.groupby(['Patient_ID'])['registration_month'].transform('nunique')
train['Unique_day_per_patient']=train.groupby(['Patient_ID'])['registration_day'].transform('nunique')

#engg for test data
test['registration_month'] = pd.to_datetime(test['Registration_Date']).dt.month
test['registration_day'] = pd.to_datetime(test['Registration_Date']).dt.day
test['registration_year'] = pd.to_datetime(test['Registration_Date']).dt.year

test['Registration_Date']=pd.to_datetime(test['Registration_Date'])
test.sort_values(by=['Patient_ID','Registration_Date'],inplace=True)

test['days_since_last_registration'] = test.groupby('Patient_ID')['Registration_Date'].diff().apply(lambda x: x.days)
test['days_since_next_registration'] = test.groupby('Patient_ID')['Registration_Date'].diff(-1) * (-1) / np.timedelta64(1, 'D')
test.reset_index(drop=True,inplace=True)

test['Unique_Health_Camp_per_patient']=test.groupby(['Patient_ID'])['Health_Camp_ID'].transform('nunique')
test['Unique_Patient_per_HealthCamp']=test.groupby(['Health_Camp_ID'])['Patient_ID'].transform('nunique')
test['Unique_year_per_patient']=test.groupby(['Patient_ID'])['registration_year'].transform('nunique')
test['Unique_months_per_patient']=test.groupby(['Patient_ID'])['registration_month'].transform('nunique')
test['Unique_day_per_patient']=test.groupby(['Patient_ID'])['registration_day'].transform('nunique')

#find overlap between train and test sets

cols =  ['Patient_ID']
for col in cols:
  print('Total unique'+col  +' values in Train are {}'.format(train[col].nunique()))
  print('Total unique'+col  +' values in Test are {}'.format(test[col].nunique()))
  print('Common'+col +' values are {}'.format(len(list(set(train[col]) & set(test[col])))))

#merging two dataset
train = pd.merge(train, pp, on = 'Patient_ID', how = 'left')
test = pd.merge(test, pp, on = 'Patient_ID', how = 'left')

#making outcome value
for c in [first_health_camp, second_health_camp, third_health_camp, train]:
  c['id'] = c['Patient_ID'].astype('str') + c['Health_Camp_ID'].astype('str')
third_health_camp = third_health_camp[third_health_camp['Number_of_stall_visited'] > 0]

all_patients_in_camp = pd.Series(first_health_camp['id'].tolist() + second_health_camp['id'].tolist() + third_health_camp['id'].tolist()).unique()

train['target'] = 0
train.loc[train['id'].isin(all_patients_in_camp), 'target'] = 1

#fe for health camp
def timediff(duration):
    duration_in_s = duration.total_seconds()
    days = divmod(duration_in_s, 86400)[0]
    return days

health_camp['Camp_Duration']=pd.to_datetime(health_camp['Camp_End_Date'])-pd.to_datetime(health_camp['Camp_Start_Date'])
health_camp['Camp_Duration']=health_camp['Camp_Duration'].apply(timediff)

health_camp['camp_start_year'] = pd.to_datetime(health_camp['Camp_Start_Date']).dt.year
health_camp['camp_start_month'] = pd.to_datetime(health_camp['Camp_Start_Date']).dt.month
health_camp['camp_start_day'] = pd.to_datetime(health_camp['Camp_Start_Date']).dt.day

health_camp['camp_end_year'] = pd.to_datetime(health_camp['Camp_End_Date']).dt.year
health_camp['camp_end_month'] = pd.to_datetime(health_camp['Camp_End_Date']).dt.month
health_camp['camp_end_day'] = pd.to_datetime(health_camp['Camp_End_Date']).dt.day

health_camp['Category1'] = health_camp['Category1'].map({'First': 1, 'Second': 2, 'Third': 3})
health_camp['Category2'] = pd.factorize(health_camp['Category2'])[0]

#merging table train with health_camp
train = pd.merge(train, health_camp, on = 'Health_Camp_ID', how = 'left')
test = pd.merge(test, health_camp, on = 'Health_Camp_ID', how = 'left')

train['Unique_camp_year_per_patient']=train.groupby(['Patient_ID'])['camp_start_year'].transform('nunique')
train['Unique_camp_months_per_patient']=train.groupby(['Patient_ID'])['camp_start_month'].transform('nunique')
train['Unique_camp_day_per_patient']=train.groupby(['Patient_ID'])['camp_start_day'].transform('nunique')

test['Unique_camp_year_per_patient']=test.groupby(['Patient_ID'])['camp_start_year'].transform('nunique')
test['Unique_camp_months_per_patient']=test.groupby(['Patient_ID'])['camp_start_month'].transform('nunique')
test['Unique_camp_day_per_patient']=test.groupby(['Patient_ID'])['camp_start_day'].transform('nunique')

train['train_or_test']='train'
test['train_or_test']='test'
df=pd.concat([train,test])

def agg_numeric(df, parent_var, df_name):
    
    # Only want the numeric variables
    parent_ids = df[parent_var].copy()
    numeric_df = df.select_dtypes('number').drop(columns={'Patient_ID', 'Health_Camp_ID','target'}).copy()
    numeric_df[parent_var] = parent_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(parent_var).agg(['count', 'mean', 'max', 'min', 'sum'])

    # Need to create new column names
    columns = []

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        if var != parent_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))
    
    agg.columns = columns
    
    # Remove the columns with all redundant values
    _, idx = np.unique(agg, axis = 1, return_index=True)
    agg = agg.iloc[:, idx]
    
    return agg

PID_aggregate = agg_numeric(df, 'Patient_ID', 'agg')
print('PID aggregate shape: ', PID_aggregate.shape)
df=df.merge(PID_aggregate, on ='Patient_ID', how = 'left')

df['Patient_Registered_before_days']=pd.to_datetime(df['Camp_Start_Date'])-pd.to_datetime(df['Registration_Date'])
df['Patient_Registered_before_days']=df['Patient_Registered_before_days'].apply(timediff)

train=df.loc[df.train_or_test.isin(['train'])]
test=df.loc[df.train_or_test.isin(['test'])]
train.drop(columns={'train_or_test'},axis=1,inplace=True)
test.drop(columns={'train_or_test'},axis=1,inplace=True)

trn=train[train['Camp_Start_Date'] <'2005-11-01']
val=train[train['Camp_Start_Date'] >'2005-10-30']

TARGET_COL = 'target'
features = [c for c in trn.columns if c not in ['Patient_ID', 'Health_Camp_ID','Category3','Registration_Date', 'id','target','Camp_Start_Date','Camp_End_Date','First_Interaction',TARGET_COL]]
len(features)

from lightgbm import LGBMClassifier
clf = LGBMClassifier(n_estimators=550,
                     learning_rate=0.03,
                     min_child_samples=40,
                     random_state=1,
                     colsample_bytree=0.5,
                     reg_alpha=2,
                     reg_lambda=2)

clf.fit(trn[features], trn[TARGET_COL], eval_set=[(val[features], val[TARGET_COL])], verbose=50,
        eval_metric = 'auc', early_stopping_rounds = 100)

preds = clf.predict_proba(test[features])[:, 1]

fi = pd.Series(index = features, data = clf.feature_importances_)
fi.sort_values(ascending=False)[0:20][::-1].plot(kind = 'barh')

sub = pd.DataFrame({"Patient_ID":test.Patient_ID.values})
sub["Health_Camp_ID"] = test.Health_Camp_ID.values
sub["Outcome"] =  preds
sub.to_csv("lgbmblending.csv", index=False)
