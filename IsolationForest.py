
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px

train_data = pd.read_excel(r'E:\唐筛\train.csv')
test_data = pd.read_csv(r'E:\唐筛\test.csv')

diag_1_data = train_data[train_data['诊断'] == 1]
diag_1_data = diag_1_data.sample(n=4000, random_state=42)
diag_2_data = train_data[train_data['诊断'] == 2]
merge_data = pd.concat([diag_1_data,diag_2_data], ignore_index=True)
merge_data = merge_data.sample(frac=1, random_state=42).reset_index(drop=True)

scaler = MinMaxScaler()

merge_data[['AFPMOM','BHCGMOM','UE3MOM']] = scaler.fit_transform(merge_data[['AFPMOM',
                                                                'BHCGMOM','UE3MOM']])
test_data[['AFPMOM','BHCGMOM','UE3MOM']] = scaler.fit_transform(test_data[['AFPMOM',
                                                                'BHCGMOM','UE3MOM']])

merge_data_1 = merge_data[merge_data['AFPMOM'] < 0.3100]
merge_data_2 = merge_data[merge_data['AFPMOM'] >= 0.3100]

X_train_1 = merge_data_1.drop(columns=['诊断','风险值','BHCGMOM/AFPMOM+UE3MOM'])
y_train_1 = merge_data_1['诊断']
X_train_2 = merge_data_2.drop(columns=['诊断','风险值','BHCGMOM/AFPMOM+UE3MOM'])
y_train_2 = merge_data_2['诊断']

clf = IsolationForest(n_estimators=500, contamination=0.45, random_state=42, n_jobs=-1)

df1 = X_train_1.copy()
df1['label'] = clf.fit_predict(X_train_1)
df1['scores'] = clf.decision_function(X_train_1)
df1['anomaly'] = df1['label'].apply(lambda x: 'outlier' if x==-1  else 'inlier') 
df1 = df1.drop(columns='scores')

df2 = X_train_2.copy()
df2['label'] = 1
df2['anomaly'] = df2['label'].apply(lambda x: 'outlier' if x==-1  else 'inlier')

df = pd.concat([df1,df2], ignore_index=True)

fig = px.scatter_3d(df,x='AFPMOM', 
                       y='BHCGMOM', 
                       z='UE3MOM', 
                       color='anomaly') 


fig.show()

fig1 = px.scatter_3d(merge_data,x='AFPMOM', 
                       y='BHCGMOM', 
                       z='UE3MOM', 
                       color='诊断') 


fig1.show()

y_pred = df['label'].values
y_train = pd.concat([y_train_1,y_train_2],ignore_index=True)
y_train[y_train == 1] = 1
y_train[y_train == 2] = -1

y_train.value_counts()
from sklearn.metrics import recall_score

y_pred = df['label'].values
y_true = y_train

recall_negitive = recall_score(y_true, y_pred, pos_label=-1)
recall_postivie = recall_score(y_true, y_pred)
print("唐氏召回率:", recall_negitive)
print("正常召回率:", recall_postivie)

X_test = test_data.drop(columns=['诊断','姓名'])
y_test = test_data['诊断']
df_test = X_test.copy()
df_test['scores'] = clf.decision_function(X_test)
df_test['label'] = clf.predict(X_test)
df_test['anomaly'] = df_test['label'].apply(lambda x: 'outlier' if x==-1  
                                            else 'inlier')

df_test.loc[(df_test['AFPMOM'] >= 0.3100), 'anomaly'] = 'inlier'
df_test.loc[(df_test['AFPMOM'] >= 0.3100), 'label'] = 1


fig = px.scatter_3d(df_test,x='AFPMOM', 
                       y='BHCGMOM', 
                       z='UE3MOM', 
                       color='anomaly') 


fig.show()

fig1 = px.scatter_3d(test_data,x='AFPMOM', 
                       y='BHCGMOM', 
                       z='UE3MOM', 
                       color='诊断') 


fig1.show()

y_pred = df_test['label'].values
y_test[y_test == 1] = 1
y_test[y_test == 2] = -1
y_true = y_test

recall_negitive = recall_score(y_true, y_pred, pos_label=-1)
recall_postivie = recall_score(y_true, y_pred)
print("唐氏召回率:", recall_negitive)
print("正常召回率:", recall_postivie)



'''
使用无监督学习方法:4000正常+80唐氏 训练
- 唐氏召回率: 0.9375
- 正常召回率: 0.58075
    
评估:使用老数据, 5000正常+50唐氏 测试
- 唐氏召回率: 0.8163
- 正常召回率: 0.1434
'''