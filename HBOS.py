
import pandas as pd
import numpy as np
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score
import plotly.express as px
import matplotlib.pyplot as plt

train_data = pd.read_excel(r'E:\唐筛\train.csv')
test_data = pd.read_csv(r'E:\唐筛\test.csv')

diag_1_data = train_data[train_data['诊断'] == 1]
diag_1_data = diag_1_data.sample(n=6000, random_state=42)
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

#clf = ECOD(contamination=0.45,n_jobs=-1)
clf = HBOS(n_bins=50, contamination=0.45)

clf.fit(X_train_1)
scores_1 = clf.decision_function(X_train_1)
pred_1 = clf.predict(X_train_1)

def count_stat(vector):
    # Because it is '0' and '1', we can run a count statistic. 
    unique, counts = np.unique(vector, return_counts=True)
    return dict(zip(unique, counts))

print("The training data:", count_stat(pred_1))

plt.hist(scores_1,bins='auto') # arguments are passed to np.histogram
plt.title("Outlier score")
plt.show()

df1 = X_train_1.copy()
df1['label'] = pred_1
df2 = X_train_2.copy()
df2['label'] = 0
df = pd.concat([df1,df2], ignore_index=True)

fig = px.scatter_3d(df,x='AFPMOM', 
                       y='BHCGMOM', 
                       z='UE3MOM', 
                       color='label') 


fig.show()

y_train = pd.concat([y_train_1,y_train_2],ignore_index=True)
y_train[y_train == 1] = 0
y_train[y_train == 2] = 1

from sklearn.metrics import recall_score

y_pred = df['label'].values
y_true = y_train

recall_negitive = recall_score(y_true, y_pred, pos_label=1)
recall_postivie = recall_score(y_true, y_pred, pos_label=0)
print("唐氏召回率:", recall_negitive)
print("正常召回率:", recall_postivie)