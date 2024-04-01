
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

train_data = pd.read_excel(r'E:\唐筛\train.csv')
test_data = pd.read_csv(r'E:\唐筛\test.csv')

diag_1_data = train_data[train_data['诊断'] == 1]
diag_1_data = diag_1_data.sample(n=1000, random_state=42)
diag_2_data = train_data[train_data['诊断'] == 2]
merge_data = pd.concat([diag_1_data,diag_2_data], ignore_index=True)
merge_data = merge_data.sample(frac=1, random_state=42).reset_index(drop=True)

scaler = MinMaxScaler()

merge_data[['AFPMOM','BHCGMOM','UE3MOM']] = scaler.fit_transform(merge_data[['AFPMOM',
                                                                'BHCGMOM','UE3MOM']])
test_data[['AFPMOM','BHCGMOM','UE3MOM']] = scaler.transform(test_data[['AFPMOM',
                                                                'BHCGMOM','UE3MOM']])
merge_data[['体重','年龄']] = scaler.fit_transform(merge_data[['体重','年龄']])
test_data[['体重','年龄']] = scaler.transform(test_data[['体重','年龄']])
                                                            
X_train = merge_data.iloc[:, [1, 2, 4, 5, 6]].values
y_train = merge_data.iloc[:, 0]
X_test = test_data.iloc[:, [1, 2, 3, 4, 5]].values
y_test = test_data.iloc[:, -1]

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,random_state=21)
clf.fit(X_train, y_train)
train_proba = clf.predict_proba(X_train)
train_proba_df = pd.DataFrame({'Label':y_train,
                               'Probability of Class 1': train_proba[:, 0], 
                               'Probability of Class 2': train_proba[:, 1]})

sns.set_style(style='whitegrid')
plt.figure(figsize=(10, 6))
sns.kdeplot(train_proba_df[train_proba_df['Label'] == 1]['Probability of Class 1'],
            color='blue', shade=True, label='Class 1 - True 1')
sns.kdeplot(train_proba_df[train_proba_df['Label'] == 2]['Probability of Class 1'],
            color='red', shade=True, label='Class 1 - True 2')
sns.kdeplot(train_proba_df[train_proba_df['Label'] == 1]['Probability of Class 2'],
            color='green', shade=True, label='Class 2 - True 1')
sns.kdeplot(train_proba_df[train_proba_df['Label'] == 2]['Probability of Class 2'],
            color='yellow', shade=True, label='Class 2 - True 2')

plt.title('Probability Distribution of Class 1 and Class 2 on Training Data')
plt.xlabel('Probability')
plt.ylabel('Density')
plt.legend(title='True Label')
plt.show()

test_proba = clf.predict_proba(X_test)
test_proba_df = pd.DataFrame({'Label':y_test,
                               'Probability of Class 1': test_proba[:, 0], 
                               'Probability of Class 2': test_proba[:, 1]})

sns.set_style(style='whitegrid')
plt.figure(figsize=(10, 6))
sns.kdeplot(test_proba_df[test_proba_df['Label'] == 1]['Probability of Class 1'],
            color='blue', shade=True, label='Class 1 - True 1')
sns.kdeplot(test_proba_df[test_proba_df['Label'] == 2]['Probability of Class 1'],
            color='red', shade=True, label='Class 1 - True 2')
sns.kdeplot(test_proba_df[test_proba_df['Label'] == 1]['Probability of Class 2'],
            color='green', shade=True, label='Class 2 - True 1')
sns.kdeplot(test_proba_df[test_proba_df['Label'] == 2]['Probability of Class 2'],
            color='yellow', shade=True, label='Class 2 - True 2')

plt.title('Probability Distribution of Class 1 and Class 2 on Test Data')
plt.xlabel('Probability')
plt.ylabel('Density')
plt.legend(title='True Label')
plt.show()