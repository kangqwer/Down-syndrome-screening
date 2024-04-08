
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


train_data = pd.read_excel(r'E:\唐筛\train.csv')
test_data = pd.read_csv(r'E:\唐筛\test.csv')

diag_1_data = train_data[train_data['诊断'] == 1]
diag_1_data = diag_1_data.sample(n=1000, random_state=42)
diag_2_data = train_data[train_data['诊断'] == 2]
merge_data = pd.concat([diag_1_data,diag_2_data], ignore_index=True)
merge_data = merge_data.sample(frac=1, random_state=42).reset_index(drop=True)

scaler = MinMaxScaler()
'''

merge_data[['AFPMOM','BHCGMOM','UE3MOM']] = scaler.fit_transform(merge_data[['AFPMOM',
                                                                'BHCGMOM','UE3MOM']])
test_data[['AFPMOM','BHCGMOM','UE3MOM']] = scaler.transform(test_data[['AFPMOM',
                                                                'BHCGMOM','UE3MOM']])
'''

merge_data[['年龄','体重']] = scaler.fit_transform(merge_data[['年龄','体重']])
test_data[['年龄','体重']] = scaler.transform(test_data[['年龄','体重']])                                                            
X_train = merge_data.iloc[:, [1, 2, 4, 5, 6]].values
y_train = merge_data.iloc[:, 0]
X_test = test_data.iloc[:, [1, 2, 3, 4, 5]].values
y_test = test_data.iloc[:, -1]

clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

train_proba = clf.predict_proba(X_train)
train_auc = roc_auc_score(y_train, train_proba[:, 1])
print('训练集AUC:', train_auc)
fpr, tpr, thresholds = roc_curve(y_train, train_proba[:, 1], pos_label=2)
sensitivity = tpr
specificity = 1 - fpr
youden_index = sensitivity + specificity - 1
best_threshold = thresholds[np.argmax(youden_index)]
print('训练集上的最佳阈值:', best_threshold)

test_proba = clf.predict_proba(X_test)
test_pred = (test_proba[:, 1] >= best_threshold).astype(int) + 1
accuracy = np.mean(test_pred == y_test)
precision = np.sum((test_pred == 2) & (y_test == 2)) / np.sum(test_pred == 2)
recall_2 = np.sum((test_pred == 2) & (y_test == 2)) / np.sum(y_test == 2)
recall_1 = np.sum((test_pred == 1) & (y_test == 1)) / np.sum(y_test == 1)
f1_score = 2 * precision * recall_2 / (precision + recall_2)
print("测试集准确率:", accuracy)
print("label=2精确率:", precision)
print("label=2召回率:", recall_2)
print('label=1召回率:', recall_1)
print("label=2F1分数:", f1_score)

