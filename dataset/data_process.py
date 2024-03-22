import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

#  Load data
data = pd.read_csv("./SWaT_Dataset_Attack_v0.csv")

data['Time'] = pd.to_datetime(data['Time'], infer_datetime_format=True, errors='coerce')
data['Time'] = pd.to_datetime(data['Time'], format='%m/%d/%Y %H:%M:%S.%f')

# 将数据按照 10 秒的时间间隔分组
groups = []
for group_key, group_data in data.groupby(pd.Grouper(key='Time', freq='10S')):
    # 获取非normal状态下的行，并将它们的label添加到该组中
    non_normal_labels = group_data[group_data['label'] != 'normal']['label'].tolist()
    if non_normal_labels:
        group_data['label_10s'] = non_normal_labels[0]  # 假设每个时间窗口只有一个非normal类别
    else:
        group_data['label_10s'] = 'normal'  # 如果没有非normal类别，则标记为normal
    groups.append(group_data)

# 将处理后的数据拼接起来
result = pd.concat(groups)

result['label_binary'] = result['label_10s'].apply(lambda x: 'normal' if x == 'normal' else 'attack') 

# 保存DataFrame为新的CSV文件
result.to_csv("./SWaT_Dataset_Attack_v0_with_label_10s_and_binary_label.csv", index=False)
