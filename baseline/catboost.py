import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve, roc_curve,confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

# Step 1: Load data
data = pd.read_csv("./SWaT_Dataset_Attack_v0_with_label_10s_and_binary_label.csv")

# Convert labels to numerical values
label_encoder = LabelEncoder()
data['label_10s'] = label_encoder.fit_transform(data['label_10s'])

# Separate features and labels
X = data.drop(columns=['label', 'label_n','label_10s','label_binary'])
y = data['label_10s']

# Step 2: Split into numerical and categorical features
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Step 3: Preprocessing pipeline
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0))
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='none'))
])

preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_features),
    ('categorical', categorical_pipeline, categorical_features)
], remainder='passthrough')

# Step 4: Join numerical and categorical features with OneHotEncoder
categorical_pipeline_with_encoding = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='none')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor_with_onehot = ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_features),
    ('categorical', categorical_pipeline_with_encoding, categorical_features)
], remainder='passthrough')

# Step 5: Transform data with the new preprocessor using OneHotEncoder
X_processed_with_onehot = preprocessor_with_onehot.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_processed_with_onehot, y, test_size=0.2, random_state=42, stratify=y)

# Step 6: Define models
cat_model = CatBoostClassifier()


# Step 7: Train models
cat_model.fit(X_train, y_train)



# Step 8: Predictions
y_pred_cat = cat_model.predict(X_test)


# Step 9: Evaluation
y_pred_proba_cat = cat_model.predict_proba(X_test)

# Compute accuracy
acc_cat = accuracy_score(y_test, np.argmax(y_pred_proba_cat, axis=1))

# Compute recall
recall_cat = recall_score(y_test, np.argmax(y_pred_proba_cat, axis=1), average='weighted')

# Compute F1-score
f1_cat = f1_score(y_test, np.argmax(y_pred_proba_cat, axis=1), average='weighted')

#Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_proba_cat)
print("Confusion Matrix:")
print(conf_matrix)


# Step 10: ROC Curve
n_classes = y_pred_proba_cat.shape[1]
fpr = dict()
tpr = dict()
auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_pred_proba_cat[:, i])
    auc[i] = roc_auc_score((y_test == i).astype(int), y_pred_proba_cat[:, i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve((y_test.ravel() == np.argmax(y_pred_proba_cat, axis=1)).astype(int), 
                                          y_pred_proba_cat[:, 1])
auc["micro"] = roc_auc_score((y_test.ravel() == np.argmax(y_pred_proba_cat, axis=1)).astype(int), 
                             y_pred_proba_cat[:, 1])

# Plot ROC curve
plt.figure()
plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'
         ''.format(auc["micro"]), color='deeppink', linestyle=':', linewidth=4)

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for CatBoost Model (Multi-class)')
plt.legend(loc="lower right")
plt.show()

# Step 11: Display Results
print("CatBoost:")
print("Accuracy:", acc_cat)
print("Recall:", recall_cat)
print("F1-score:", f1_cat)
