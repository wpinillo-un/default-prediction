# flake8: noqa: E501

import pandas as pd
import numpy as np

test_data = pd.read_csv(
    "files/input/test_data.csv.zip",
    index_col=False,
    compression="zip",
)

train_data = pd.read_csv(
    "files/input/train_data.csv.zip",
    index_col=False,
    compression="zip",
)

# Clean the dataset
def clear_data(df):
    df=df.rename(columns={'default payment next month': 'default'})
    df=df.drop(columns=['ID'])
    df = df.loc[df["MARRIAGE"] != 0]
    df = df.loc[df["EDUCATION"] != 0]
    df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 4 if x > 4 else x)

    return df
train_data=clear_data(train_data)
test_data=clear_data(test_data)

# Split the data into x_train, y_train, x_test, y_test

x_train=train_data.drop(columns="default")
y_train=train_data["default"]


x_test=test_data.drop(columns="default")
y_test=test_data["default"]

# Create a pipeline for the classification model

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Categorical features 
categorical_features=["SEX","EDUCATION","MARRIAGE"]

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ],
    remainder="passthrough"
)

#pipeline
pipeline=Pipeline(
    [
        ("preprocessor",preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ]
)

# Optimize hyperparameters using crossvalidation 


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, balanced_accuracy_score

# Definir the param grid
param_grid = {
    'classifier__n_estimators': [100],
    'classifier__max_depth': [None],
    'classifier__min_samples_split': [10],
    'classifier__min_samples_leaf': [4],
    "classifier__max_features":[23]
}

model=GridSearchCV(
    pipeline,
    param_grid,
    cv=10,
    scoring="balanced_accuracy",
    n_jobs=-1,
    refit=True
    )

model.fit(x_train, y_train)

# Save the model "files/models/model.pkl"
import pickle
import os

models_dir = 'files/models'
os.makedirs(models_dir, exist_ok=True)

with open("files/models/model.pkl","wb") as file:
    pickle.dump(model,file)

# Calculate metrics

import json
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score

def metrics(model, X_train, X_test, y_train, y_test):

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics_train = {
        'type': 'metrics',
        'dataset': 'train',
        'precision': precision_score(y_train, y_train_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred, zero_division=0),
        'f1_score': f1_score(y_train, y_train_pred, zero_division=0)
    }

    metrics_test = {
        'type': 'metrics',
        'dataset': 'test',
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_test_pred, zero_division=0)
    }

    output_dir = 'files/output'
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'metrics.json')
    with open(output_path, 'w') as f:
        f.write(json.dumps(metrics_train) + '\n')
        f.write(json.dumps(metrics_test) + '\n')

# Compute confusion matrix

from sklearn.metrics import confusion_matrix

def matrices(model, X_train, X_test, y_train, y_test):

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    def format_confusion_matrix(cm, dataset_type):
        return {
            'type': 'cm_matrix',
            'dataset': dataset_type,
            'true_0': {
                'predicted_0': int(cm[0, 0]),
                'predicted_1': int(cm[0, 1])
            },
            'true_1': {
                'predicted_0': int(cm[1, 0]),
                'predicted_1': int(cm[1, 1])
            }
        }

    metrics = [
        format_confusion_matrix(cm_train, 'train'),
        format_confusion_matrix(cm_test, 'test')
    ]

    output_path = 'files/output/metrics.json'
    with open(output_path, 'a') as f:
        for metric in metrics:
            f.write(json.dumps(metric) + '\n')

# Main function
def main(model, X_train, X_test, y_train, y_test):
    import os
    os.makedirs('files/output', exist_ok=True)
    metrics(model, X_train, X_test, y_train, y_test)
    matrices(model, X_train, X_test, y_train, y_test)

main(model, x_train, x_test, y_train, y_test)