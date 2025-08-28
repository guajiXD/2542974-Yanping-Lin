import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV,cross_val_score,train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,make_scorer,recall_score,
    classification_report,roc_curve, auc,matthews_corrcoef
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")


def train_smooth_encoding_model(X_train, y_train, X_test, y_test, weight=100):
    print("================== Smooth Encoding ==================")
    categorical_cols = ['Reco_Policy_Cat', 'City_Code', 'Region_Code']

    train_data = X_train.copy()
    test_data = X_test.copy()
    train_data['Response'] = y_train.values

    def smooth_target_encode(train_df, test_df, column, target, weight):
        global_mean = train_df[target].mean()
        stats = train_df.groupby(column)[target].agg(['count', 'mean'])
        smooth_means = (stats['count'] * stats['mean'] + weight * global_mean) / (stats['count'] + weight)
        train_df[f'{column}_enc'] = train_df[column].map(smooth_means)
        test_df[f'{column}_enc'] = test_df[column].map(smooth_means)
        test_df[f'{column}_enc'] = test_df[f'{column}_enc'].fillna(global_mean)
        return train_df, test_df

    for col in categorical_cols:
        train_data, test_data = smooth_target_encode(train_data, test_data, col, 'Response', weight)

    # Delete original columns
    train_data = train_data.drop(columns=categorical_cols)
    test_data = test_data.drop(columns=categorical_cols)

    # One-Hot
    onehot_cols = ['Reco_Insurance_Type', 'Accomodation_Type', 'Health_Indicator',
                  'Holding_Policy_Type', 'Is_Spouse']

    train_onehot = pd.get_dummies(train_data[onehot_cols], drop_first=True)
    test_onehot = pd.get_dummies(test_data[onehot_cols], drop_first=True)

    missing_cols = set(train_onehot.columns) - set(test_onehot.columns)
    for col in missing_cols:
        test_onehot[col] = 0
    test_onehot = test_onehot[train_onehot.columns]

    # Combine
    num_features = ["Upper_Age", "Lower_Age", "Reco_Policy_Premium", 'Holding_Policy_Duration']
    X_train_final = pd.concat([train_data.drop(columns=onehot_cols + ['Response']),
                              train_onehot, train_data[num_features]], axis=1)
    X_test_final = pd.concat([test_data.drop(columns=onehot_cols),
                             test_onehot, test_data[num_features]], axis=1)

    scaler = StandardScaler()
    X_train_final[num_features] = scaler.fit_transform(X_train_final[num_features])
    X_test_final[num_features] = scaler.transform(X_test_final[num_features])

    return X_train_final, X_test_final, train_data['Response'], y_test

def train_bayesian_encoding_model(X_train, y_train, X_test, y_test):
    print("=" * 30,"Bayesian","=" * 30)
    categorical_cols = ['Reco_Policy_Cat', 'City_Code', 'Region_Code']

    train_data = X_train.copy()
    test_data = X_test.copy()
    train_data['Response'] = y_train.values

    def bayesian_target_encode(train_df, test_df, column, target):
        prior = train_df[target].mean()
        stats = train_df.groupby(column)[target].agg(['count', 'mean'])
        bayesian_means = (stats['count'] * stats['mean'] + prior) / (stats['count'] + 1)
        train_df[f'{column}_enc'] = train_df[column].map(bayesian_means)
        test_df[f'{column}_enc'] = test_df[column].map(bayesian_means)
        test_df[f'{column}_enc'] = test_df[f'{column}_enc'].fillna(prior)
        return train_df, test_df

    # Encode
    for col in categorical_cols:
        train_data, test_data = bayesian_target_encode(train_data, test_data, col, 'Response')

    # Delete
    train_data = train_data.drop(columns=categorical_cols)
    test_data = test_data.drop(columns=categorical_cols)

    # One-Hot
    onehot_cols = ['Reco_Insurance_Type', 'Accomodation_Type', 'Health_Indicator',
                  'Holding_Policy_Type', 'Is_Spouse']

    train_onehot = pd.get_dummies(train_data[onehot_cols], drop_first=True)
    test_onehot = pd.get_dummies(test_data[onehot_cols], drop_first=True)

    missing_cols = set(train_onehot.columns) - set(test_onehot.columns)
    for col in missing_cols:
        test_onehot[col] = 0
    test_onehot = test_onehot[train_onehot.columns]

    # Combine
    num_features = ["Upper_Age", "Lower_Age", "Reco_Policy_Premium", 'Holding_Policy_Duration']
    X_train_final = pd.concat([train_data.drop(columns=onehot_cols + ['Response']),
                              train_onehot, train_data[num_features]], axis=1)
    X_test_final = pd.concat([test_data.drop(columns=onehot_cols),
                             test_onehot, test_data[num_features]], axis=1)

    scaler = StandardScaler()
    X_train_final[num_features] = scaler.fit_transform(X_train_final[num_features])
    X_test_final[num_features] = scaler.transform(X_test_final[num_features])

    return X_train_final, X_test_final, train_data['Response'], y_test

def train_kfold_encoding_model(X_train, y_train, X_test, y_test, n_splits=5):
    print("=" * 30,"K Fold","=" * 30)
    categorical_cols = ['Reco_Policy_Cat', 'City_Code', 'Region_Code']

    train_data = X_train.copy()
    test_data = X_test.copy()
    train_data['Response'] = y_train.values

    def kfold_target_encode(train_df, column, target, n_splits):
        train_df[f'{column}_enc'] = 0
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(train_df):
            train_fold = train_df.iloc[train_idx]
            val_fold = train_df.iloc[val_idx]

            mean_encoding = train_fold.groupby(column)[target].mean()
            train_df.iloc[val_idx, train_df.columns.get_loc(f'{column}_enc')] = \
                val_fold[column].map(mean_encoding).fillna(train_df[target].mean())

        return train_df

    # K-Fold in Train data
    for col in categorical_cols:
        train_data = kfold_target_encode(train_data, col, 'Response', n_splits)

    # Mean in Test data
    for col in categorical_cols:
        global_mean = train_data.groupby(col)['Response'].mean()
        test_data[f'{col}_enc'] = test_data[col].map(global_mean)
        test_data[f'{col}_enc'] = test_data[f'{col}_enc'].fillna(train_data['Response'].mean())

    # Delete
    train_data = train_data.drop(columns=categorical_cols)
    test_data = test_data.drop(columns=categorical_cols)

    # One-Hot
    onehot_cols = ['Reco_Insurance_Type', 'Accomodation_Type', 'Health_Indicator',
                  'Holding_Policy_Type', 'Is_Spouse']

    train_onehot = pd.get_dummies(train_data[onehot_cols], drop_first=True)
    test_onehot = pd.get_dummies(test_data[onehot_cols], drop_first=True)

    missing_cols = set(train_onehot.columns) - set(test_onehot.columns)
    for col in missing_cols:
        test_onehot[col] = 0
    test_onehot = test_onehot[train_onehot.columns]

    # Combine
    num_features = ["Upper_Age", "Lower_Age", "Reco_Policy_Premium", 'Holding_Policy_Duration']
    X_train_final = pd.concat([train_data.drop(columns=onehot_cols + ['Response']),
                              train_onehot, train_data[num_features]], axis=1)
    X_test_final = pd.concat([test_data.drop(columns=onehot_cols),
                             test_onehot, test_data[num_features]], axis=1)

    scaler = StandardScaler()
    X_train_final[num_features] = scaler.fit_transform(X_train_final[num_features])
    X_test_final[num_features] = scaler.transform(X_test_final[num_features])

    return X_train_final, X_test_final, train_data['Response'], y_test
