import streamlit as st
# Feature Engineering
from sklearn.preprocessing import FunctionTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import root_mean_squared_error, accuracy_score, f1_score, roc_auc_score

# Prediction
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    filepath = Path('data') / 'german_credit_data.csv'
    df = pd.read_csv(filepath)#.drop(columns=['Unnamed: 0'])
    # assigning categorical columns to categorical type
    cleaned_df = df.copy()
    categorical = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    for column in categorical:
        cleaned_df[column] = df[column].astype(dtype='category')
        
    return cleaned_df

@st.cache_resource
def load_model():
    def add_ratio_features(X):
        X = X.copy()
        
        wealth_mapper = {
            'little': 1,
            'moderate':3,
            'rich': 8,
            'quite rich':10
        }
        credit_mapper = {
        'low': -1,
        'medium':0,
        'high':1, 
        'very high':3
        }
        duration_mapper = {
            'short':1,
            'average':0,
            'long':-1,
            'very long':-2
        }

        
        X['Credit amount bin'] = pd.cut(X['Credit amount'], bins = 4, labels=['low', 'medium', 'high', 'very high']).map(credit_mapper).astype(int)
        X['Duration bin'] = pd.cut(X['Duration'], bins = 4, labels=['short', 'average', 'long', 'very long']).map(duration_mapper).astype(int)
        X['is renting'] = X['Housing'] == 'rent'

        X['Saving checking average'] = X['Saving accounts'].map(wealth_mapper).astype(int) + X['Checking account'].map(wealth_mapper).astype(int) /2
        X['Credit amount score'] = X['Saving checking average'] - X['Credit amount bin'] * X['Duration bin']

        checking = X['Checking account'].map(wealth_mapper)
        saving = X['Saving accounts'].map(wealth_mapper)

        X['Credit income ratio'] = X['Credit amount'] / (checking.astype(int) + saving.astype(int))
        
        return X
    

    analysis_df = load_data().drop(columns=["Unnamed: 0", 'Sex']).dropna()
    analysis_df.head()
    
    X, y = analysis_df.drop(columns='Risk'), analysis_df['Risk']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)    
    
    # cat_columns = ["Sex", 'Purpose']
    cat_columns = ['Purpose']
    ord_columns = ["Saving accounts", "Checking account", "Housing"]
    standardized_cols = ['Credit amount', 'Age', 'Credit income ratio']

    savings_orderings = list(analysis_df['Saving accounts'].unique())
    checking_orderings = list(analysis_df['Checking account'].unique())


    column_trans = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_columns),
            ('ord', OrdinalEncoder(
                categories=[
                    ['little', 'moderate', 'rich', 'quite rich'],
                    ['little', 'moderate', 'rich'],
                    ['free', 'rent', 'own']
                ]
                ), ord_columns),
            ('standardize', StandardScaler(), standardized_cols)
            ],
        remainder='passthrough'
    )

    func_trans = FunctionTransformer(add_ratio_features)

    xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    use_label_encoder=False,
    eval_metric='logloss'
    )

    xgb_pipe = Pipeline(steps=[
        ('credit ratio', func_trans),
        ('preprocess', column_trans),
        ('predict', xgb)
    ])

    xgb_y_train = y_train.map({'good':1,'bad':0})
    xgb_y_test = y_test.map({'good':1,'bad':0})


    xgb_pipe.fit(X_train, xgb_y_train)
    
    return xgb_pipe


# ----------------------------------------------------------------------------------- # 
# App Portion

skill_map = {'Unskilled': 0, 'Unskilled Resident': 1, 'Skilled': 2,  'Highly Skilled':3}

st.title("Loan Acceptance Predictor")

st.sidebar.header("Credit Information Form")
Age = st.sidebar.number_input("Age.", format="%d", step=1)
# sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
skill =  st.sidebar.selectbox("Job Skill", ['Unskilled', 'Unskilled Resident', 'Skilled', 'Highly Skilled'])
housing = st.sidebar.selectbox("Housing", ['free', 'own', 'rent'])
saving_account = st.sidebar.selectbox("Savings Account Level", ['little', 'moderate', 'quite rich', 'rich'])
checking_account = st.sidebar.selectbox("Checking Account Level", ['little', 'moderate', 'rich'])
credit_amount = st.sidebar.number_input("Credit Needed in Euros", format = "%d", step=1)
duration = st.sidebar.number_input("Enter Loan Duration", format = "%d", step=1)
purpose = st.sidebar.selectbox("Purpose", ['radio/TV', 'furniture/equipment', 'car', 'business', 'domestic appliances', 'repairs', 'vacation/others', 'education'])

submit_button = st.sidebar.button("Submit")
if submit_button:
    # st.write(f"work skill{skill_map[skill]}")
    st.write(skill_map[skill])
    mdl = load_model()
    # mdl.predict()
    input = pd.DataFrame({
        'Age':[Age],
        # 'Sex':[sex],
        'Job':[skill_map[skill]],
        'Housing':[housing],
        'Saving accounts' :[saving_account],
        'Checking account':[checking_account],
        'Credit amount': [credit_amount],
        'Duration': [duration],
        'Purpose':[purpose]
    })
    
    outcome_mapper = {
        0:'bad',
        1:'good'
    }
    outcome = outcome_mapper[mdl.predict(input)[0]]
    probs = mdl.predict_proba(input)[0]
     
    st.write(f"Percieved as a {outcome} risk.")    

    
