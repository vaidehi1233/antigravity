import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from imblearn.over_sampling import SMOTE

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

def train_classification_model(df):
    print("Training Classification Model (Eligibility)...")
    
    X = df[['Credit_Score', 'Age', 'Monthly_Income', 'Previous_Loan']]
    y = df['Is_Eligible'].map({'Yes': 1, 'No': 0})
    
    # Preprocessing
    numeric_features = ['Credit_Score', 'Age', 'Monthly_Income']
    categorical_features = ['Previous_Loan']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply preprocessing to training data for SMOTE (since SMOTE needs numeric inputs)
    # Actually, it's better to use SMOTENC if we have categorical, 
    # but here we only have one categorical feature.
    # For simplicity, we'll preprocess then SMOTE.
    
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    # Handle Class Imbalance
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_preprocessed, y_train)
    
    # Train Model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_res, y_train_res)
    
    # Evaluate
    y_pred = clf.predict(X_test_preprocessed)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save Pipeline (Preprocessor + Model)
    # We need to create a new pipeline that includes the trained model but uses the preprocessor
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    joblib.dump(full_pipeline, 'models/classification_pipeline.joblib')
    print("Classification model saved to models/classification_pipeline.joblib")
    return full_pipeline

def train_regression_models(df):
    print("\nTraining Regression Models (Loan Terms)...")
    
    # Only train on eligible samples (real-world scenario)
    eligible_df = df[df['Is_Eligible'] == 'Yes']
    
    X = eligible_df[['Credit_Score', 'Age', 'Monthly_Income', 'Previous_Loan']]
    y = eligible_df[['Loan_Amount', 'Interest_Rate', 'Loan_Tenure']]
    
    # Preprocessing
    numeric_features = ['Credit_Score', 'Age', 'Monthly_Income']
    categorical_features = ['Previous_Loan']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model (Multi-output Regressor using Random Forest)
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    
    reg_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', reg)
    ])
    
    reg_pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = reg_pipeline.predict(X_test)
    print(f"Regression R2 Score: {r2_score(y_test, y_pred, multioutput='uniform_average'):.4f}")
    print(f"Regression MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    
    # Save Pipeline
    joblib.dump(reg_pipeline, 'models/regression_pipeline.joblib')
    print("Regression model saved to models/regression_pipeline.joblib")
    return reg_pipeline

if __name__ == "__main__":
    if not os.path.exists('loan_data.csv'):
        print("Data file not found. Please run data_generator.py first.")
    else:
        df = pd.read_csv('loan_data.csv')
        train_classification_model(df)
        train_regression_models(df)
