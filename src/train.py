import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score

from data_processing import load_data, preprocess_data, get_features_and_target

def train_model():
    # Load and preprocess data
    data_path = os.path.join("Data", "cleanedData.csv")
    if not os.path.exists(data_path):
        # Fallback if running from src directory
        data_path = os.path.join("..", "Data", "cleanedData.csv")
        
    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    
    print("Preprocessing data...")
    df_processed = preprocess_data(df)
    
    target_col = 'Churn'
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col].map({'Yes': 1, 'No': 0})
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Define features for transformation
    numeric_features = ['tenure', 'MonthlyCharges', 'charges_per_tenure']
    categorical_features = ['tenure_bucket', 'Contract', 'InternetService', 'PaymentMethod', 'gender']
    binary_features = ['SeniorCitizen', 'Partner', 'Dependents', 'family_status', 'PaperlessBilling', 'is_auto_payment']
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
            ('bin', 'passthrough', binary_features)
        ],
        remainder='drop' # Drop other columns like TotalCharges if present
    )
    
    # XGBoost specific parameters from notebook analysis
    # scale_pos_weight is calculated to handle class imbalance
    neg_count = y_train.value_counts()[0]
    pos_count = y_train.value_counts()[1]
    scale_pos_weight_value = neg_count / pos_count
    
    xgb_classifier = XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight_value,
        use_label_encoder=False
    )
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb_classifier)
    ])
    
    # Train
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Save model
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_path = os.path.join(model_dir, "xgb_model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
