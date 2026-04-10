"""
Employee Attrition Prediction - Data Preprocessing and Model Training
This script handles data loading, preprocessing, model training, and evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    """Load the dataset from CSV file"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Dataset loaded successfully! Shape: {df.shape}")
    return df

def preprocess_data(df):
    """Preprocess the data: handle missing values, encode categorical variables, scale features"""
    print("\n--- Data Preprocessing ---")
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Drop columns that are not useful for prediction
    cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    data = data.drop(columns=cols_to_drop, errors='ignore')
    
    # Handle missing values (fill with median/mode)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    # Fill numeric missing values with median
    for col in numeric_cols:
        if data[col].isnull().sum() > 0:
            data[col].fillna(data[col].median(), inplace=True)
    
    # Fill categorical missing values with mode
    for col in categorical_cols:
        if data[col].isnull().sum() > 0:
            data[col].fillna(data[col].mode()[0], inplace=True)
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        if col != 'Attrition':  # Don't encode target yet
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
    
    # Encode target variable (Attrition: Yes=1, No=0)
    data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})
    
    print(f"Categorical columns encoded: {list(label_encoders.keys())}")
    print(f"Target variable encoded: Yes=1, No=0")
    
    # Separate features and target
    X = data.drop('Attrition', axis=1)
    y = data['Attrition']
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Data preprocessing completed!")
    return X_scaled, y, X.columns.tolist(), label_encoders, scaler

def train_model(X_train, y_train, X_test, y_test):
    """Train multiple models and return the best one"""
    print("\n--- Model Training ---")
    
    # Train Logistic Regression with class weight balancing
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', C=0.5)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    lr_recall = recall_score(y_test, lr_pred)
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
    print(f"Logistic Regression Recall: {lr_recall:.4f}")
    
    # Train Random Forest with better hyperparameters
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=15, 
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_recall = recall_score(y_test, rf_pred)
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    print(f"Random Forest Recall: {rf_recall:.4f}")
    
    # Train Gradient Boosting
    print("Training Gradient Boosting...")
    from sklearn.ensemble import GradientBoostingClassifier
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_accuracy = accuracy_score(y_test, gb_pred)
    gb_recall = recall_score(y_test, gb_pred)
    print(f"Gradient Boosting Accuracy: {gb_accuracy:.4f}")
    print(f"Gradient Boosting Recall: {gb_recall:.4f}")
    
    # Choose the best model based on balanced performance
    models = [
        (lr_model, "Logistic Regression", lr_accuracy, lr_recall),
        (rf_model, "Random Forest", rf_accuracy, rf_recall),
        (gb_model, "Gradient Boosting", gb_accuracy, gb_recall)
    ]
    
    # Select model with best F1-like balance (considering both accuracy and recall)
    best_score = 0
    best_model = None
    best_name = ""
    
    for model, name, acc, rec in models:
        # Use harmonic mean-like score to balance accuracy and recall
        score = 2 * acc * rec / (acc + rec) if (acc + rec) > 0 else 0
        if score > best_score:
            best_score = score
            best_model = model
            best_name = name
    
    print(f"\nBest Model: {best_name}")
    return best_model, best_name

def evaluate_model(model, X_test, y_test, feature_names):
    """Evaluate the model and display metrics"""
    print("\n--- Model Evaluation ---")
    
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nClassification Report:\n{classification_report(y_test, predictions)}")
    
    # Feature Importance (if available)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        print(f"\nTop 10 Important Features:")
        print(feature_importance.head(10))
    
    return accuracy, precision, recall

def save_model(model, scaler, label_encoders, feature_names, filepath='models/attrition_model.pkl'):
    """Save the trained model and preprocessing objects"""
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': feature_names
    }
    joblib.dump(model_data, filepath)
    print(f"\nModel saved to {filepath}")

def main():
    """Main function to run the complete pipeline"""
    print("="*60)
    print("EMPLOYEE ATTRITION PREDICTION SYSTEM")
    print("="*60)
    
    # Load data
    df = load_data('/workspace/employee-attrition-prediction-1/10000 HRA Records.csv')
    
    # Display basic info
    print(f"\nDataset Info:")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nTarget Variable Distribution:")
    print(df['Attrition'].value_counts())
    
    # Preprocess data
    X_scaled, y, feature_names, label_encoders, scaler = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Train model
    best_model, model_name = train_model(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    accuracy, precision, recall = evaluate_model(best_model, X_test, y_test, feature_names)
    
    # Save model
    save_model(best_model, scaler, label_encoders, feature_names)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Model Type: {model_name}")
    print(f"Final Accuracy: {accuracy:.2%}")
    print("="*60)

if __name__ == "__main__":
    main()
