import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from config import PROCESSED_DATA_PATH, TEST_SIZE, RANDOM_STATE, METRICS_PATH, RESULTS_DIR
from feature_extraction import extract_temporal_features
from model_utils import save_model

from sklearn.model_selection import GroupShuffleSplit

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    return metrics, classification_report(y_test, y_pred)

def main():
    print("Loading data...")
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {PROCESSED_DATA_PATH}. Run generate_sample_data.py first.")
        
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    print("Extracting features...")
    df = extract_temporal_features(df)
    
    # Drop rows with NaN from rolling calculations
    df.dropna(inplace=True)
    
    y = df['label']
    groups = df['session_id']
    X = df.drop(['label', 'session_id'], axis=1)
    
    # Use GroupShuffleSplit to ensure frames from the same session don't leak into the test set
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'SVM': SVC(random_state=RANDOM_STATE, probability=True)
    }
    
    results = {}
    best_model_name = None
    best_f1 = -1
    best_model = None
    
    print("\n--- Training and Evaluation ---")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        metrics, report = evaluate_model(model, X_test_scaled, y_test)
        
        results[name] = metrics
        print(f"\n{name} Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f} (Crucial for drowsiness detection)")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_model_name = name
            best_model = model
            
    print(f"\nBest Model: {best_model_name} with F1-Score: {best_f1:.4f}")
    
    # Save the best model
    save_model(best_model, scaler)
    print("Best model and scaler saved successfully.")
    
    # Save metrics
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, 'w') as f:
        json.dump({'best_model': best_model_name, 'metrics': results}, f, indent=4)
    print(f"Evaluation metrics saved to {METRICS_PATH}")

if __name__ == "__main__":
    main()
