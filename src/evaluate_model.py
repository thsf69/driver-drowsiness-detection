import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GroupShuffleSplit
from config import PROCESSED_DATA_PATH, TEST_SIZE, RANDOM_STATE, RESULTS_DIR
from feature_extraction import extract_temporal_features
from model_utils import load_model

def main():
    print("Loading data for visualization...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df = extract_temporal_features(df)
    df.dropna(inplace=True)
    
    y = df['label']
    groups = df['session_id']
    X = df.drop(['label', 'session_id'], axis=1)
    
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    
    print("Loading best model...")
    model, scaler = load_model()
    
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Alert', 'Drowsy'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    
    os.makedirs(os.path.join(RESULTS_DIR, 'figures'), exist_ok=True)
    cm_path = os.path.join(RESULTS_DIR, 'figures', 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    plt.close()
    
    # 2. Feature Importance (if Random Forest)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        features = X.columns
        indices = importances.argsort()
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        
        fi_path = os.path.join(RESULTS_DIR, 'figures', 'feature_importance.png')
        plt.savefig(fi_path)
        print(f"Feature importance plot saved to {fi_path}")
        plt.close()
        
if __name__ == "__main__":
    main()
