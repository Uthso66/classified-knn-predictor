import pandas as pd
import joblib
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_config(path='config/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def load_data(features_dir, processed_dir):
    X_test = pd.read_csv(f"{features_dir}/X_test_scaled.csv")
    y_test = pd.read_csv(f"{processed_dir}/y_test.csv")
    return X_test, y_test

def evaluate_knn(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🧪 Test Accuracy:{acc:.4f}")
    print(f"📊 Test Classification Report:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('outputs/confusion_matrix.png')
    plt.show()

def main():
    config = load_config()
    features_dir = config['data']['features_dir']
    processed_dir = config['data']['processed_dir']
    model_path = config['model']['save_path']

    X_test, y_test = load_data(features_dir, processed_dir)
    model = joblib.load(model_path)

    evaluate_knn(model, X_test, y_test)

if __name__ == "__main__":
    main()
