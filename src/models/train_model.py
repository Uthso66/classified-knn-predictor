import pandas as pd
import yaml
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

def load_config(path='config/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_data(features_dir, processed_dir):
    
    X_train = pd.read_csv(f"{features_dir}/X_train_scaled.csv")
    X_val = pd.read_csv(f"{features_dir}/X_val_scaled.csv")
    y_train = pd.read_csv(f"{processed_dir}/y_train.csv").values.ravel()
    y_val = pd.read_csv(f"{processed_dir}/y_val.csv").values.ravel()

    return X_train, X_val, y_train, y_val

def train_knn(X_train, y_train, k, metric, p):
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric, p=p)
    knn.fit(X_train, y_train)
    return knn


def validate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    print(f"✅ Validation Accuracy: {acc:.4f}")
    print("📊 Classification Report:")
    print(classification_report(y_val, y_pred))

def main():
    config = load_config()
    k = config['model']['hyperparameters']['n_neighbors']  
    metric = config['model']['hyperparameters']['metric']
    p = config['model']['hyperparameters']['p']
    features_dir = config['data']['features_dir']
    processed_dir = config['data']['processed_dir']
    model_path = config['model']['save_path']

    X_train, X_val, y_train, y_val = load_data(features_dir, processed_dir)
    model = train_knn(X_train, y_train, k, metric, p)

    validate_model(model, X_val, y_val)

    joblib.dump(model, model_path)
    print(f"💾 Model save to: {model_path}")

if __name__ == '__main__':
    main()
