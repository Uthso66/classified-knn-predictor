import os
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
from src.data.preprocess import load_config

def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def save_scaled(X_train, X_val, X_test, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(X_train).to_csv(f"{output_dir}/X_train_scaled.csv", index=False)
    pd.DataFrame(X_val).to_csv(f"{output_dir}/X_val_scaled.csv", index=False)
    pd.DataFrame(X_test).to_csv(f"{output_dir}/X_test_scaled.csv", index=False)

def main():
    config = load_config()
    processed_dir = config['data']['processed_dir']
    features_dir = config['data']['features_dir']

    X_train = pd.read_csv(f"{processed_dir}/X_train.csv")
    X_val = pd.read_csv(f"{processed_dir}/X_val.csv")
    X_test = pd.read_csv(f"{processed_dir}/X_test.csv")

    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_data(
        X_train, X_val, X_test
    )

    save_scaled(X_train_scaled, X_val_scaled, X_test_scaled, features_dir)

    print("✅ Features scaled & saved!")

if __name__ == '__main__':
    main()
