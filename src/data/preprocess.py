import os
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_config(path='config/config.yaml'):
    """⚙️ Load project settings from YAML file"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_data(df):
    """🧹 Clean data and separate features/target"""
    # Validate dataset
    if 'TARGET CLASS' not in df.columns:
        raise KeyError("Target column 'TARGET CLASS' not found in dataset")
        
    if df.isnull().sum().sum() > 0:
        missing = df.isnull().sum()
        raise ValueError(f"Missing values detected:\n{missing[missing > 0]}")
    
    # Feature-target separation
    X = df.drop(columns=['TARGET CLASS'])
    y = df['TARGET CLASS']
    
    # Check class balance
    class_counts = y.value_counts()
    if min(class_counts) / max(class_counts) < 0.3:
        print(f"⚠️ Class imbalance detected: {class_counts.to_dict()}")
    
    return X, y

def split_and_save(X, y, test_size, val_size, random_state, processed_dir):
    """✂️ Split data into train/val/test sets and save"""
    # First split: full → temp + test (stratified)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y,
        random_state=random_state
    )
    
    # Second split: temp → train + val (stratified)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=val_ratio, 
        stratify=y_temp,
        random_state=random_state
    )
    
    # Create output directory
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save features
    X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    X_val.to_csv(os.path.join(processed_dir, "X_val.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    
    # Save targets
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(processed_dir, "y_val.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)
    
    # Save class distribution report
    class_dist = pd.DataFrame({
        'set': ['Full', 'Train', 'Validation', 'Test'],
        'size': [len(y), len(y_train), len(y_val), len(y_test)],
        'class_0': [sum(y==0), sum(y_train==0), sum(y_val==0), sum(y_test==0)],
        'class_1': [sum(y==1), sum(y_train==1), sum(y_val==1), sum(y_test==1)]
    })
    class_dist.to_csv(os.path.join(processed_dir, "class_distribution.csv"), index=False)
    
    # Create class distribution visualization
    plt.figure(figsize=(10, 6))
    for i, (name, group) in enumerate([('Train', y_train), ('Val', y_val), ('Test', y_test)]):
        counts = group.value_counts().sort_index()
        plt.bar([x + i*0.25 for x in counts.index], counts.values, width=0.25, label=name)
    
    plt.title('Class Distribution Across Sets')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks([0,1], ['Class 0', 'Class 1'])
    plt.legend()
    plt.savefig(os.path.join("outputs/class_distribution.png"))
    plt.close()
    
    print(f"✅ Data split and saved to {processed_dir}")
    print(f"📊 Dataset sizes: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
    print(f"📈 Class distribution saved to class_distribution.png")

def main():
    # ⚙️ Load configuration
    config = load_config()
    
    # 📥 Load data
    print(f"⏳ Loading data from {config['data']['raw_path']}")
    df = pd.read_csv(config['data']['raw_path'])
    print(f"🔍 Data shape: {df.shape}")
    
    # 🧹 Preprocess
    X, y = preprocess_data(df)
    
    # ✂️ Split and save
    split_and_save(
        X, y,
        test_size=config['data']['test_size'],
        val_size=config['data']['val_size'],
        random_state=config['data']['random_state'],
        processed_dir=config['data']['processed_dir']
    )

if __name__ == '__main__':
    main()