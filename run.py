import os

print("🚀 Running KNN Classification Pipeline...")

# 1️⃣ Preprocess Data
os.system("python src/data/preprocess.py")

# 2️⃣ Build Features
os.system("python -m src.features.build_features")

# 3️⃣ Train Model
os.system("python src/models/train_model.py")

# 4️⃣ Evaluate on Test Set
os.system("python src/models/evaluate_test.py")

print("✅ Full KNN pipeline completed!")
