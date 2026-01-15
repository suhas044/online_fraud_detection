import pandas as pd
import os
from data_generator import generate_synthetic_data
from fraud_detection_system import FraudDetectionSystem

def main():
    # 1. Data Setup
    data_path = 'synthetic_fraud_data.csv'
    if not os.path.exists(data_path):
        print("Dataset not found. Generating synthetic data...")
        df = generate_synthetic_data(n_samples=100000)
        df.to_csv(data_path, index=False)
    else:
        print("Loading existing dataset...")
        df = pd.read_csv(data_path)
        
    print(f"Data Loaded. Shape: {df.shape}")
    print(df.head())

    # 2. Prepare Features and Target
    X = df.drop(['isFraud'], axis=1)
    y = df['isFraud']

    # Define categorical and numeric features
    # Note: 'type' is categorical. 'device_status', 'location_consistency' are already 0/1 but can be treated as numeric or categorical.
    # We will treat them as numeric (binary) since they are already encoded as 0/1. 
    # 'type' needs encoding.
    categorical_features = ['type']
    numeric_features = [col for col in X.columns if col not in categorical_features]
    
    # 3. Initialize System
    fds = FraudDetectionSystem()
    
    # Create preprocessor first to define the pipeline structure
    fds._create_preprocessor(numeric_features, categorical_features)
    
    # Split Data
    # Stratify to ensure fraud cases are distributed
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Train ML Model
    fds.train_ml_model(X_train, y_train, model_type='xgboost')
    
    # 5. Evaluate ML Model
    print("\n--- ML Model Evaluation (XGBoost) ---")
    ml_metrics = fds.evaluate(X_test, y_test, model_type='ml')
    for k, v in ml_metrics.items():
        print(f"{k}: {v:.4f}")
        
    # 6. Train DL Model
    # fds.train_dl_model(X_train, y_train, epochs=5) # Reduced epochs for demo speed
    
    # 7. Evaluate DL Model
    # print("\n--- DL Model Evaluation (ANN) ---")
    # dl_metrics = fds.evaluate(X_test, y_test, model_type='dl')
    # for k, v in dl_metrics.items():
    #     print(f"{k}: {v:.4f}")
        
    # Commenting out DL training by default to avoid extensive dependency issues if user lacks tensorflow/pytorch setup,
    # but the code is there. If the user installed requirements, they can uncomment.
    # Actually, for the request, I should probably run it if I can. 
    # Let's try running it. If it fails, the user will see.
    try:
        fds.train_dl_model(X_train, y_train, epochs=5)
        print("\n--- DL Model Evaluation (ANN) ---")
        dl_metrics = fds.evaluate(X_test, y_test, model_type='dl')
        for k, v in dl_metrics.items():
            print(f"{k}: {v:.4f}")
    except Exception as e:
        print(f"\n[Warning] Deep Learning Model training failed (likely missing tensorflow): {e}")

    # 8. Sample Prediction
    print("\n--- Sample Prediction ---")
    sample_legit = X_test[y_test == 0].iloc[0:1]
    sample_fraud = X_test[y_test == 1].iloc[0:1]
    
    print("Legitimate Transaction Prediction:")
    prob_legit = fds.predict(sample_legit, model_type='ml')[0]
    print(f"Probability of Fraud: {prob_legit:.4f}")
    
    print("\nFraudulent Transaction Prediction:")
    prob_fraud = fds.predict(sample_fraud, model_type='ml')[0]
    print(f"Probability of Fraud: {prob_fraud:.4f}")
    
    # Add manual suspicious case
    manual_case = pd.DataFrame([{
        'oldbalanceOrg': 10000,
        'newbalanceOrig': 0,
        'amount': 10000,
        'balance_diff': 10000,
        'type': 'UPI',
        'hour': 3,          # Late night
        'num_tx_24h': 15,   # High freq
        'location_consistency': 0, # Mismatch
        'device_status': 0, # New device
        'account_age_days': 5
    }])
    
    print("\nManual Suspicious Case Prediction:")
    prob_suspicious = fds.predict(manual_case, model_type='ml')[0]
    print(f"Probability of Fraud: {prob_suspicious:.4f}")

if __name__ == "__main__":
    main()
