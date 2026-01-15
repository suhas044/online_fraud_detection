
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fraud_detection_system import FraudDetectionSystem
from data_generator import generate_synthetic_data

app = FastAPI()

# Global variable to store the trained system
fds = None

class TransactionData(BaseModel):
    oldbalanceOrg: float
    newbalanceOrig: float
    amount: float
    type: str
    hour: int
    num_tx_24h: int
    location_consistency: int
    device_status: int
    account_age_days: int

@app.on_event("startup")
def startup_event():
    global fds
    print("Initializing Fraud Detection System...")
    fds = FraudDetectionSystem()
    
    # Check for data or generate it
    data_path = 'synthetic_fraud_data.csv'
    if not os.path.exists(data_path):
        print("Generating synthetic data...")
        df = generate_synthetic_data(n_samples=50000) # Smaller sample for quick startup
        df.to_csv(data_path, index=False)
    else:
        print("Loading existing data...")
        df = pd.read_csv(data_path)
    
    # Prepare data
    X = df.drop(['isFraud'], axis=1)
    y = df['isFraud']
    
    categorical_features = ['type']
    numeric_features = [col for col in X.columns if col not in categorical_features]
    
    fds._create_preprocessor(numeric_features, categorical_features)
    
    # Split and Train (using ML model for speed and reliability in demo)
    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training ML Model...")
    fds.train_ml_model(X_train, y_train, model_type='xgboost')
    print("System Ready.")

@app.post("/predict")
async def predict_fraud(data: TransactionData):
    if fds is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Calculate derived feature
    balance_diff = data.oldbalanceOrg - data.newbalanceOrig
    
    # Create DataFrame for prediction
    input_df = pd.DataFrame([{
        'oldbalanceOrg': data.oldbalanceOrg,
        'newbalanceOrig': data.newbalanceOrig,
        'amount': data.amount,
        'balance_diff': balance_diff,
        'type': data.type,
        'hour': data.hour,
        'num_tx_24h': data.num_tx_24h,
        'location_consistency': data.location_consistency,
        'device_status': data.device_status,
        'account_age_days': data.account_age_days
    }])
    
    try:
        # Predict probability
        prob = fds.predict(input_df, model_type='ml')[0]
        # Predict class (threshold 0.5)
        is_fraud = bool(prob > 0.5)
        
        return {
            "probability": float(prob),
            "is_fraud": is_fraud,
            "risk_level": "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")
