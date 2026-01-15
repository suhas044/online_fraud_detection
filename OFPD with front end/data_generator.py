import pandas as pd
import numpy as np
import random
from datetime import datetime

def generate_synthetic_data(n_samples=100000):
    """
    Generates a synthetic dataset for online payment fraud detection.
    
    Features:
    - oldbalanceOrg: Old account balance
    - newbalanceOrig: New account balance
    - amount: Transaction amount
    - type: Transaction type (UPI, Card, Net Banking, Wallet)
    - hour: Transaction hour (0-23)
    - num_tx_24h: Number of transactions in the last 24 hours
    - location_consistency: 0 or 1 (same city/country)
    - device_status: 0 (new) or 1 (known)
    - account_age_days: Account age in days
    - isFraud: Target variable (0 or 1)
    """
    
    np.random.seed(42)
    random.seed(42)
    
    print(f"Generating {n_samples} samples...")
    
    # 1. Transaction Types
    types = ['UPI', 'Card', 'Net Banking', 'Wallet']
    # Probabilities roughly mimicking typical distribution
    type_probs = [0.4, 0.3, 0.2, 0.1] 
    data_types = np.random.choice(types, n_samples, p=type_probs)
    
    # 2. Account Age
    account_age = np.random.randint(1, 3650, n_samples) # 1 day to 10 years
    
    # 3. Transaction Hour
    hours = np.random.randint(0, 24, n_samples)
    
    # 4. Device Status (Most are known devices)
    # 1 = Known, 0 = New
    device_status = np.random.choice([0, 1], n_samples, p=[0.1, 0.9])
    
    # 5. Location Consistency (Most are consistent)
    # 1 = Consistent, 0 = Inconsistent
    location_consistency = np.random.choice([0, 1], n_samples, p=[0.05, 0.95])
    
    # 6. Number of transactions in last 24h
    # Skewed distribution
    num_tx_24h = np.random.poisson(lam=3, size=n_samples)
    
    # Placeholder arrays for dependent features
    amounts = np.zeros(n_samples)
    old_balances = np.zeros(n_samples)
    new_balances = np.zeros(n_samples)
    is_fraud = np.zeros(n_samples, dtype=int)
    
    # Loop to generate context-dependent values (slower but easier to control logic)
    # Optimizing with vectorization where possible for speed
    
    # Base amounts (log-normal distribution for realistic transaction sizes)
    amounts = np.random.lognormal(mean=6, sigma=1.5, size=n_samples)
    
    # Base Balances
    old_balances = np.random.lognormal(mean=9, sigma=2, size=n_samples)
    
    # Logic for fraud injection
    for i in range(n_samples):
        # Default non-fraud behavior
        fraud_prob = 0.001 # Base probability
        
        # Risk factors
        if hours[i] < 5: # Late night
            fraud_prob += 0.02
        if location_consistency[i] == 0:
            fraud_prob += 0.05
        if device_status[i] == 0: # New device
            fraud_prob += 0.05
        if num_tx_24h[i] > 10:
            fraud_prob += 0.1
        if data_types[i] == 'UPI' and amounts[i] > 50000:
            fraud_prob += 0.05
            
        # Determine fraud status based on accumulated probability
        if random.random() < fraud_prob:
            is_fraud[i] = 1
            
        # Refine amount/balance for fraud cases
        if is_fraud[i] == 1:
            # Fraud patterns:
            # 1. Drain account
            if random.random() < 0.3:
                amounts[i] = old_balances[i] * 0.99
            # 2. Large outlier amount
            elif random.random() < 0.3:
                amounts[i] = np.random.uniform(100000, 1000000)
            
            # Adjust location/device for confirmed fraud to make pattern stronger for model to learn
            if random.random() < 0.7:
                location_consistency[i] = 0
            if random.random() < 0.6:
                device_status[i] = 0
                
    # Calculate New Balance
    # For some transactions, balance decreases (payments), for others it might stay same (failed?) or increase (refunds - rare here)
    # We assume standard payment scenario: Old - Amount = New
    new_balances = old_balances - amounts
    
    # Fix negative balances (allow overdraft only if realistic, else cap at 0 for simplicity or assume rejected)
    # Here we assume successful transactions mostly.
    # If amount > old_balance, it might be a credit card or overdraft.
    # Let's clean up logic: new balance is result of transaction.
    
    # Create DataFrame
    df = pd.DataFrame({
        'oldbalanceOrg': old_balances,
        'newbalanceOrig': new_balances,
        'amount': amounts,
        'balance_diff': old_balances - new_balances, # Engineered feature requested
        'type': data_types,
        'hour': hours,
        'num_tx_24h': num_tx_24h,
        'location_consistency': location_consistency,
        'device_status': device_status,
        'account_age_days': account_age,
        'isFraud': is_fraud
    })
    
    # Round monetary values
    df['oldbalanceOrg'] = df['oldbalanceOrg'].round(2)
    df['newbalanceOrig'] = df['newbalanceOrig'].round(2)
    df['amount'] = df['amount'].round(2)
    df['balance_diff'] = df['balance_diff'].round(2)
    
    print(f"Data generation complete.")
    print(f"Fraud samples: {df['isFraud'].sum()} ({df['isFraud'].mean()*100:.2f}%)")
    
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv('synthetic_fraud_data.csv', index=False)
    print("Saved to synthetic_fraud_data.csv")
