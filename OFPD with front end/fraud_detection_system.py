import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    Sequential = None
    Dense = None
    Dropout = None
    Adam = None

class FraudDetectionSystem:
    def __init__(self):
        self.preprocessor = None
        self.ml_model = None
        self.dl_model = None
        self.feature_names = None
        
    def _create_preprocessor(self, numeric_features, categorical_features):
        """
        Creates a preprocessing pipeline for numeric and categorical features.
        """
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        return self.preprocessor

    def preprocess_data(self, X, y=None, fit=False):
        """
        Applies preprocessing to the data.
        """
        if fit:
            X_processed = self.preprocessor.fit_transform(X)
            # Save feature names after encoding for interpretability if needed
            # (Simplification: getting feature names from OneHotEncoder can be tricky in older versions, skipping strict name mapping for now)
        else:
            X_processed = self.preprocessor.transform(X)
            
        return X_processed

    def train_ml_model(self, X_train, y_train, model_type='xgboost'):
        """
        Trains a Machine Learning model (XGBoost) with SMOTE handling.
        """
        print(f"Training ML Model ({model_type})...")
        
        # Define the pipeline with SMOTE
        # Note: SMOTE is applied only during training
        
        if model_type == 'xgboost':
            clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        else:
            raise ValueError("Model type not supported yet.")

        # ImbPipeline allows SMOTE to be part of the pipeline steps
        # However, since we already separated preprocessor (to handle DL inputs specially), 
        # we will apply transformation first, then SMOTE, then fit.
        # Ideally we stick to one pipeline, but for flexibility with DL, let's keep them somewhat decoupled 
        # or duplicate the preprocessor.
        
        # Let's trust the preprocessor state.
        X_train_transformed = self.preprocess_data(X_train, fit=True)
        
        print("Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train_transformed, y_train)
        
        print(f"Fitting {model_type} on resampled data...")
        self.ml_model = clf
        self.ml_model.fit(X_resampled, y_resampled)
        print("ML Training Complete.")

    def train_dl_model(self, X_train, y_train, epochs=10, batch_size=64):
        """
        Trains a Deep Learning model (ANN).
        """
        print("Training Deep Learning Model...")
        
        # Preprocess
        # Ensure preprocessor is fitted. If ML train ran first, it is. If not, this might fail if not careful.
        # Assumption: train_ml_model is called first or we check.
        try:
             X_train_transformed = self.preprocess_data(X_train, fit=False)
        except:
             # If not fitted yet
             X_train_transformed = self.preprocess_data(X_train, fit=True)
             
        # SMOTE for DL? Often useful.
        print("Applying SMOTE for DL...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train_transformed, y_train)
        
        input_dim = X_resampled.shape[1]
        
        model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        
        print("Fitting ANN...")
        model.fit(X_resampled, y_resampled, epochs=epochs, batch_size=batch_size, verbose=1)
        self.dl_model = model
        print("DL Training Complete.")

    def evaluate(self, X_test, y_test, model_type='ml'):
        """
        Evaluates the specified model.
        """
        X_test_transformed = self.preprocess_data(X_test, fit=False)
        
        if model_type == 'ml':
            y_pred = self.ml_model.predict(X_test_transformed)
            y_prob = self.ml_model.predict_proba(X_test_transformed)[:, 1]
        elif model_type == 'dl':
            y_prob = self.dl_model.predict(X_test_transformed).flatten()
            y_pred = (y_prob > 0.5).astype(int)
            
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob)
        }
        
        return metrics

    def predict(self, input_data, model_type='ml'):
        """
        Predicts fraud for new data. 
        input_data: pandas DataFrame
        """
        X_transformed = self.preprocess_data(input_data, fit=False)
        
        if model_type == 'ml':
            prob = self.ml_model.predict_proba(X_transformed)[:, 1]
        else:
            prob = self.dl_model.predict(X_transformed).flatten()
            
        return prob
