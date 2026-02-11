import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def generate_synthetic_health_data(n_samples=10000):
    """Generate synthetic health report data for training"""
    print("Generating synthetic health data...")
    
    # Medical test categories
    test_types = [
        'Complete Blood Count', 'Lipid Profile', 'Liver Function Test',
        'Kidney Function Test', 'Thyroid Profile', 'Blood Sugar',
        'Vitamin D', 'Vitamin B12', 'Hemoglobin A1C', 'Urinalysis'
    ]
    
    # Generate test results with realistic ranges
    data = []
    
    for i in range(n_samples):
        # Patient demographics
        age = np.random.randint(18, 85)
        gender = np.random.choice(['Male', 'Female'])
        
        # Test results - normalized values
        hemoglobin = np.random.normal(14, 2).clip(8, 18)  # g/dL
        wbc = np.random.normal(7500, 2000).clip(4000, 15000)  # cells/µL
        platelets = np.random.normal(250000, 50000).clip(150000, 450000)
        
        # Lipid profile
        cholesterol = np.random.normal(200, 40).clip(150, 300)  # mg/dL
        ldl = np.random.normal(130, 30).clip(70, 200)
        hdl = np.random.normal(50, 15).clip(30, 80)
        triglycerides = np.random.normal(150, 50).clip(50, 300)
        
        # Liver function
        sgpt = np.random.normal(30, 15).clip(10, 100)  # U/L
        sgot = np.random.normal(28, 12).clip(10, 90)
        
        # Kidney function
        creatinine = np.random.normal(1.0, 0.3).clip(0.5, 2.0)  # mg/dL
        urea = np.random.normal(30, 10).clip(15, 50)
        
        # Diabetes markers
        fasting_glucose = np.random.normal(95, 20).clip(70, 200)  # mg/dL
        hba1c = np.random.normal(5.5, 1.0).clip(4.0, 10.0)  # %
        
        # Thyroid
        tsh = np.random.lognormal(0.5, 0.8).clip(0.5, 10.0)  # mIU/L
        
        # Vitamins
        vitamin_d = np.random.normal(30, 15).clip(10, 60)  # ng/mL
        vitamin_b12 = np.random.normal(400, 150).clip(200, 900)  # pg/mL
        
        # Calculate health score (0-100)
        # Lower is better for some, higher for others
        health_score = 100
        
        # Penalize abnormal values
        if hemoglobin < 12 or hemoglobin > 16:
            health_score -= abs(hemoglobin - 14) * 3
        
        if cholesterol > 200:
            health_score -= (cholesterol - 200) * 0.2
        
        if ldl > 130:
            health_score -= (ldl - 130) * 0.3
        
        if hdl < 40:
            health_score -= (40 - hdl) * 0.5
        
        if fasting_glucose > 100:
            health_score -= (fasting_glucose - 100) * 0.4
        
        if hba1c > 5.7:
            health_score -= (hba1c - 5.7) * 8
        
        if sgpt > 40 or sgot > 40:
            health_score -= max(sgpt - 40, sgot - 40) * 0.3
        
        if creatinine > 1.2:
            health_score -= (creatinine - 1.2) * 20
        
        if tsh < 0.5 or tsh > 5.0:
            health_score -= abs(tsh - 2.5) * 5
        
        if vitamin_d < 30:
            health_score -= (30 - vitamin_d) * 0.5
        
        health_score = max(0, min(100, health_score))
        
        # Risk categorization
        if health_score >= 80:
            risk_level = 'Low Risk'
            status = 'Healthy'
        elif health_score >= 60:
            risk_level = 'Moderate Risk'
            status = 'Attention Needed'
        elif health_score >= 40:
            risk_level = 'High Risk'
            status = 'Medical Consultation Required'
        else:
            risk_level = 'Critical'
            status = 'Urgent Medical Attention'
        
        data.append({
            'age': age,
            'gender': gender,
            'hemoglobin': hemoglobin,
            'wbc': wbc,
            'platelets': platelets,
            'cholesterol': cholesterol,
            'ldl': ldl,
            'hdl': hdl,
            'triglycerides': triglycerides,
            'sgpt': sgpt,
            'sgot': sgot,
            'creatinine': creatinine,
            'urea': urea,
            'fasting_glucose': fasting_glucose,
            'hba1c': hba1c,
            'tsh': tsh,
            'vitamin_d': vitamin_d,
            'vitamin_b12': vitamin_b12,
            'health_score': health_score,
            'risk_level': risk_level,
            'status': status
        })
    
    df = pd.DataFrame(data)
    print(f"Generated {len(df)} synthetic health records")
    print(f"Health score range: {df['health_score'].min():.2f} - {df['health_score'].max():.2f}")
    print(f"Risk distribution:\n{df['risk_level'].value_counts()}")
    
    return df

def create_health_score_model(input_dim):
    """Create deep learning model for health score prediction"""
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(1, activation='linear')  # Regression output
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
    )
    
    return model

def create_risk_classification_model(input_dim, num_classes):
    """Create model for risk level classification"""
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_models():
    """Main training function"""
    print("=" * 80)
    print("HEALTH REPORT ANALYZER - AI MODEL TRAINING")
    print("=" * 80)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Generate training data
    print("\n[1/6] Generating synthetic health data...")
    df = generate_synthetic_health_data(n_samples=10000)
    
    # Prepare features
    print("\n[2/6] Preparing features...")
    
    # Encode gender
    gender_encoder = LabelEncoder()
    df['gender_encoded'] = gender_encoder.fit_transform(df['gender'])
    
    # Risk level encoder
    risk_encoder = LabelEncoder()
    df['risk_encoded'] = risk_encoder.fit_transform(df['risk_level'])
    
    # Feature columns
    feature_cols = [
        'age', 'gender_encoded', 'hemoglobin', 'wbc', 'platelets',
        'cholesterol', 'ldl', 'hdl', 'triglycerides',
        'sgpt', 'sgot', 'creatinine', 'urea',
        'fasting_glucose', 'hba1c', 'tsh', 'vitamin_d', 'vitamin_b12'
    ]
    
    X = df[feature_cols].values
    y_score = df['health_score'].values
    y_risk = df['risk_encoded'].values
    
    # Split data
    X_train, X_test, y_score_train, y_score_test, y_risk_train, y_risk_test = train_test_split(
        X, y_score, y_risk, test_size=0.2, random_state=42
    )
    
    # Scale features
    print("\n[3/6] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train health score prediction model
    print("\n[4/6] Training health score prediction model...")
    score_model = create_health_score_model(input_dim=X_train_scaled.shape[1])
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001
    )
    
    history_score = score_model.fit(
        X_train_scaled, y_score_train,
        validation_split=0.2,
        epochs=100,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate score model
    score_pred = score_model.predict(X_test_scaled, verbose=0)
    score_mse = np.mean((score_pred.flatten() - y_score_test) ** 2)
    score_mae = np.mean(np.abs(score_pred.flatten() - y_score_test))
    score_r2 = 1 - (np.sum((y_score_test - score_pred.flatten()) ** 2) / 
                     np.sum((y_score_test - np.mean(y_score_test)) ** 2))
    
    print(f"\nHealth Score Model Performance:")
    print(f"  MSE: {score_mse:.4f}")
    print(f"  MAE: {score_mae:.4f}")
    print(f"  R² Score: {score_r2:.4f}")
    
    # Train risk classification model
    print("\n[5/6] Training risk classification model...")
    num_risk_classes = len(risk_encoder.classes_)
    risk_model = create_risk_classification_model(
        input_dim=X_train_scaled.shape[1],
        num_classes=num_risk_classes
    )
    
    history_risk = risk_model.fit(
        X_train_scaled, y_risk_train,
        validation_split=0.2,
        epochs=80,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate risk model
    risk_pred = risk_model.predict(X_test_scaled, verbose=0)
    risk_accuracy = np.mean(np.argmax(risk_pred, axis=1) == y_risk_test)
    
    print(f"\nRisk Classification Model Performance:")
    print(f"  Accuracy: {risk_accuracy:.4f}")
    
    # Save models
    print("\n[6/6] Saving models and artifacts...")
    
    score_model.save('models/health_score_model.keras')
    risk_model.save('models/risk_classification_model.keras')
    
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(gender_encoder, 'models/gender_encoder.pkl')
    joblib.dump(risk_encoder, 'models/risk_encoder.pkl')
    
    # Save metadata
    metadata = {
        'feature_columns': feature_cols,
        'risk_classes': risk_encoder.classes_.tolist(),
        'gender_classes': gender_encoder.classes_.tolist(),
        'model_version': '1.0.0',
        'training_samples': len(df),
        'score_model_performance': {
            'mse': float(score_mse),
            'mae': float(score_mae),
            'r2_score': float(score_r2)
        },
        'risk_model_performance': {
            'accuracy': float(risk_accuracy)
        },
        'normal_ranges': {
            'hemoglobin': {'min': 12.0, 'max': 16.0, 'unit': 'g/dL'},
            'wbc': {'min': 4000, 'max': 11000, 'unit': 'cells/µL'},
            'platelets': {'min': 150000, 'max': 450000, 'unit': 'cells/µL'},
            'cholesterol': {'min': 0, 'max': 200, 'unit': 'mg/dL'},
            'ldl': {'min': 0, 'max': 100, 'unit': 'mg/dL'},
            'hdl': {'min': 40, 'max': 100, 'unit': 'mg/dL'},
            'triglycerides': {'min': 0, 'max': 150, 'unit': 'mg/dL'},
            'sgpt': {'min': 0, 'max': 40, 'unit': 'U/L'},
            'sgot': {'min': 0, 'max': 40, 'unit': 'U/L'},
            'creatinine': {'min': 0.6, 'max': 1.2, 'unit': 'mg/dL'},
            'urea': {'min': 15, 'max': 40, 'unit': 'mg/dL'},
            'fasting_glucose': {'min': 70, 'max': 100, 'unit': 'mg/dL'},
            'hba1c': {'min': 4.0, 'max': 5.7, 'unit': '%'},
            'tsh': {'min': 0.5, 'max': 5.0, 'unit': 'mIU/L'},
            'vitamin_d': {'min': 30, 'max': 100, 'unit': 'ng/mL'},
            'vitamin_b12': {'min': 200, 'max': 900, 'unit': 'pg/mL'}
        }
    }
    
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # Save sample report templates
    report_templates = {
        'blood_test_keywords': [
            'hemoglobin', 'hb', 'wbc', 'white blood cell', 'platelets',
            'rbc', 'red blood cell', 'hematocrit', 'mcv', 'mch', 'mchc'
        ],
        'lipid_keywords': [
            'cholesterol', 'ldl', 'hdl', 'triglycerides', 'vldl',
            'lipid profile', 'total cholesterol'
        ],
        'liver_keywords': [
            'sgpt', 'sgot', 'alt', 'ast', 'alp', 'alkaline phosphatase',
            'bilirubin', 'liver function'
        ],
        'kidney_keywords': [
            'creatinine', 'urea', 'bun', 'uric acid', 'kidney function',
            'renal function'
        ],
        'diabetes_keywords': [
            'glucose', 'sugar', 'hba1c', 'glycated hemoglobin',
            'fasting blood sugar', 'fbs', 'ppbs', 'random blood sugar'
        ],
        'thyroid_keywords': [
            'tsh', 'thyroid', 't3', 't4', 'free t3', 'free t4'
        ],
        'vitamin_keywords': [
            'vitamin d', 'vitamin b12', 'vitamin b', 'folate', 'folic acid'
        ]
    }
    
    with open('models/report_templates.json', 'w') as f:
        json.dump(report_templates, f, indent=4)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print("Saved files:")
    print("  - models/health_score_model.keras")
    print("  - models/risk_classification_model.keras")
    print("  - models/scaler.pkl")
    print("  - models/gender_encoder.pkl")
    print("  - models/risk_encoder.pkl")
    print("  - models/model_metadata.json")
    print("  - models/report_templates.json")
    print("=" * 80)

if __name__ == "__main__":
    train_models()
