import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import json
import os
import warnings

warnings.filterwarnings('ignore')

# Force lower memory usage
tf.config.set_visible_devices([], 'GPU')  # disable GPU even if present
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress most TF logs

# Set smaller seeds and control randomness
np.random.seed(42)
tf.random.set_seed(42)

def generate_synthetic_health_data(n_samples=3000):  # ← reduced from 10000
    """Generate smaller synthetic dataset to save memory"""
    print(f"Generating {n_samples} synthetic health records...")

    data = []
    for i in range(n_samples):
        age = np.random.randint(18, 85)
        gender = np.random.choice(['Male', 'Female'])

        hemoglobin     = np.random.normal(14, 2).clip(8, 18)
        wbc            = np.random.normal(7500, 2000).clip(4000, 15000)
        platelets      = np.random.normal(250000, 50000).clip(150000, 450000)
        cholesterol    = np.random.normal(180, 40).clip(100, 300)
        ldl            = np.random.normal(110, 30).clip(50, 200)
        hdl            = np.random.normal(55, 15).clip(30, 90)
        triglycerides  = np.random.normal(130, 50).clip(40, 300)
        sgpt           = np.random.normal(25, 12).clip(5, 80)
        sgot           = np.random.normal(24, 10).clip(5, 70)
        creatinine     = np.random.normal(0.9, 0.25).clip(0.4, 1.8)
        urea           = np.random.normal(28, 10).clip(10, 50)
        fasting_glucose= np.random.normal(92, 18).clip(60, 180)
        hba1c          = np.random.normal(5.4, 0.9).clip(4.0, 9.0)
        tsh            = np.random.lognormal(0.4, 0.7).clip(0.4, 8.0)
        vitamin_d      = np.random.normal(32, 14).clip(8, 70)
        vitamin_b12    = np.random.normal(420, 140).clip(180, 950)

        # Very simple health score (faster to compute)
        health_score = 100
        if hemoglobin < 12 or hemoglobin > 16:    health_score -= 12
        if cholesterol > 200:                     health_score -= 10
        if ldl > 130:                             health_score -= 12
        if hdl < 40:                              health_score -= 15
        if fasting_glucose > 100:                 health_score -= 15
        if hba1c > 5.7:                           health_score -= 20
        if sgpt > 40 or sgot > 40:                health_score -= 10
        if creatinine > 1.2:                      health_score -= 18
        if tsh < 0.5 or tsh > 5.0:                health_score -= 12
        if vitamin_d < 30:                        health_score -= 10

        health_score = np.clip(health_score, 0, 100)

        # Risk level
        if health_score >= 80:    risk_level = 'Low Risk'
        elif health_score >= 60:  risk_level = 'Moderate Risk'
        elif health_score >= 40:  risk_level = 'High Risk'
        else:                     risk_level = 'Critical'

        data.append({
            'age': age, 'gender': gender,
            'hemoglobin': hemoglobin, 'wbc': wbc, 'platelets': platelets,
            'cholesterol': cholesterol, 'ldl': ldl, 'hdl': hdl, 'triglycerides': triglycerides,
            'sgpt': sgpt, 'sgot': sgot, 'creatinine': creatinine, 'urea': urea,
            'fasting_glucose': fasting_glucose, 'hba1c': hba1c, 'tsh': tsh,
            'vitamin_d': vitamin_d, 'vitamin_b12': vitamin_b12,
            'health_score': health_score, 'risk_level': risk_level
        })

    df = pd.DataFrame(data)
    print(f"Generated dataset shape: {df.shape}")
    return df

def create_health_score_model(input_dim):
    """Smaller model → less memory"""
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),  # ← was 256
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),                             # ← was 128
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_risk_classification_model(input_dim, num_classes):
    """Smaller classification model"""
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),  # ← was 256
        layers.BatchNormalization(),
        layers.Dropout(0.35),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_models():
    print("=" * 70)
    print("HEALTH MODEL TRAINING (memory-optimized version)")
    print("=" * 70)

    os.makedirs('models', exist_ok=True)

    print("\n[1/5] Generating data...")
    df = generate_synthetic_health_data(n_samples=3000)

    print("\n[2/5] Preparing features...")
    gender_encoder = LabelEncoder()
    df['gender_encoded'] = gender_encoder.fit_transform(df['gender'])

    risk_encoder = LabelEncoder()
    df['risk_encoded'] = risk_encoder.fit_transform(df['risk_level'])

    feature_cols = [
        'age', 'gender_encoded', 'hemoglobin', 'wbc', 'platelets',
        'cholesterol', 'ldl', 'hdl', 'triglycerides',
        'sgpt', 'sgot', 'creatinine', 'urea',
        'fasting_glucose', 'hba1c', 'tsh', 'vitamin_d', 'vitamin_b12'
    ]

    X = df[feature_cols].values
    y_score = df['health_score'].values
    y_risk  = df['risk_encoded'].values

    X_train, X_test, y_score_train, y_score_test, y_risk_train, y_risk_test = train_test_split(
        X, y_score, y_risk, test_size=0.25, random_state=42
    )

    print("\n[3/5] Scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ────────────────────────────────────────────────
    # Health score model
    # ────────────────────────────────────────────────
    print("\n[4/5] Training health score model...")
    score_model = create_health_score_model(input_dim=X_train_scaled.shape[1])

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
    reduce_lr  = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

    score_model.fit(
        X_train_scaled, y_score_train,
        validation_split=0.2,
        epochs=60,                # ← reduced
        batch_size=512,           # ← much larger batch = less memory spikes
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # ────────────────────────────────────────────────
    # Risk classification model
    # ────────────────────────────────────────────────
    print("\n[5/5] Training risk classification model...")
    num_classes = len(risk_encoder.classes_)
    risk_model = create_risk_classification_model(X_train_scaled.shape[1], num_classes)

    risk_model.fit(
        X_train_scaled, y_risk_train,
        validation_split=0.2,
        epochs=50,
        batch_size=512,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Save everything
    print("\nSaving models and artifacts...")
    score_model.save('models/health_score_model.keras')
    risk_model.save('models/risk_classification_model.keras')

    joblib.dump(scaler,         'models/scaler.pkl')
    joblib.dump(gender_encoder, 'models/gender_encoder.pkl')
    joblib.dump(risk_encoder,   'models/risk_encoder.pkl')

    # Minimal metadata
    metadata = {
        'feature_columns': feature_cols,
        'risk_classes': risk_encoder.classes_.tolist(),
        'model_version': '0.1-light',
        'training_samples': len(df)
    }
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\nTraining finished. Models saved in /models/")
    print("Files created:")
    for root, _, files in os.walk('models'):
        for f in files:
            print(f"  - {os.path.join(root, f)}")

if __name__ == "__main__":
    train_models()
