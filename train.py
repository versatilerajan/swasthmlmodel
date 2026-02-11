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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU')

np.random.seed(42)
tf.random.set_seed(42)

def generate_synthetic_health_data(n_samples=3000):
    print(f"Generating {n_samples} synthetic health records...")

    age = np.random.randint(18, 85, size=n_samples)
    gender = np.random.choice(['Male', 'Female'], size=n_samples)

    hemoglobin       = np.random.normal(14,  2,    size=n_samples).clip(8,   18)
    wbc              = np.random.normal(7500, 2000, size=n_samples).clip(4000, 15000)
    platelets        = np.random.normal(250000, 50000, size=n_samples).clip(150000, 450000)
    cholesterol      = np.random.normal(180, 40,   size=n_samples).clip(100, 300)
    ldl              = np.random.normal(110, 30,   size=n_samples).clip(50,  200)
    hdl              = np.random.normal(55,  15,   size=n_samples).clip(30,  90)
    triglycerides    = np.random.normal(130, 50,   size=n_samples).clip(40,  300)
    sgpt             = np.random.normal(25,  12,   size=n_samples).clip(5,   80)
    sgot             = np.random.normal(24,  10,   size=n_samples).clip(5,   70)
    creatinine       = np.random.normal(0.9, 0.25, size=n_samples).clip(0.4, 1.8)
    urea             = np.random.normal(28,  10,   size=n_samples).clip(10,  50)
    fasting_glucose  = np.random.normal(92,  18,   size=n_samples).clip(60,  180)
    hba1c            = np.random.normal(5.4, 0.9,  size=n_samples).clip(4.0, 9.0)
    tsh              = np.random.lognormal(0.4, 0.7, size=n_samples).clip(0.4, 8.0)
    vitamin_d        = np.random.normal(32,  14,   size=n_samples).clip(8,   70)
    vitamin_b12      = np.random.normal(420, 140,  size=n_samples).clip(180, 950)

    health_score = np.full(n_samples, 100.0, dtype=np.float32)

    health_score -= np.where((hemoglobin < 12) | (hemoglobin > 16), 12, 0)
    health_score -= np.where(cholesterol > 200,                     10, 0)
    health_score -= np.where(ldl > 130,                             12, 0)
    health_score -= np.where(hdl < 40,                              15, 0)
    health_score -= np.where(fasting_glucose > 100,                 15, 0)
    health_score -= np.where(hba1c > 5.7,                           20, 0)
    health_score -= np.where((sgpt > 40) | (sgot > 40),             10, 0)
    health_score -= np.where(creatinine > 1.2,                      18, 0)
    health_score -= np.where((tsh < 0.5) | (tsh > 5.0),             12, 0)
    health_score -= np.where(vitamin_d < 30,                        10, 0)

    health_score = np.clip(health_score, 0, 100)

    risk_level = np.select(
        [health_score >= 80, health_score >= 60, health_score >= 40],
        ['Low Risk', 'Moderate Risk', 'High Risk'],
        default='Critical'
    )

    df = pd.DataFrame({
        'age': age, 'gender': gender, 'hemoglobin': hemoglobin, 'wbc': wbc,
        'platelets': platelets, 'cholesterol': cholesterol, 'ldl': ldl, 'hdl': hdl,
        'triglycerides': triglycerides, 'sgpt': sgpt, 'sgot': sgot,
        'creatinine': creatinine, 'urea': urea, 'fasting_glucose': fasting_glucose,
        'hba1c': hba1c, 'tsh': tsh, 'vitamin_d': vitamin_d, 'vitamin_b12': vitamin_b12,
        'health_score': health_score, 'risk_level': risk_level
    })

    print(f"Dataset shape: {df.shape}")
    return df

def create_health_score_model(input_dim):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_risk_classification_model(input_dim, num_classes):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.35),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_models():
    print("=== Starting model training ===")
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

    X = df[feature_cols].values.astype(np.float32)
    y_score = df['health_score'].values.astype(np.float32)
    y_risk  = df['risk_encoded'].values.astype(np.int32)

    X_train, _, y_score_train, _, y_risk_train, _ = train_test_split(
        X, y_score, y_risk, test_size=0.25, random_state=42
    )

    print("\n[3/5] Scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5)
    ]

    print("\n[4/5] Training health score model...")
    score_model = create_health_score_model(X_train_scaled.shape[1])
    score_model.fit(
        X_train_scaled, y_score_train,
        validation_split=0.2,
        epochs=60,
        batch_size=512,
        callbacks=callbacks,
        verbose=1
    )

    print("\n[5/5] Training risk model...")
    num_classes = len(risk_encoder.classes_)
    risk_model = create_risk_classification_model(X_train_scaled.shape[1], num_classes)
    risk_model.fit(
        X_train_scaled, y_risk_train,
        validation_split=0.2,
        epochs=50,
        batch_size=512,
        callbacks=callbacks,
        verbose=1
    )

    print("\nSaving models...")
    score_model.save('models/health_score_model.keras')
    risk_model.save('models/risk_classification_model.keras')
    joblib.dump(scaler,         'models/scaler.pkl')
    joblib.dump(gender_encoder, 'models/gender_encoder.pkl')
    joblib.dump(risk_encoder,   'models/risk_encoder.pkl')

    with open('models/model_metadata.json', 'w') as f:
        json.dump({
            'feature_columns': feature_cols,
            'risk_classes': risk_encoder.classes_.tolist(),
            'model_version': '2025-02-fixed'
        }, f, indent=2)

    print("\nTraining finished.")
    os.system('ls -la models/')

if __name__ == "__main__":
    train_models()
