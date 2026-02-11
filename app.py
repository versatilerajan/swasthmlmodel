from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
import json
import os
import re
from datetime import datetime
import traceback
import warnings
warnings.filterwarnings('ignore')

# PDF and Image processing
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import io
import cv2

# OCR
try:
    import pytesseract
except:
    pytesseract = None

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['JSON_SORT_KEYS'] = False

# Global variables
health_score_model = None
risk_model = None
scaler = None
gender_encoder = None
risk_encoder = None
metadata = {}
report_templates = {}
models_loaded = False

def load_models():
    """Load all models and artifacts on startup"""
    global health_score_model, risk_model, scaler, gender_encoder, risk_encoder
    global metadata, report_templates, models_loaded
    
    try:
        print("=" * 60)
        print("LOADING HEALTH ANALYSIS MODELS...")
        print("=" * 60)
        
        if not os.path.exists('models'):
            print("ERROR: models directory not found")
            return False
        
        print("Models directory contents:")
        for f in os.listdir('models'):
            print(f" - {f}")
        
        health_score_model = keras.models.load_model('models/health_score_model.keras')
        print("✓ Health score model loaded")
        
        risk_model = keras.models.load_model('models/risk_classification_model.keras')
        print("✓ Risk classification model loaded")
        
        scaler = joblib.load('models/scaler.pkl')
        print("✓ Feature scaler loaded")
        
        gender_encoder = joblib.load('models/gender_encoder.pkl')
        print("✓ Gender encoder loaded")
        
        risk_encoder = joblib.load('models/risk_encoder.pkl')
        print("✓ Risk encoder loaded")
        
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        print("✓ Metadata loaded")
        
        with open('models/report_templates.json', 'r') as f:
            report_templates = json.load(f)
        print("✓ Report templates loaded")
        
        print("=" * 60)
        print("ALL MODELS LOADED SUCCESSFULLY")
        print("=" * 60)
        
        models_loaded = True
        return True
        
    except Exception as e:
        print(f"ERROR loading models: {e}")
        print(traceback.format_exc())
        models_loaded = False
        return False

# Load models on startup
print("Initializing Health Report Analyzer...")
load_models()

# ==================== PDF/IMAGE PROCESSING ====================

def extract_text_from_pdf_pymupdf(file_bytes):
    """Extract text from PDF using PyMuPDF"""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"PyMuPDF extraction failed: {e}")
        return None

def extract_text_from_pdf_pdfplumber(file_bytes):
    """Extract text from PDF using pdfplumber"""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        print(f"pdfplumber extraction failed: {e}")
        return None

def extract_text_from_image(image_bytes):
    """Extract text from image using OCR"""
    if pytesseract is None:
        return None
    
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array for OpenCV processing
        img_array = np.array(image)
        
        # Preprocessing for better OCR
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR
        text = pytesseract.image_to_string(thresh)
        return text
    except Exception as e:
        print(f"OCR extraction failed: {e}")
        return None

def process_uploaded_file(file):
    """Process uploaded file (PDF or image) and extract text"""
    file_bytes = file.read()
    filename = file.filename.lower()
    
    extracted_text = None
    
    # Try PDF extraction
    if filename.endswith('.pdf'):
        print("Processing PDF file...")
        
        # Try PyMuPDF first
        extracted_text = extract_text_from_pdf_pymupdf(file_bytes)
        
        # Fallback to pdfplumber
        if not extracted_text or len(extracted_text.strip()) < 50:
            extracted_text = extract_text_from_pdf_pdfplumber(file_bytes)
    
    # Try image extraction
    elif any(filename.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']):
        print("Processing image file...")
        extracted_text = extract_text_from_image(file_bytes)
    
    return extracted_text

# ==================== TEXT ANALYSIS ====================

def extract_numeric_value(text, pattern):
    """Extract numeric value from text using regex"""
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            value = match.group(1).replace(',', '')
            return float(value)
        except:
            return None
    return None

def parse_health_report(text):
    """Parse health report text and extract test values"""
    if not text:
        return None
    
    text = text.lower()
    
    extracted_values = {}
    
    # Hemoglobin patterns
    hb_patterns = [
        r'hemoglobin[:\s]+([0-9.]+)',
        r'hb[:\s]+([0-9.]+)',
        r'haemoglobin[:\s]+([0-9.]+)'
    ]
    for pattern in hb_patterns:
        val = extract_numeric_value(text, pattern)
        if val and 5 <= val <= 20:
            extracted_values['hemoglobin'] = val
            break
    
    # WBC
    wbc_patterns = [
        r'wbc[:\s]+([0-9,]+)',
        r'white blood cell[:\s]+([0-9,]+)',
        r'leucocyte[:\s]+([0-9,]+)'
    ]
    for pattern in wbc_patterns:
        val = extract_numeric_value(text, pattern)
        if val and 1000 <= val <= 50000:
            extracted_values['wbc'] = val
            break
    
    # Platelets
    platelet_patterns = [
        r'platelet[:\s]+([0-9,]+)',
        r'plt[:\s]+([0-9,]+)'
    ]
    for pattern in platelet_patterns:
        val = extract_numeric_value(text, pattern)
        if val and 50000 <= val <= 1000000:
            extracted_values['platelets'] = val
            break
    
    # Cholesterol
    chol_patterns = [
        r'total cholesterol[:\s]+([0-9.]+)',
        r'cholesterol[:\s]+([0-9.]+)',
        r't\.? ?cholesterol[:\s]+([0-9.]+)'
    ]
    for pattern in chol_patterns:
        val = extract_numeric_value(text, pattern)
        if val and 100 <= val <= 400:
            extracted_values['cholesterol'] = val
            break
    
    # LDL
    ldl_patterns = [
        r'ldl[:\s]+([0-9.]+)',
        r'ldl cholesterol[:\s]+([0-9.]+)',
        r'low density lipoprotein[:\s]+([0-9.]+)'
    ]
    for pattern in ldl_patterns:
        val = extract_numeric_value(text, pattern)
        if val and 20 <= val <= 300:
            extracted_values['ldl'] = val
            break
    
    # HDL
    hdl_patterns = [
        r'hdl[:\s]+([0-9.]+)',
        r'hdl cholesterol[:\s]+([0-9.]+)',
        r'high density lipoprotein[:\s]+([0-9.]+)'
    ]
    for pattern in hdl_patterns:
        val = extract_numeric_value(text, pattern)
        if val and 20 <= val <= 150:
            extracted_values['hdl'] = val
            break
    
    # Triglycerides
    trig_patterns = [
        r'triglyceride[s]?[:\s]+([0-9.]+)',
        r'tg[:\s]+([0-9.]+)'
    ]
    for pattern in trig_patterns:
        val = extract_numeric_value(text, pattern)
        if val and 30 <= val <= 500:
            extracted_values['triglycerides'] = val
            break
    
    # SGPT/ALT
    sgpt_patterns = [
        r'sgpt[:\s]+([0-9.]+)',
        r'alt[:\s]+([0-9.]+)',
        r'alanine aminotransferase[:\s]+([0-9.]+)'
    ]
    for pattern in sgpt_patterns:
        val = extract_numeric_value(text, pattern)
        if val and 5 <= val <= 200:
            extracted_values['sgpt'] = val
            break
    
    # SGOT/AST
    sgot_patterns = [
        r'sgot[:\s]+([0-9.]+)',
        r'ast[:\s]+([0-9.]+)',
        r'aspartate aminotransferase[:\s]+([0-9.]+)'
    ]
    for pattern in sgot_patterns:
        val = extract_numeric_value(text, pattern)
        if val and 5 <= val <= 200:
            extracted_values['sgot'] = val
            break
    
    # Creatinine
    creat_patterns = [
        r'creatinine[:\s]+([0-9.]+)',
        r'serum creatinine[:\s]+([0-9.]+)'
    ]
    for pattern in creat_patterns:
        val = extract_numeric_value(text, pattern)
        if val and 0.3 <= val <= 5:
            extracted_values['creatinine'] = val
            break
    
    # Urea/BUN
    urea_patterns = [
        r'urea[:\s]+([0-9.]+)',
        r'blood urea[:\s]+([0-9.]+)',
        r'bun[:\s]+([0-9.]+)'
    ]
    for pattern in urea_patterns:
        val = extract_numeric_value(text, pattern)
        if val and 5 <= val <= 100:
            extracted_values['urea'] = val
            break
    
    # Fasting Glucose
    glucose_patterns = [
        r'fasting glucose[:\s]+([0-9.]+)',
        r'fbs[:\s]+([0-9.]+)',
        r'fasting blood sugar[:\s]+([0-9.]+)',
        r'glucose[:\s]+([0-9.]+)'
    ]
    for pattern in glucose_patterns:
        val = extract_numeric_value(text, pattern)
        if val and 40 <= val <= 400:
            extracted_values['fasting_glucose'] = val
            break
    
    # HbA1c
    hba1c_patterns = [
        r'hba1c[:\s]+([0-9.]+)',
        r'glycated hemoglobin[:\s]+([0-9.]+)',
        r'glycosylated hemoglobin[:\s]+([0-9.]+)'
    ]
    for pattern in hba1c_patterns:
        val = extract_numeric_value(text, pattern)
        if val and 3 <= val <= 15:
            extracted_values['hba1c'] = val
            break
    
    # TSH
    tsh_patterns = [
        r'tsh[:\s]+([0-9.]+)',
        r'thyroid stimulating hormone[:\s]+([0-9.]+)'
    ]
    for pattern in tsh_patterns:
        val = extract_numeric_value(text, pattern)
        if val and 0.1 <= val <= 20:
            extracted_values['tsh'] = val
            break
    
    # Vitamin D
    vitd_patterns = [
        r'vitamin d[:\s]+([0-9.]+)',
        r'25-oh vitamin d[:\s]+([0-9.]+)',
        r'vit\.? ?d[:\s]+([0-9.]+)'
    ]
    for pattern in vitd_patterns:
        val = extract_numeric_value(text, pattern)
        if val and 5 <= val <= 150:
            extracted_values['vitamin_d'] = val
            break
    
    # Vitamin B12
    vitb12_patterns = [
        r'vitamin b12[:\s]+([0-9.]+)',
        r'b12[:\s]+([0-9.]+)',
        r'cobalamin[:\s]+([0-9.]+)'
    ]
    for pattern in vitb12_patterns:
        val = extract_numeric_value(text, pattern)
        if val and 100 <= val <= 2000:
            extracted_values['vitamin_b12'] = val
            break
    
    # Extract age if present
    age_patterns = [
        r'age[:\s]+([0-9]+)',
        r'([0-9]+)\s*y(?:ears?)?(?:\s|$)',
        r'([0-9]+)\s*yrs?'
    ]
    for pattern in age_patterns:
        val = extract_numeric_value(text, pattern)
        if val and 1 <= val <= 120:
            extracted_values['age'] = int(val)
            break
    
    # Extract gender
    if 'male' in text and 'female' not in text:
        extracted_values['gender'] = 'Male'
    elif 'female' in text:
        extracted_values['gender'] = 'Female'
    
    return extracted_values if extracted_values else None

def fill_missing_values(data):
    """Fill missing values with normal/default values"""
    defaults = {
        'age': 35,
        'gender': 'Male',
        'hemoglobin': 14.0,
        'wbc': 7500,
        'platelets': 250000,
        'cholesterol': 180,
        'ldl': 100,
        'hdl': 50,
        'triglycerides': 120,
        'sgpt': 25,
        'sgot': 24,
        'creatinine': 1.0,
        'urea': 28,
        'fasting_glucose': 90,
        'hba1c': 5.4,
        'tsh': 2.5,
        'vitamin_d': 35,
        'vitamin_b12': 400
    }
    
    for key, value in defaults.items():
        if key not in data:
            data[key] = value
    
    return data

def prepare_features(data):
    """Prepare features for model prediction"""
    # Ensure gender is encoded
    if isinstance(data['gender'], str):
        try:
            gender_encoded = gender_encoder.transform([data['gender']])[0]
        except:
            gender_encoded = 0  # Default to Male
    else:
        gender_encoded = data['gender']
    
    features = np.array([[
        data['age'],
        gender_encoded,
        data['hemoglobin'],
        data['wbc'],
        data['platelets'],
        data['cholesterol'],
        data['ldl'],
        data['hdl'],
        data['triglycerides'],
        data['sgpt'],
        data['sgot'],
        data['creatinine'],
        data['urea'],
        data['fasting_glucose'],
        data['hba1c'],
        data['tsh'],
        data['vitamin_d'],
        data['vitamin_b12']
    ]])
    
    return features

def analyze_test_results(data):
    """Analyze test results and provide insights"""
    normal_ranges = metadata.get('normal_ranges', {})
    
    abnormal_tests = []
    warnings_list = []
    recommendations = []
    
    for test_name, test_data in normal_ranges.items():
        if test_name in data:
            value = data[test_name]
            min_val = test_data.get('min', float('-inf'))
            max_val = test_data.get('max', float('inf'))
            unit = test_data.get('unit', '')
            
            if value < min_val:
                severity = 'High' if value < min_val * 0.7 else 'Moderate'
                abnormal_tests.append({
                    'test': test_name.replace('_', ' ').title(),
                    'value': round(value, 2),
                    'normal_range': f"{min_val}-{max_val} {unit}",
                    'status': 'Low',
                    'severity': severity,
                    'deviation': round(((min_val - value) / min_val * 100), 1)
                })
            elif value > max_val:
                severity = 'High' if value > max_val * 1.3 else 'Moderate'
                abnormal_tests.append({
                    'test': test_name.replace('_', ' ').title(),
                    'value': round(value, 2),
                    'normal_range': f"{min_val}-{max_val} {unit}",
                    'status': 'High',
                    'severity': severity,
                    'deviation': round(((value - max_val) / max_val * 100), 1)
                })
    
    # Generate warnings
    if data.get('cholesterol', 0) > 240:
        warnings_list.append({
            'category': 'Cardiovascular Risk',
            'message': 'High cholesterol detected',
            'urgency': 'High'
        })
    
    if data.get('hba1c', 0) > 6.5:
        warnings_list.append({
            'category': 'Diabetes Risk',
            'message': 'HbA1c indicates diabetic range',
            'urgency': 'Critical'
        })
    elif data.get('hba1c', 0) > 5.7:
        warnings_list.append({
            'category': 'Diabetes Risk',
            'message': 'Pre-diabetic HbA1c levels',
            'urgency': 'Moderate'
        })
    
    if data.get('creatinine', 0) > 1.5:
        warnings_list.append({
            'category': 'Kidney Function',
            'message': 'Elevated creatinine levels',
            'urgency': 'High'
        })
    
    if data.get('sgpt', 0) > 60 or data.get('sgot', 0) > 60:
        warnings_list.append({
            'category': 'Liver Function',
            'message': 'Elevated liver enzymes',
            'urgency': 'Moderate'
        })
    
    # Generate recommendations
    if len(abnormal_tests) == 0:
        recommendations.append({
            'priority': 'Maintain',
            'action': 'Continue healthy lifestyle habits',
            'details': 'Your test results are within normal ranges. Maintain regular exercise, balanced diet, and annual check-ups.'
        })
    else:
        if any(t['test'].lower() in ['cholesterol', 'ldl', 'triglycerides'] for t in abnormal_tests):
            recommendations.append({
                'priority': 'High',
                'action': 'Consult a cardiologist',
                'details': 'Consider lipid-lowering medication, low-fat diet, and regular exercise.'
            })
        
        if any(t['test'].lower() in ['hba1c', 'fasting glucose'] for t in abnormal_tests):
            recommendations.append({
                'priority': 'High',
                'action': 'Consult an endocrinologist',
                'details': 'Blood sugar management needed. Consider diabetes screening and dietary modifications.'
            })
        
        if any(t['test'].lower() in ['vitamin d', 'vitamin b12'] for t in abnormal_tests):
            recommendations.append({
                'priority': 'Medium',
                'action': 'Vitamin supplementation',
                'details': 'Consult your doctor for appropriate supplementation and dietary changes.'
            })
    
    return {
        'abnormal_tests': abnormal_tests,
        'warnings': warnings_list,
        'recommendations': recommendations
    }

# ==================== API ENDPOINTS ====================

@app.route('/')
def home():
    return jsonify({
        'service': 'Health Report Analyzer',
        'version': '1.0.0',
        'status': 'online' if models_loaded else 'models not loaded',
        'description': 'AI-powered health report analysis with OCR support',
        'capabilities': [
            'PDF report processing (PyMuPDF, pdfplumber)',
            'Image OCR (Tesseract)',
            'Automatic test value extraction',
            'Health score prediction',
            'Risk assessment',
            'Personalized recommendations',
            'Abnormal test detection'
        ],
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/analyze': 'POST - Upload and analyze health report (PDF/Image/JSON)',
            '/analyze/manual': 'POST - Manual test values input (JSON)'
        },
        'supported_formats': ['PDF', 'PNG', 'JPG', 'JPEG', 'JSON'],
        'max_file_size': '50MB'
    })

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy' if models_loaded else 'degraded',
        'models_loaded': models_loaded,
        'timestamp': datetime.now().isoformat(),
        'pytesseract_available': pytesseract is not None
    })

@app.route('/analyze', methods=['POST'])
def analyze_report():
    """Main analysis endpoint - handles file upload or JSON input"""
    
    if not models_loaded:
        return jsonify({
            'error': 'Models not loaded',
            'message': 'The analysis engine is currently unavailable'
        }), 500
    
    try:
        start_time = datetime.now()
        extracted_data = None
        source = None
        
        # Check if file was uploaded
        if 'file' in request.files:
            file = request.files['file']
            print(f"Processing uploaded file: {file.filename}")
            
            # Extract text from file
            extracted_text = process_uploaded_file(file)
            
            if not extracted_text:
                return jsonify({
                    'error': 'Text extraction failed',
                    'message': 'Could not extract text from the uploaded file'
                }), 400
            
            print(f"Extracted text length: {len(extracted_text)} characters")
            
            # Parse health data from text
            extracted_data = parse_health_report(extracted_text)
            source = 'file_upload'
            
            if not extracted_data:
                return jsonify({
                    'error': 'No health data found',
                    'message': 'Could not extract health test values from the document',
                    'extracted_text_preview': extracted_text[:500]
                }), 400
        
        # Check for JSON input
        elif request.is_json:
            extracted_data = request.get_json()
            source = 'json_input'
            print("Processing JSON input")
        
        else:
            return jsonify({
                'error': 'No input provided',
                'message': 'Please upload a file or send JSON data'
            }), 400
        
        # Fill missing values with defaults
        extracted_data = fill_missing_values(extracted_data)
        
        # Prepare features
        features = prepare_features(extracted_data)
        features_scaled = scaler.transform(features)
        
        # Predict health score
        health_score_pred = health_score_model.predict(features_scaled, verbose=0)[0][0]
        health_score = float(np.clip(health_score_pred, 0, 100))
        
        # Predict risk level
        risk_probs = risk_model.predict(features_scaled, verbose=0)[0]
        risk_class_idx = int(np.argmax(risk_probs))
        risk_level = risk_encoder.inverse_transform([risk_class_idx])[0]
        
        # Analyze test results
        analysis = analyze_test_results(extracted_data)
        
        # Determine status
        if health_score >= 80:
            status = 'Healthy'
            status_color = 'green'
        elif health_score >= 60:
            status = 'Attention Needed'
            status_color = 'yellow'
        elif health_score >= 40:
            status = 'Medical Consultation Required'
            status_color = 'orange'
        else:
            status = 'Urgent Medical Attention'
            status_color = 'red'
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Compile response
        response = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'processing_time_seconds': round(processing_time, 2),
                'source': source,
                'model_version': metadata.get('model_version', '1.0.0')
            },
            
            'health_summary': {
                'health_score': round(health_score, 2),
                'status': status,
                'status_color': status_color,
                'risk_level': risk_level,
                'risk_probabilities': {
                    risk_encoder.classes_[i]: round(float(risk_probs[i]) * 100, 2)
                    for i in range(len(risk_probs))
                }
            },
            
            'extracted_values': {
                k: round(v, 2) if isinstance(v, (int, float)) else v
                for k, v in extracted_data.items()
            },
            
            'detailed_analysis': {
                'abnormal_tests_count': len(analysis['abnormal_tests']),
                'abnormal_tests': analysis['abnormal_tests'],
                'warnings_count': len(analysis['warnings']),
                'warnings': analysis['warnings']
            },
            
            'recommendations': analysis['recommendations'],
            
            'next_steps': {
                'immediate': 'Review abnormal test results with your healthcare provider' if analysis['abnormal_tests'] else 'Maintain current health practices',
                'follow_up': 'Schedule follow-up tests in 3-6 months' if health_score < 70 else 'Annual health check-up recommended',
                'lifestyle': 'Continue healthy diet and regular exercise' if health_score >= 70 else 'Implement recommended lifestyle changes'
            }
        }
        
        print(f"✓ Analysis complete - Health Score: {health_score:.2f}, Risk: {risk_level}")
        
        return jsonify(response)
    
    except Exception as e:
        print(f"✗ Error during analysis: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': 'Analysis failed',
            'message': str(e),
            'type': type(e).__name__
        }), 500

@app.route('/analyze/manual', methods=['POST'])
def analyze_manual():
    """Manual input endpoint for test values"""
    
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Fill missing values
        data = fill_missing_values(data)
        
        # Prepare and predict
        features = prepare_features(data)
        features_scaled = scaler.transform(features)
        
        health_score = float(health_score_model.predict(features_scaled, verbose=0)[0][0])
        health_score = np.clip(health_score, 0, 100)
        
        risk_probs = risk_model.predict(features_scaled, verbose=0)[0]
        risk_class_idx = int(np.argmax(risk_probs))
        risk_level = risk_encoder.inverse_transform([risk_class_idx])[0]
        
        analysis = analyze_test_results(data)
        
        return jsonify({
            'health_score': round(health_score, 2),
            'risk_level': risk_level,
            'abnormal_tests': analysis['abnormal_tests'],
            'warnings': analysis['warnings'],
            'recommendations': analysis['recommendations']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'error': 'File too large',
        'message': 'Maximum file size is 50MB'
    }), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': ['/health', '/analyze', '/analyze/manual']
    }), 404

if __name__ == '__main__':
    if not models_loaded:
        print("\n" + "="*60)
        print("WARNING: Models failed to load!")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("Health Report Analyzer Ready!")
        print("="*60 + "\n")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
