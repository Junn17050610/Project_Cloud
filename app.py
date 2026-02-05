"""
Flask Backend API - Cloud Classification with Multi-Class Explanation
Dengan 2-Stage Validation: Sky Detection → Cloud Classification
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import json
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ============================================================================
# KONFIGURASI
# ============================================================================
class Config:
    # Path models - EDIT SESUAI PATH ANDA
    # Stage 1: Sky Detector (langit vs bukan langit)
    SKY_DETECTOR_PATH = 'models_skyimage_2/skyimage_model_20251222_133555.h5'
    
    # Stage 2: Cloud Classification
    CLOUD_MODEL_PATH = 'models_cloud_Similarity_2/model_cloud_classification_20260205_084714.h5'
    CLOUD_METADATA_PATH = 'results_cloud_Similarity_2/model_metadata_20260205_084714.json'
    
    IMG_SIZE = (224, 224)
    
    # Threshold untuk sky detector
    SKY_CONFIDENCE_THRESHOLD = 0.7  # Minimal 70% yakin ini gambar langit
    
    # Mapping class ke individual cloud types
    CLASS_MAPPING = {
        '2_altocumulus_cirrocumulus': {
            'types': ['Altocumulus', 'Cirrocumulus'],
            'altitude': 'Medium to High (2-7 km)',
            'weather': 'Cuaca cerah hingga berawan, kemungkinan hujan ringan dalam 12-24 jam',
            'icon': '⛅'
        },
        '3_cirrus_cirrostratus': {
            'types': ['Cirrus', 'Cirrostratus'],
            'altitude': 'High (5-13 km)',
            'weather': 'Cuaca baik, mungkin berubah dalam 24-48 jam',
            'icon': '🌤️'
        },
        '4_clearsky': {
            'types': ['Clear Sky'],
            'altitude': 'N/A',
            'weather': 'Cuaca cerah dan stabil',
            'icon': '☀️'
        },
        '5_stratocumulus_stratus_altostratus': {
            'types': ['Stratocumulus', 'Stratus', 'Altostratus'],
            'altitude': 'Low to Medium (0-4 km)',
            'weather': 'Cuaca mendung, kemungkinan hujan gerimis atau ringan',
            'icon': '☁️'
        },
        '6_cumulonimbus_nimbostratus': {
            'types': ['Cumulonimbus', 'Nimbostratus'],
            'altitude': 'Low to High (0-13 km)',
            'weather': '⚠️ Hujan lebat, petir, dan badai kemungkinan besar',
            'icon': '⛈️'
        },
        '7_contrail': {
            'types': ['Contrail'],
            'altitude': 'Very High (8-12 km)',
            'weather': 'Jejak pesawat - indikator kelembaban tinggi',
            'icon': '✈️'
        }
    }

# ============================================================================
# LOAD MODELS
# ============================================================================
print("=" * 80)
print("CLOUD CLASSIFICATION API - 2-STAGE VALIDATION")
print("=" * 80)

# ========== STAGE 1: SKY DETECTOR ==========
sky_detector = None
sky_classes = None

if os.path.exists(Config.SKY_DETECTOR_PATH):
    try:
        print(f"\n[Stage 1] Loading Sky Detector: {Config.SKY_DETECTOR_PATH}")
        sky_detector = keras.models.load_model(Config.SKY_DETECTOR_PATH)
        print("✓ Sky Detector loaded successfully")
        
        # Asumsi class order: ['bukan_langit', 'langit'] atau ['langit', 'bukan_langit']
        # PENTING: Sesuaikan dengan urutan class saat training sky detector!
        sky_classes = {
            0: 'bukan_langit',  # Index 0
            1: 'langit'         # Index 1
        }
        print(f"  Sky classes: {sky_classes}")
        
    except Exception as e:
        print(f"⚠ Warning: Sky Detector gagal dimuat: {str(e)}")
        print("  System akan skip validasi langit")
else:
    print(f"\n⚠ Warning: Sky Detector tidak ditemukan: {Config.SKY_DETECTOR_PATH}")
    print("  System akan skip validasi langit")

# ========== STAGE 2: CLOUD CLASSIFIER ==========
cloud_model = None
metadata = None

try:
    print(f"\n[Stage 2] Loading Cloud Model: {Config.CLOUD_MODEL_PATH}")
    cloud_model = keras.models.load_model(Config.CLOUD_MODEL_PATH)
    print("✓ Cloud Model loaded successfully")
    
    if os.path.exists(Config.CLOUD_METADATA_PATH):
        with open(Config.CLOUD_METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        print(f"✓ Metadata loaded - Accuracy: {metadata['performance']['accuracy']*100:.2f}%")
        Config.CLASSES = metadata.get('classes', list(Config.CLASS_MAPPING.keys()))
        print(f"  Cloud classes: {len(Config.CLASSES)} types")
    else:
        Config.CLASSES = list(Config.CLASS_MAPPING.keys())
        print("⚠ Metadata not found, using default classes")
        
except Exception as e:
    print(f"✗ Error loading Cloud Model: {str(e)}")

print("=" * 80)
print(f"Status:")
print(f"  - Sky Detector: {'ACTIVE ✓' if sky_detector else 'INACTIVE ✗'}")
print(f"  - Cloud Model: {'ACTIVE ✓' if cloud_model else 'INACTIVE ✗'}")
print("=" * 80)

# ============================================================================
# FUNCTIONS
# ============================================================================

def preprocess_image(image_bytes):
    """Preprocess image"""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(Config.IMG_SIZE)
    img_array = np.array(img).astype('float32') / 255.0
    return np.expand_dims(img_array, axis=0)

def generate_explanation(predicted_class):
    """Generate user-friendly explanation"""
    info = Config.CLASS_MAPPING.get(predicted_class, {
        'types': ['Unknown'],
        'altitude': 'Unknown',
        'weather': 'Unknown',
        'icon': '❓'
    })
    
    types = info['types']
    num = len(types)
    
    if num == 1:
        explanation = f"Gambar ini menunjukkan <strong>{types[0]}</strong>."
    elif num == 2:
        explanation = f"Dari gambar ini, terdapat <strong>2 kemungkinan</strong>: <strong>{types[0]}</strong> atau <strong>{types[1]}</strong>."
    elif num == 3:
        explanation = f"Dari gambar ini, terdapat <strong>3 kemungkinan</strong>: <strong>{types[0]}</strong>, <strong>{types[1]}</strong>, atau <strong>{types[2]}</strong>."
    else:
        types_str = ', '.join([f"<strong>{t}</strong>" for t in types])
        explanation = f"Dari gambar ini, terdapat {num} kemungkinan: {types_str}."
    
    return {
        'explanation': explanation,
        'cloud_types': types,
        'altitude': info['altitude'],
        'weather': info['weather'],
        'icon': info['icon']
    }

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'service': 'Cloud Classification API',
        'model_loaded': cloud_model is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    2-STAGE Cloud Classification:
    Stage 1: Validasi apakah gambar langit (Sky Detector)
    Stage 2: Klasifikasi jenis awan (Cloud Classifier)
    """
    # Check cloud model
    if cloud_model is None:
        return jsonify({
            'status': 'error',
            'message': 'Cloud model belum dimuat'
        }), 500
    
    # Check file
    if 'image' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'Tidak ada file gambar yang diupload'
        }), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'File gambar kosong'
        }), 400
    
    try:
        # Read and preprocess image
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)
        
        # ====================================================================
        # STAGE 1: SKY DETECTION (Validasi gambar langit)
        # ====================================================================
        
        is_sky = True  # Default jika sky detector tidak ada
        sky_confidence = 1.0
        sky_class_name = "unknown"
        
        if sky_detector is not None:
            print("\n[Stage 1] Running Sky Detection...")
            
            # Predict dengan sky detector
            sky_predictions = sky_detector.predict(processed_image, verbose=0)
            sky_class_idx = np.argmax(sky_predictions[0])
            sky_confidence = float(sky_predictions[0][sky_class_idx])
            sky_class_name = sky_classes.get(sky_class_idx, 'unknown')
            
            print(f"  Predicted class: {sky_class_name} (index: {sky_class_idx})")
            print(f"  Confidence: {sky_confidence*100:.2f}%")
            print(f"  Probabilities: langit={sky_predictions[0][1]*100:.2f}%, bukan_langit={sky_predictions[0][0]*100:.2f}%")
            
            # Check apakah gambar langit
            # Asumsi: index 1 = langit, index 0 = bukan_langit
            if sky_class_idx == 0:  # Bukan langit
                is_sky = False
                print(f"  ❌ Result: Bukan gambar langit!")
            elif sky_confidence < Config.SKY_CONFIDENCE_THRESHOLD:
                is_sky = False
                print(f"  ⚠️  Result: Confidence terlalu rendah ({sky_confidence*100:.2f}% < {Config.SKY_CONFIDENCE_THRESHOLD*100}%)")
            else:
                print(f"  ✓ Result: Gambar langit tervalidasi")
        else:
            print("\n[Stage 1] Sky Detector tidak aktif, skip validation")
        
        # Jika bukan gambar langit, return error dengan peringatan
        if not is_sky:
            return jsonify({
                'status': 'error',
                'error_type': 'not_sky_image',
                'message': 'Gambar yang diupload bukan gambar langit/awan',
                'detail': {
                    'is_sky': False,
                    'predicted_class': sky_class_name,
                    'sky_confidence': round(sky_confidence * 100, 2),
                    'threshold': Config.SKY_CONFIDENCE_THRESHOLD * 100,
                    'suggestion': 'Silakan upload gambar langit atau awan untuk klasifikasi',
                    'examples': [
                        '✓ Foto langit dengan awan',
                        '✓ Foto awan dari bawah',
                        '✓ Foto langit cerah',
                        '✗ Foto selfie, makanan, bangunan, dll'
                    ]
                }
            }), 400
        
        # ====================================================================
        # STAGE 2: CLOUD CLASSIFICATION
        # ====================================================================
        
        print("\n[Stage 2] Running Cloud Classification...")
        
        # Predict dengan cloud classifier
        predictions = cloud_model.predict(processed_image, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx]) * 100
        predicted_class = Config.CLASSES[predicted_idx]
        
        print(f"  Predicted class: {predicted_class}")
        print(f"  Confidence: {confidence:.2f}%")
        
        # Generate explanation
        explanation_data = generate_explanation(predicted_class)
        
        # Probabilities
        probs = {
            Config.CLASSES[i]: float(predictions[0][i]) * 100 
            for i in range(len(Config.CLASSES))
        }
        
        # ====================================================================
        # RESPONSE
        # ====================================================================
        
        return jsonify({
            'status': 'success',
            'data': {
                # Cloud classification results
                'prediction': predicted_class,
                'confidence': round(confidence, 2),
                'probabilities': probs,
                'cloud_info': explanation_data,
                
                # Sky validation info
                'validation': {
                    'is_sky_image': True,
                    'sky_confidence': round(sky_confidence * 100, 2),
                    'sky_detector_active': sky_detector is not None
                },
                
                'timestamp': datetime.now().isoformat()
            }
        })
    
    except Exception as e:
        print(f"\n✗ Error during prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error saat prediksi: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check dengan info kedua model"""
    return jsonify({
        'status': 'healthy',
        'models': {
            'sky_detector': {
                'status': 'loaded' if sky_detector is not None else 'not_loaded',
                'active': sky_detector is not None
            },
            'cloud_classifier': {
                'status': 'loaded' if cloud_model is not None else 'not_loaded',
                'active': cloud_model is not None
            }
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/web')
def web_ui():
    """Web UI"""
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)