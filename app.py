# """
# app.py
# Flask Backend API - Cloud Classification with Multi-Class Explanation
# Dengan 2-Stage Validation: Sky Detection → Cloud Classification
# """

# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# from PIL import Image
# import io
# import json
# import os
# from datetime import datetime

# app = Flask(__name__)
# CORS(app)

# # ============================================================================
# # KONFIGURASI
# # ============================================================================
# class Config:
#     # Path models - EDIT SESUAI PATH ANDA
#     # Stage 1: Sky Detector (langit vs bukan langit)
#     SKY_DETECTOR_PATH = 'models_skyimage_2/skyimage_model_20251222_133555.h5'
    
#     # Stage 2: Cloud Classification
#     CLOUD_MODEL_PATH = 'models_cloud_Similarity_3/model_cloud_classification_20260205_193255.h5'
#     CLOUD_METADATA_PATH = 'results_cloud_Similarity_3/model_metadata_20260205_193255.json'
    
#     IMG_SIZE = (224, 224)
    
#     # Threshold untuk sky detector
#     SKY_CONFIDENCE_THRESHOLD = 0.7  # Minimal 70% yakin ini gambar langit
    
#     # Mapping class ke individual cloud types
#     CLASS_MAPPING = {
#         '2_altocumulus_cirrocumulus': {
#             'types': ['Altocumulus', 'Cirrocumulus'],
#             'altitude': 'Medium to High (2-7 km)',
#             'weather': 'Cuaca cerah hingga berawan, kemungkinan hujan ringan dalam 12-24 jam',
#             'icon': '⛅'
#         },
#         '3_cirrus_cirrostratus': {
#             'types': ['Cirrus', 'Cirrostratus'],
#             'altitude': 'High (5-13 km)',
#             'weather': 'Cuaca baik, mungkin berubah dalam 24-48 jam',
#             'icon': '🌤️'
#         },
#         '4_clearsky': {
#             'types': ['Clear Sky'],
#             'altitude': 'N/A',
#             'weather': 'Cuaca cerah dan stabil',
#             'icon': '☀️'
#         },
#         '5_stratocumulus_stratus_altostratus': {
#             'types': ['Stratocumulus', 'Stratus', 'Altostratus'],
#             'altitude': 'Low to Medium (0-4 km)',
#             'weather': 'Cuaca mendung, kemungkinan hujan gerimis atau ringan',
#             'icon': '☁️'
#         },
#         '6_cumulonimbus_nimbostratus': {
#             'types': ['Cumulonimbus', 'Nimbostratus'],
#             'altitude': 'Low to High (0-13 km)',
#             'weather': '⚠️ Hujan lebat, petir, dan badai kemungkinan besar',
#             'icon': '⛈️'
#         },
#         '7_contrail': {
#             'types': ['Contrail'],
#             'altitude': 'Very High (8-12 km)',
#             'weather': 'Jejak pesawat - indikator kelembaban tinggi',
#             'icon': '✈️'
#         }
#     }

# # ============================================================================
# # LOAD MODELS
# # ============================================================================
# print("=" * 80)
# print("CLOUD CLASSIFICATION API - 2-STAGE VALIDATION")
# print("=" * 80)

# # ========== STAGE 1: SKY DETECTOR ==========
# sky_detector = None
# sky_classes = None

# if os.path.exists(Config.SKY_DETECTOR_PATH):
#     try:
#         print(f"\n[Stage 1] Loading Sky Detector: {Config.SKY_DETECTOR_PATH}")
#         sky_detector = keras.models.load_model(Config.SKY_DETECTOR_PATH)
#         print("✓ Sky Detector loaded successfully")
        
#         # Asumsi class order: ['bukan_langit', 'langit'] atau ['langit', 'bukan_langit']
#         # PENTING: Sesuaikan dengan urutan class saat training sky detector!
#         sky_classes = {
#             0: 'bukan_langit',  # Index 0
#             1: 'langit'         # Index 1
#         }
#         print(f"  Sky classes: {sky_classes}")
        
#     except Exception as e:
#         print(f"⚠ Warning: Sky Detector gagal dimuat: {str(e)}")
#         print("  System akan skip validasi langit")
# else:
#     print(f"\n⚠ Warning: Sky Detector tidak ditemukan: {Config.SKY_DETECTOR_PATH}")
#     print("  System akan skip validasi langit")

# # ========== STAGE 2: CLOUD CLASSIFIER ==========
# cloud_model = None
# metadata = None

# try:
#     print(f"\n[Stage 2] Loading Cloud Model: {Config.CLOUD_MODEL_PATH}")
#     cloud_model = keras.models.load_model(Config.CLOUD_MODEL_PATH)
#     print("✓ Cloud Model loaded successfully")
    
#     if os.path.exists(Config.CLOUD_METADATA_PATH):
#         with open(Config.CLOUD_METADATA_PATH, 'r') as f:
#             metadata = json.load(f)
#         print(f"✓ Metadata loaded - Accuracy: {metadata['performance']['accuracy']*100:.2f}%")
#         Config.CLASSES = metadata.get('classes', list(Config.CLASS_MAPPING.keys()))
#         print(f"  Cloud classes: {len(Config.CLASSES)} types")
#     else:
#         Config.CLASSES = list(Config.CLASS_MAPPING.keys())
#         print("⚠ Metadata not found, using default classes")
        
# except Exception as e:
#     print(f"✗ Error loading Cloud Model: {str(e)}")

# print("=" * 80)
# print(f"Status:")
# print(f"  - Sky Detector: {'ACTIVE ✓' if sky_detector else 'INACTIVE ✗'}")
# print(f"  - Cloud Model: {'ACTIVE ✓' if cloud_model else 'INACTIVE ✗'}")
# print("=" * 80)

# # ============================================================================
# # FUNCTIONS
# # ============================================================================

# def preprocess_image(image_bytes):
#     """Preprocess image"""
#     img = Image.open(io.BytesIO(image_bytes))
#     if img.mode != 'RGB':
#         img = img.convert('RGB')
#     img = img.resize(Config.IMG_SIZE)
#     img_array = np.array(img).astype('float32') / 255.0
#     return np.expand_dims(img_array, axis=0)

# def generate_explanation(predicted_class):
#     """Generate user-friendly explanation"""
#     info = Config.CLASS_MAPPING.get(predicted_class, {
#         'types': ['Unknown'],
#         'altitude': 'Unknown',
#         'weather': 'Unknown',
#         'icon': '❓'
#     })
    
#     types = info['types']
#     num = len(types)
    
#     if num == 1:
#         explanation = f"Gambar ini menunjukkan <strong>{types[0]}</strong>."
#     elif num == 2:
#         explanation = f"Dari gambar ini, terdapat <strong>2 kemungkinan</strong>: <strong>{types[0]}</strong> atau <strong>{types[1]}</strong>."
#     elif num == 3:
#         explanation = f"Dari gambar ini, terdapat <strong>3 kemungkinan</strong>: <strong>{types[0]}</strong>, <strong>{types[1]}</strong>, atau <strong>{types[2]}</strong>."
#     else:
#         types_str = ', '.join([f"<strong>{t}</strong>" for t in types])
#         explanation = f"Dari gambar ini, terdapat {num} kemungkinan: {types_str}."
    
#     return {
#         'explanation': explanation,
#         'cloud_types': types,
#         'altitude': info['altitude'],
#         'weather': info['weather'],
#         'icon': info['icon']
#     }

# # ============================================================================
# # API ENDPOINTS
# # ============================================================================

# @app.route('/')
# def home():
#     return jsonify({
#         'status': 'online',
#         'service': 'Cloud Classification API',
#         'model_loaded': cloud_model is not None
#     })

# @app.route('/api/predict', methods=['POST'])
# def predict():
#     """
#     2-STAGE Cloud Classification:
#     Stage 1: Validasi apakah gambar langit (Sky Detector)
#     Stage 2: Klasifikasi jenis awan (Cloud Classifier)
#     """
#     # Check cloud model
#     if cloud_model is None:
#         return jsonify({
#             'status': 'error',
#             'message': 'Cloud model belum dimuat'
#         }), 500
    
#     # Check file
#     if 'image' not in request.files:
#         return jsonify({
#             'status': 'error',
#             'message': 'Tidak ada file gambar yang diupload'
#         }), 400
    
#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({
#             'status': 'error',
#             'message': 'File gambar kosong'
#         }), 400
    
#     try:
#         # Read and preprocess image
#         image_bytes = file.read()
#         processed_image = preprocess_image(image_bytes)
        
#         # ====================================================================
#         # STAGE 1: SKY DETECTION (Validasi gambar langit)
#         # ====================================================================
        
#         is_sky = True  # Default jika sky detector tidak ada
#         sky_confidence = 1.0
#         sky_class_name = "unknown"
        
#         if sky_detector is not None:
#             print("\n[Stage 1] Running Sky Detection...")
            
#             # Predict dengan sky detector
#             sky_predictions = sky_detector.predict(processed_image, verbose=0)
#             sky_class_idx = np.argmax(sky_predictions[0])
#             sky_confidence = float(sky_predictions[0][sky_class_idx])
#             sky_class_name = sky_classes.get(sky_class_idx, 'unknown')
            
#             print(f"  Predicted class: {sky_class_name} (index: {sky_class_idx})")
#             print(f"  Confidence: {sky_confidence*100:.2f}%")
#             print(f"  Probabilities: langit={sky_predictions[0][1]*100:.2f}%, bukan_langit={sky_predictions[0][0]*100:.2f}%")
            
#             # Check apakah gambar langit
#             # Asumsi: index 1 = langit, index 0 = bukan_langit
#             if sky_class_idx == 0:  # Bukan langit
#                 is_sky = False
#                 print(f"  ❌ Result: Bukan gambar langit!")
#             elif sky_confidence < Config.SKY_CONFIDENCE_THRESHOLD:
#                 is_sky = False
#                 print(f"  ⚠️  Result: Confidence terlalu rendah ({sky_confidence*100:.2f}% < {Config.SKY_CONFIDENCE_THRESHOLD*100}%)")
#             else:
#                 print(f"  ✓ Result: Gambar langit tervalidasi")
#         else:
#             print("\n[Stage 1] Sky Detector tidak aktif, skip validation")
        
#         # Jika bukan gambar langit, return error dengan peringatan
#         if not is_sky:
#             return jsonify({
#                 'status': 'error',
#                 'error_type': 'not_sky_image',
#                 'message': 'Gambar yang diupload bukan gambar langit/awan',
#                 'detail': {
#                     'is_sky': False,
#                     'predicted_class': sky_class_name,
#                     'sky_confidence': round(sky_confidence * 100, 2),
#                     'threshold': Config.SKY_CONFIDENCE_THRESHOLD * 100,
#                     'suggestion': 'Silakan upload gambar langit atau awan untuk klasifikasi',
#                     'examples': [
#                         '✓ Foto langit dengan awan',
#                         '✓ Foto awan dari bawah',
#                         '✓ Foto langit cerah',
#                         '✗ Foto selfie, makanan, bangunan, dll'
#                     ]
#                 }
#             }), 400
        
#         # ====================================================================
#         # STAGE 2: CLOUD CLASSIFICATION
#         # ====================================================================
        
#         print("\n[Stage 2] Running Cloud Classification...")
        
#         # Predict dengan cloud classifier
#         predictions = cloud_model.predict(processed_image, verbose=0)
#         predicted_idx = np.argmax(predictions[0])
#         confidence = float(predictions[0][predicted_idx]) * 100
#         predicted_class = Config.CLASSES[predicted_idx]
        
#         print(f"  Predicted class: {predicted_class}")
#         print(f"  Confidence: {confidence:.2f}%")
        
#         # Generate explanation
#         explanation_data = generate_explanation(predicted_class)
        
#         # Probabilities
#         probs = {
#             Config.CLASSES[i]: float(predictions[0][i]) * 100 
#             for i in range(len(Config.CLASSES))
#         }
        
#         # ====================================================================
#         # RESPONSE
#         # ====================================================================
        
#         return jsonify({
#             'status': 'success',
#             'data': {
#                 # Cloud classification results
#                 'prediction': predicted_class,
#                 'confidence': round(confidence, 2),
#                 'probabilities': probs,
#                 'cloud_info': explanation_data,
                
#                 # Sky validation info
#                 'validation': {
#                     'is_sky_image': True,
#                     'sky_confidence': round(sky_confidence * 100, 2),
#                     'sky_detector_active': sky_detector is not None
#                 },
                
#                 'timestamp': datetime.now().isoformat()
#             }
#         })
    
#     except Exception as e:
#         print(f"\n✗ Error during prediction: {str(e)}")
#         return jsonify({
#             'status': 'error',
#             'message': f'Error saat prediksi: {str(e)}'
#         }), 500

# @app.route('/api/health', methods=['GET'])
# def health_check():
#     """Health check dengan info kedua model"""
#     return jsonify({
#         'status': 'healthy',
#         'models': {
#             'sky_detector': {
#                 'status': 'loaded' if sky_detector is not None else 'not_loaded',
#                 'active': sky_detector is not None
#             },
#             'cloud_classifier': {
#                 'status': 'loaded' if cloud_model is not None else 'not_loaded',
#                 'active': cloud_model is not None
#             }
#         },
#         'timestamp': datetime.now().isoformat()
#     })

# @app.route('/web')
# def web_ui():
#     """Web UI"""
#     return render_template("index.html")

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)

"""
app.py - UPDATED WITH ADVANCED PREPROCESSING
Flask Backend API - Cloud Classification
Dengan preprocessing khusus untuk regular camera (handphone)
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2
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
    # Path models
    SKY_DETECTOR_PATH = 'models_skyimage_2/skyimage_model_20251222_133555.h5'
    CLOUD_MODEL_PATH = 'models_cloud_Similarity_Fisheye_4/model_cloud_classification_20260206_092744.h5'
    CLOUD_METADATA_PATH = 'results_cloud_Similarity_Fisheye_4/model_metadata_20260206_092744.json'
    
    IMG_SIZE = (224, 224)
    SKY_CONFIDENCE_THRESHOLD = 0.7
    
    # ✨ NEW: Preprocessing options
    APPLY_PREPROCESSING = True  # Enable preprocessing untuk regular camera
    PREPROCESSING_LEVEL = 'optimal'  # 'minimal', 'optimal', 'aggressive'
    
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
        # '7_contrail': {
        #     'types': ['Contrail'],
        #     'altitude': 'Very High (8-12 km)',
        #     'weather': 'Jejak pesawat - indikator kelembaban tinggi',
        #     'icon': '✈️'
        # }
    }

# ============================================================================
# LOAD MODELS
# ============================================================================
print("=" * 80)
print("CLOUD CLASSIFICATION API - WITH PREPROCESSING")
print("=" * 80)

# Load Sky Detector
sky_detector = None
sky_classes = None

if os.path.exists(Config.SKY_DETECTOR_PATH):
    try:
        print(f"\n[Stage 1] Loading Sky Detector: {Config.SKY_DETECTOR_PATH}")
        sky_detector = keras.models.load_model(Config.SKY_DETECTOR_PATH)
        print("✓ Sky Detector loaded")
        
        sky_classes = {
            0: 'bukan_langit',
            1: 'langit'
        }
        print(f"  Sky classes: {sky_classes}")
    except Exception as e:
        print(f"⚠ Warning: {str(e)}")
else:
    print(f"\n⚠ Sky Detector not found")

# Load Cloud Classifier
cloud_model = None
metadata = None

try:
    print(f"\n[Stage 2] Loading Cloud Model: {Config.CLOUD_MODEL_PATH}")
    cloud_model = keras.models.load_model(Config.CLOUD_MODEL_PATH)
    print("✓ Cloud Model loaded")
    
    if os.path.exists(Config.CLOUD_METADATA_PATH):
        with open(Config.CLOUD_METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        print(f"✓ Metadata loaded - Accuracy: {metadata['performance']['accuracy']*100:.2f}%")
        Config.CLASSES = metadata.get('classes', list(Config.CLASS_MAPPING.keys()))
    else:
        Config.CLASSES = list(Config.CLASS_MAPPING.keys())
except Exception as e:
    print(f"✗ Error: {str(e)}")

print("=" * 80)
print(f"Status:")
print(f"  - Sky Detector: {'ACTIVE ✓' if sky_detector else 'INACTIVE ✗'}")
print(f"  - Cloud Model: {'ACTIVE ✓' if cloud_model else 'INACTIVE ✗'}")
print(f"  - Preprocessing: {'ENABLED ✓' if Config.APPLY_PREPROCESSING else 'DISABLED'}")
print(f"  - Preprocessing Level: {Config.PREPROCESSING_LEVEL}")
print("=" * 80)

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def apply_clahe(image):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)
    Meningkatkan contrast lokal untuk better cloud detail
    """
    # Ensure uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # Merge back
    lab_clahe = cv2.merge([l_clahe, a, b])
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
    return enhanced


def white_balance(image):
    """
    Simple white balance correction
    Normalize warna untuk consistency
    """
    result = image.copy().astype(np.float32)
    
    for i in range(3):
        channel = result[:, :, i]
        min_val = np.percentile(channel, 1)
        max_val = np.percentile(channel, 99)
        result[:, :, i] = np.clip(
            (channel - min_val) * 255 / (max_val - min_val + 1e-6),
            0, 255
        )
    
    return result.astype(np.uint8)


def enhance_texture(image, sigma=1.0, amount=1.0):
    """
    Texture enhancement via unsharp masking
    Membuat pattern awan lebih jelas
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def remove_foreground(image):
    """
    Remove dark foreground objects (buildings, wires, etc)
    Fokus ke sky region
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]
    
    # Create mask: sky biasanya bright
    # Threshold adaptif berdasarkan image
    threshold = np.percentile(v_channel, 25)  # 25th percentile
    sky_mask = v_channel > max(threshold, 80)  # Minimal 80 brightness
    
    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    sky_mask = cv2.morphologyEx(sky_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)
    
    # Apply mask
    result = image.copy()
    # Set non-sky to light blue (blend with sky)
    result[sky_mask == 0] = [135, 206, 235]  # Sky blue
    
    return result


def preprocess_for_cloud_classification(image_bytes, level='optimal'):
    """
    Complete preprocessing pipeline
    
    Args:
        image_bytes: Raw image bytes from upload
        level: 'minimal', 'optimal', or 'aggressive'
    
    Returns:
        Preprocessed numpy array ready for model
    """
    # Load image
    img = Image.open(io.BytesIO(image_bytes))
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize
    img = img.resize(Config.IMG_SIZE)
    img_array = np.array(img)
    
    print(f"\n  [Preprocessing] Level: {level}")
    
    # Apply preprocessing based on level
    if level == 'minimal':
        print("    1/3 CLAHE...", end='')
        img_array = apply_clahe(img_array)
        print(" ✓")
    
    elif level == 'optimal':
        print("    1/5 Foreground removal...", end='')
        img_array = remove_foreground(img_array)
        print(" ✓")
        
        print("    2/5 CLAHE...", end='')
        img_array = apply_clahe(img_array)
        print(" ✓")
        
        print("    3/5 White balance...", end='')
        img_array = white_balance(img_array)
        print(" ✓")
        
        print("    4/5 Texture enhancement...", end='')
        img_array = enhance_texture(img_array, sigma=1.0, amount=1.0)
        print(" ✓")
        
        print("    5/5 Final CLAHE...", end='')
        img_array = apply_clahe(img_array)
        print(" ✓")
    
    elif level == 'aggressive':
        print("    1/6 Foreground removal...", end='')
        img_array = remove_foreground(img_array)
        print(" ✓")
        
        print("    2/6 White balance...", end='')
        img_array = white_balance(img_array)
        print(" ✓")
        
        print("    3/6 CLAHE (strong)...", end='')
        # Stronger CLAHE
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img_array = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
        print(" ✓")
        
        print("    4/6 Texture enhancement (strong)...", end='')
        img_array = enhance_texture(img_array, sigma=1.2, amount=1.5)
        print(" ✓")
        
        print("    5/6 Denoising...", end='')
        img_array = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
        print(" ✓")
        
        print("    6/6 Final CLAHE...", end='')
        img_array = apply_clahe(img_array)
        print(" ✓")
    
    print("  [Preprocessing] Complete!")
    
    # Normalize to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
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
        'preprocessing': Config.APPLY_PREPROCESSING,
        'model_loaded': cloud_model is not None
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """2-STAGE Cloud Classification dengan Preprocessing"""
    
    if cloud_model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'Empty file'}), 400
    
    try:
        image_bytes = file.read()
        
        # ====================================================================
        # STAGE 1: SKY DETECTION
        # ====================================================================
        
        is_sky = True
        sky_confidence = 1.0
        sky_class_name = "unknown"
        
        if sky_detector is not None:
            print("\n[Stage 1] Sky Detection...")
            
            # Preprocess untuk sky detector (minimal preprocessing)
            processed_for_sky = preprocess_for_cloud_classification(
                image_bytes, 
                level='minimal'  # Sky detector hanya perlu basic preprocessing
            )
            
            sky_predictions = sky_detector.predict(processed_for_sky, verbose=0)
            sky_class_idx = np.argmax(sky_predictions[0])
            sky_confidence = float(sky_predictions[0][sky_class_idx])
            sky_class_name = sky_classes.get(sky_class_idx, 'unknown')
            
            print(f"  Result: {sky_class_name} ({sky_confidence*100:.2f}%)")
            
            if sky_class_idx == 0:
                is_sky = False
                print("  ❌ Not sky image!")
            elif sky_confidence < Config.SKY_CONFIDENCE_THRESHOLD:
                is_sky = False
                print(f"  ⚠️  Low confidence")
            else:
                print("  ✓ Sky validated")
        
        if not is_sky:
            return jsonify({
                'status': 'error',
                'error_type': 'not_sky_image',
                'message': 'Gambar bukan langit/awan',
                'detail': {
                    'is_sky': False,
                    'predicted_class': sky_class_name,
                    'sky_confidence': round(sky_confidence * 100, 2),
                    'threshold': Config.SKY_CONFIDENCE_THRESHOLD * 100,
                    'suggestion': 'Upload gambar langit/awan',
                    'examples': [
                        '✓ Foto langit dengan awan',
                        '✓ Foto awan dari bawah',
                        '✓ Foto langit cerah',
                        '✗ Foto selfie, makanan, bangunan'
                    ]
                }
            }), 400
        
        # ====================================================================
        # STAGE 2: CLOUD CLASSIFICATION dengan PREPROCESSING
        # ====================================================================
        
        print("\n[Stage 2] Cloud Classification...")
        
        # Preprocess dengan level sesuai config
        processed_image = preprocess_for_cloud_classification(
            image_bytes,
            level=Config.PREPROCESSING_LEVEL
        )
        
        # Predict
        predictions = cloud_model.predict(processed_image, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx]) * 100
        predicted_class = Config.CLASSES[predicted_idx]
        
        print(f"  Result: {predicted_class} ({confidence:.2f}%)")
        
        # Generate explanation
        explanation_data = generate_explanation(predicted_class)
        
        # Probabilities
        probs = {
            Config.CLASSES[i]: float(predictions[0][i]) * 100
            for i in range(len(Config.CLASSES))
        }
        
        return jsonify({
            'status': 'success',
            'data': {
                'prediction': predicted_class,
                'confidence': round(confidence, 2),
                'probabilities': probs,
                'cloud_info': explanation_data,
                'validation': {
                    'is_sky_image': True,
                    'sky_confidence': round(sky_confidence * 100, 2),
                    'sky_detector_active': sky_detector is not None,
                    'preprocessing_applied': Config.APPLY_PREPROCESSING,
                    'preprocessing_level': Config.PREPROCESSING_LEVEL
                },
                'timestamp': datetime.now().isoformat()
            }
        })
    
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models': {
            'sky_detector': {'active': sky_detector is not None},
            'cloud_classifier': {'active': cloud_model is not None}
        },
        'preprocessing': {
            'enabled': Config.APPLY_PREPROCESSING,
            'level': Config.PREPROCESSING_LEVEL
        },
        'timestamp': datetime.now().isoformat()
    })


@app.route('/web')
def web_ui():
    return render_template("index.html")


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("Starting Flask API Server...")
    print("API: http://localhost:5000")
    print("Web UI: http://localhost:5000/web")
    print("=" * 80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)