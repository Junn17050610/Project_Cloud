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
app.py
Flask Backend API - Cloud Classification
Alur: Upload → Sky Detection → Fisheye Conversion → Contour Enhancement → Classification
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
import cv2

app = Flask(__name__)
CORS(app)

# ============================================================================
# KONFIGURASI
# ============================================================================
class Config:
    # Path models
    SKY_DETECTOR_PATH = 'new_sky_detection_1/model_weather_cnn_20260206_174556.h5'
    CLOUD_MODEL_PATH = '1_Model_Fisheye/model_cloud_20260207_132904.h5'  # Model yang dilatih dengan fisheye+contour
    CLOUD_METADATA_PATH = '1_Result_Fisheye/model_metadata_20260207_132904.json'
    
    IMG_SIZE = (224, 224)
    SKY_CONFIDENCE_THRESHOLD = 0.7
    
    # Fisheye conversion settings (CONVERT normal image TO fisheye)
    ENABLE_FISHEYE_CONVERSION = True
    FISHEYE_STRENGTH = 1.5  # Strength untuk convert ke fisheye
    
    # Contour enhancement settings (sesuai training)
    ENABLE_CONTOUR_ENHANCEMENT = True
    CONTOUR_STRENGTH = 0.3
    ENHANCEMENT_STRENGTH = 'balanced'
    
    # Cloud categories
    CLASS_MAPPING = {
        "1_cumulus": {
            'category': 'Convective Clouds',
            'types': ['Cumulus'],
            'altitude': '500 - 2,000 meter',
            'characteristics': [
                '☁️ Awan putih bergumpal dengan dasar datar',
                '🌤️ Biasanya muncul saat cuaca cerah',
                '🌡️ Terbentuk dari udara hangat yang naik'
            ],
            'weather_forecast': '🌤️ Umumnya menandakan cuaca cerah',
            'precipitation': 'Biasanya tidak hujan',
            'icon': '🌤️'
        },
        "2_altocumulus_cirrocumulus": {
            'category': 'Mid & High Level Clouds',
            'types': ['Altocumulus', 'Cirrocumulus'],
            'altitude': '2,000 - 7,000 meter',
            'characteristics': [
                '☁️ Awan bergelombang atau berbintik kecil',
                '⚪ Berwarna putih hingga abu-abu'
            ],
            'weather_forecast': '⛅ Bisa menjadi tanda perubahan cuaca dalam 12-24 jam',
            'precipitation': 'Biasanya tidak menghasilkan hujan',
            'icon': '⛅'
        },
        "3_cirrus_cirrostratus": {
            'category': 'High-Level Clouds',
            'types': ['Cirrus', 'Cirrostratus'],
            'altitude': '5,000 - 13,000 meter',
            'characteristics': [
                '☁️ Awan tipis dan halus',
                '❄️ Terbuat dari kristal es'
            ],
            'weather_forecast': '🌤️ Umumnya cuaca baik, bisa tanda perubahan cuaca',
            'precipitation': 'Tidak menghasilkan hujan',
            'icon': '🌤️'
        },
        "4_clearsky": {
            'category': 'Clear Conditions',
            'types': ['Clear Sky'],
            'altitude': 'N/A',
            'characteristics': [
                '☀️ Langit cerah tanpa awan signifikan',
                '🌞 Visibilitas sangat baik'
            ],
            'weather_forecast': '☀️ Cuaca cerah dan stabil',
            'precipitation': 'Tidak ada',
            'icon': '☀️'
        },
        "5_stratocumulus_stratus_altostratus": {
            'category': 'Low & Mid Level Clouds',
            'types': ['Stratocumulus', 'Stratus', 'Altostratus'],
            'altitude': '0 - 7,000 meter',
            'characteristics': [
                '☁️ Awan berlapis dan menutupi langit',
                '🌧️ Dapat menghasilkan hujan ringan'
            ],
            'weather_forecast': '🌥️ Cuaca mendung, kemungkinan hujan ringan',
            'precipitation': 'Gerimis hingga hujan ringan',
            'icon': '☁️'
        },
        "6_cumulonimbus_nimbostratus": {
            'category': 'Rain & Storm Clouds',
            'types': ['Cumulonimbus', 'Nimbostratus'],
            'altitude': '0 - 13,000 meter',
            'characteristics': [
                '⛈️ Awan hujan tebal dan gelap',
                '⚡ Berpotensi menghasilkan badai'
            ],
            'weather_forecast': '⛈️ Potensi hujan deras dan badai petir',
            'precipitation': 'Hujan sedang hingga sangat lebat',
            'icon': '⛈️',
            'warning': True
        }
    }

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

class CloudImagePreprocessor:
    """
    Preprocessing pipeline: Normal Image → Fisheye → Contour Enhancement
    """
    
    @staticmethod
    def apply_fisheye_effect(img_array, strength=1.5):
        """
        Convert normal image TO fisheye (sesuai dengan training data)
        """
        height, width = img_array.shape[:2]
        
        cx, cy = width // 2, height // 2
        max_radius = min(cx, cy)
        
        # Mesh grid
        y, x = np.mgrid[0:height, 0:width]
        dx = x - cx
        dy = y - cy
        distance = np.sqrt(dx**2 + dy**2)
        normalized_distance = distance / max_radius
        
        # Apply fisheye transformation (FORWARD, not inverse!)
        fisheye_distance = normalized_distance ** strength
        
        # Calculate new coordinates
        angle = np.arctan2(dy, dx)
        new_distance = fisheye_distance * max_radius
        new_x = (cx + new_distance * np.cos(angle)).astype(np.float32)
        new_y = (cy + new_distance * np.sin(angle)).astype(np.float32)
        
        # Apply transformation
        fisheye_img = cv2.remap(img_array, new_x, new_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        # Create circular mask
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), max_radius, 255, -1)
        
        if len(fisheye_img.shape) == 3:
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            fisheye_img = cv2.bitwise_and(fisheye_img, mask_3ch)
        
        return fisheye_img
    
    @staticmethod
    def enhance_with_contours(img_array, strength='balanced', contour_strength=0.3):
        """
        Enhance cloud patterns dengan contour (sesuai training)
        """
        # Convert to LAB
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # CLAHE
        strength_params = {'subtle': 1.5, 'balanced': 2.0, 'strong': 2.5}
        clip_limit = strength_params.get(strength, 2.0)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        
        # Edge detection
        l_bilateral = cv2.bilateralFilter(l_enhanced, 9, 75, 75)
        edges_canny = cv2.Canny(l_bilateral, 40, 120)
        
        # Sobel
        sobelx = cv2.Sobel(l_bilateral, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(l_bilateral, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        sobel_mag = np.uint8(255 * sobel_mag / (np.max(sobel_mag) + 1e-5))
        
        # Combine edges
        edges_combined = cv2.addWeighted(edges_canny, 0.6, sobel_mag, 0.4, 0)
        kernel = np.ones((2, 2), np.uint8)
        edges_dilated = cv2.dilate(edges_combined, kernel, iterations=1)
        
        # Overlay edges to L channel
        l_with_edges = cv2.addWeighted(l_enhanced, 1.0, edges_dilated, contour_strength, 0)
        l_with_edges = np.clip(l_with_edges, 0, 255).astype(np.uint8)
        
        # Sharpening
        if strength in ['balanced', 'strong']:
            gaussian = cv2.GaussianBlur(l_with_edges, (0, 0), 1.5)
            sharp_amount = 0.4 if strength == 'balanced' else 0.6
            l_final = cv2.addWeighted(l_with_edges, 1.0 + sharp_amount, gaussian, -sharp_amount, 0)
            l_final = np.clip(l_final, 0, 255).astype(np.uint8)
        else:
            l_final = l_with_edges
        
        # Merge back
        lab_enhanced = cv2.merge([l_final, a_channel, b_channel])
        rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        return rgb_enhanced

# ============================================================================
# LOAD MODELS
# ============================================================================

print("=" * 80)
print("CLOUD CLASSIFICATION API - FISHEYE + CONTOUR ENHANCED")
print("=" * 80)

# Load Sky Detector
sky_detector = None
sky_classes = None

if os.path.exists(Config.SKY_DETECTOR_PATH):
    try:
        print(f"\n[Model 1] Loading Sky Detector...")
        sky_detector = keras.models.load_model(Config.SKY_DETECTOR_PATH)
        print("✓ Sky Detector loaded")
        sky_classes = {0: 'bukan_langit', 1: 'langit'}
    except Exception as e:
        print(f"⚠ Sky Detector error: {str(e)}")

# Load Cloud Classifier
cloud_model = None

try:
    print(f"\n[Model 2] Loading Cloud Classifier...")
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
    print(f"✗ Cloud Model error: {str(e)}")

print("\n[Preprocessing Pipeline]")
print(f"  1. Fisheye Conversion: {'ENABLED' if Config.ENABLE_FISHEYE_CONVERSION else 'DISABLED'}")
print(f"  2. Contour Enhancement: {'ENABLED' if Config.ENABLE_CONTOUR_ENHANCEMENT else 'DISABLED'}")
print("=" * 80)

# ============================================================================
# PREPROCESSING PIPELINE
# ============================================================================

def preprocess_image(image_bytes):
    """
    Complete preprocessing pipeline
    """
    # Load image
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize first
    img = img.resize(Config.IMG_SIZE)
    img_array = np.array(img).astype(np.uint8)
    
    preprocessing_steps = []
    
    # Step 1: Convert to Fisheye
    if Config.ENABLE_FISHEYE_CONVERSION:
        img_array = CloudImagePreprocessor.apply_fisheye_effect(
            img_array,
            strength=Config.FISHEYE_STRENGTH
        )
        preprocessing_steps.append('Fisheye Conversion')
    
    # Step 2: Contour Enhancement
    if Config.ENABLE_CONTOUR_ENHANCEMENT:
        img_array = CloudImagePreprocessor.enhance_with_contours(
            img_array,
            strength=Config.ENHANCEMENT_STRENGTH,
            contour_strength=Config.CONTOUR_STRENGTH
        )
        preprocessing_steps.append('Contour Enhancement')
    
    # Normalize for model
    img_normalized = img_array.astype('float32') / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    metadata = {
        'preprocessing_steps': preprocessing_steps,
        'fisheye_applied': Config.ENABLE_FISHEYE_CONVERSION,
        'contour_applied': Config.ENABLE_CONTOUR_ENHANCEMENT
    }
    
    return img_batch, metadata

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'service': 'Cloud Classification API',
        'preprocessing': {
            'fisheye_conversion': Config.ENABLE_FISHEYE_CONVERSION,
            'contour_enhancement': Config.ENABLE_CONTOUR_ENHANCEMENT
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Cloud Classification Pipeline:
    1. Sky Detection
    2. Fisheye Conversion
    3. Contour Enhancement
    4. Classification
    """
    
    if cloud_model is None:
        return jsonify({'status': 'error', 'message': 'Model belum dimuat'}), 500
    
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'Tidak ada gambar'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'File kosong'}), 400
    
    try:
        print("\n" + "="*60)
        print("NEW REQUEST")
        print("="*60)
        
        image_bytes = file.read()
        
        # ====== STAGE 1: SKY DETECTION ======
        print("\n[Stage 1] Sky Detection...")
        
        # Quick preprocess for sky detection
        img_temp = Image.open(io.BytesIO(image_bytes))
        if img_temp.mode != 'RGB':
            img_temp = img_temp.convert('RGB')
        img_temp = img_temp.resize(Config.IMG_SIZE)
        img_temp_array = np.array(img_temp).astype('float32') / 255.0
        img_temp_batch = np.expand_dims(img_temp_array, axis=0)
        
        is_sky = True
        sky_confidence = 1.0
        
        if sky_detector is not None:
            sky_pred = sky_detector.predict(img_temp_batch, verbose=0)
            sky_idx = np.argmax(sky_pred[0])
            sky_confidence = float(sky_pred[0][sky_idx])
            sky_class = sky_classes.get(sky_idx, 'unknown')
            
            print(f"  Result: {sky_class} ({sky_confidence*100:.1f}%)")
            
            if sky_idx == 0 or sky_confidence < Config.SKY_CONFIDENCE_THRESHOLD:
                is_sky = False
        
        if not is_sky:
            return jsonify({
                'status': 'error',
                'error_type': 'not_sky_image',
                'message': 'Gambar bukan langit/awan',
                'detail': {
                    'sky_confidence': round(sky_confidence * 100, 2),
                    'suggestion': 'Upload gambar langit atau awan'
                }
            }), 400
        
        # ====== STAGE 2: PREPROCESSING ======
        print("\n[Stage 2] Preprocessing...")
        processed_image, preproc_meta = preprocess_image(image_bytes)
        
        print(f"  Steps: {', '.join(preproc_meta['preprocessing_steps'])}")
        
        # ====== STAGE 3: CLASSIFICATION ======
        print("\n[Stage 3] Cloud Classification...")
        
        predictions = cloud_model.predict(processed_image, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx]) * 100
        predicted_class = Config.CLASSES[predicted_idx]
        
        print(f"  Predicted: {predicted_class} ({confidence:.1f}%)")
        
        # Get cloud info
        cloud_info = Config.CLASS_MAPPING.get(predicted_class, {})
        
        # Build explanation
        types = cloud_info.get('types', ['Unknown'])
        if len(types) == 1:
            explanation = f"Terdeteksi <strong>{types[0]}</strong>"
        else:
            explanation = f"Terdeteksi {' atau '.join(types)}"
        explanation += f" dengan keyakinan <strong>{confidence:.1f}%</strong>"
        
        # Probabilities
        probs = {
            Config.CLASSES[i]: float(predictions[0][i]) * 100
            for i in range(len(Config.CLASSES))
        }
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        print("="*60)
        print("✓ SUCCESS")
        print("="*60 + "\n")
        
        return jsonify({
            'status': 'success',
            'data': {
                'prediction': predicted_class,
                'confidence': round(confidence, 2),
                'cloud_info': {
                    'explanation': explanation,
                    'cloud_types': types,
                    'altitude': cloud_info.get('altitude', 'Unknown'),
                    'weather': cloud_info.get('weather_forecast', 'Unknown'),
                    'icon': cloud_info.get('icon', '☁️'),
                    'characteristics': cloud_info.get('characteristics', []),
                    'precipitation': cloud_info.get('precipitation', 'Unknown')
                },
                'probabilities': {k: round(v, 2) for k, v in sorted_probs},
                'preprocessing': preproc_meta,
                'validation': {
                    'is_sky_image': True,
                    'sky_confidence': round(sky_confidence * 100, 2)
                },
                'timestamp': datetime.now().isoformat()
            }
        })
    
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'models': {
            'sky_detector': 'loaded' if sky_detector else 'not_loaded',
            'cloud_classifier': 'loaded' if cloud_model else 'not_loaded'
        }
    })

@app.route('/web')
def web_ui():
    return render_template("index.html")

if __name__ == '__main__':
    print("\n" + "="*80)
    print("Starting Server...")
    print("="*80 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)