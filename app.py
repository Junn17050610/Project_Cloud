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
Flask Backend API - Cloud Classification with Detailed Weather Prediction
Kategori: HIGH_CLOUD, MID_CLOUD, LOW_CLOUD, CONVECTIVE, CONTRAIL, CLEAR_SKY
"""

"""
app.py
Flask Backend API - Cloud Classification with Fisheye Preprocessing
Otomatis mendeteksi dan memproses gambar fisheye dari kamera handphone
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
    CLOUD_MODEL_PATH = 'models_cloud_Similarity_Fisheye_4/model_cloud_classification_20260206_092744.h5'
    CLOUD_METADATA_PATH = 'results_cloud_Similarity_Fisheye_4/model_metadata_20260206_092744.json'
    
    IMG_SIZE = (224, 224)
    SKY_CONFIDENCE_THRESHOLD = 0.7
    
    # Fisheye preprocessing config
    FISHEYE_DETECTION_THRESHOLD = 0.3  # Threshold untuk deteksi otomatis fisheye
    ENABLE_AUTO_FISHEYE_CORRECTION = True  # Auto-correct fisheye jika terdeteksi
    
    # KATEGORI dengan detail lengkap
    CLASS_MAPPING = {
        'HIGH_CLOUD': {
            'category': 'High-Level Clouds',
            'types': ['Cirrus', 'Cirrostratus', 'Cirrocumulus'],
            'altitude': '5,000 - 13,000 meter (16,500 - 43,000 kaki)',
            'characteristics': [
                '☁️ Awan tipis dan halus seperti bulu burung atau serat',
                '🌡️ Terbentuk dari kristal es karena suhu sangat dingin',
                '☀️ Sering terlihat putih atau transparan',
                '🌅 Membuat halo di sekitar matahari atau bulan'
            ],
            'cloud_details': {
                'Cirrus': {
                    'description': 'Awan tinggi berbentuk serat halus dan melengkung',
                    'appearance': 'Seperti bulu burung atau ekor kuda',
                    'composition': 'Kristal es'
                },
                'Cirrostratus': {
                    'description': 'Lapisan tipis awan tinggi yang menutupi langit',
                    'appearance': 'Seperti tirai putih tipis',
                    'composition': 'Kristal es'
                },
                'Cirrocumulus': {
                    'description': 'Awan tinggi berbentuk butiran kecil atau riak',
                    'appearance': 'Seperti sisik ikan (mackerel sky)',
                    'composition': 'Kristal es'
                }
            },
            'weather_forecast': '🌤️ Cuaca umumnya baik, namun bisa menjadi tanda perubahan cuaca dalam 24-48 jam ke depan',
            'precipitation': 'Tidak menghasilkan hujan',
            'icon': '🌤️'
        },
        
        'MID_CLOUD': {
            'category': 'Mid-Level Clouds',
            'types': ['Altocumulus', 'Altostratus'],
            'altitude': '2,000 - 7,000 meter (6,500 - 23,000 kaki)',
            'characteristics': [
                '☁️ Awan bergelombang atau berlapis di ketinggian menengah',
                '🌡️ Tersusun dari tetesan air dan kristal es',
                '⚪ Berwarna abu-abu hingga putih kebiruan',
                '🌥️ Bisa menutupi sebagian atau seluruh langit'
            ],
            'cloud_details': {
                'Altocumulus': {
                    'description': 'Awan menengah berbentuk gumpalan atau lembaran',
                    'appearance': 'Seperti kapas bergelombang tersusun rapi',
                    'composition': 'Tetesan air dan kristal es'
                },
                'Altostratus': {
                    'description': 'Lapisan awan menengah yang luas dan seragam',
                    'appearance': 'Seperti selimut abu-abu menutupi langit',
                    'composition': 'Tetesan air dan kristal es'
                }
            },
            'weather_forecast': '⛅ Cuaca berawan hingga mendung. Kemungkinan hujan ringan hingga sedang dalam 12-24 jam',
            'precipitation': 'Altocumulus: Biasanya tidak hujan. Altostratus: Hujan ringan atau gerimis',
            'icon': '⛅'
        },
        
        'LOW_CLOUD': {
            'category': 'Low-Level Clouds',
            'types': ['Stratus', 'Stratocumulus', 'Nimbostratus'],
            'altitude': '0 - 2,000 meter (0 - 6,500 kaki)',
            'characteristics': [
                '☁️ Awan rendah yang gelap dan tebal',
                '🌫️ Bisa menutupi seluruh langit dengan lapisan seragam',
                '🌧️ Sering dikaitkan dengan cuaca mendung dan hujan',
                '❄️ Tersusun dari tetesan air (atau kristal es jika sangat dingin)'
            ],
            'cloud_details': {
                'Stratus': {
                    'description': 'Lapisan awan rendah yang seragam dan mendatar',
                    'appearance': 'Seperti kabut yang terangkat dari tanah',
                    'composition': 'Tetesan air kecil'
                },
                'Stratocumulus': {
                    'description': 'Awan rendah berbentuk gumpalan atau bergumpal',
                    'appearance': 'Seperti gulungan kapas gelap tersusun',
                    'composition': 'Tetesan air'
                },
                'Nimbostratus': {
                    'description': 'Awan hujan tebal yang gelap dan luas',
                    'appearance': 'Lapisan gelap tebal tanpa bentuk jelas',
                    'composition': 'Tetesan air dan kristal es'
                }
            },
            'weather_forecast': '🌧️ Cuaca mendung dan basah. Hujan gerimis hingga sedang yang berkelanjutan',
            'precipitation': 'Stratus: Gerimis ringan. Stratocumulus: Hujan ringan. Nimbostratus: Hujan sedang berkelanjutan',
            'icon': '☁️'
        },
        
        'CONVECTIVE': {
            'category': 'Convective Clouds',
            'types': ['Cumulus', 'Cumulonimbus'],
            'altitude': '0 - 13,000 meter (vertikal, dari rendah hingga sangat tinggi)',
            'characteristics': [
                '⛈️ Awan konvektif dengan perkembangan vertikal kuat',
                '🌪️ Terbentuk dari udara hangat yang naik cepat',
                '⚡ Dapat menghasilkan cuaca ekstrem (petir, hujan deras, angin kencang)',
                '☁️ Bentuk menjulang tinggi seperti menara atau landasan'
            ],
            'cloud_details': {
                'Cumulus': {
                    'description': 'Awan putih bergumpal dengan dasar datar',
                    'appearance': 'Seperti kapas atau kembang kol mengembang',
                    'composition': 'Tetesan air',
                    'types': 'Cumulus humilis (kecil), Cumulus mediocris (sedang), Cumulus congestus (besar)'
                },
                'Cumulonimbus': {
                    'description': 'Awan badai yang sangat besar dan tinggi',
                    'appearance': 'Seperti menara raksasa dengan puncak berbentuk landasan',
                    'composition': 'Tetesan air, kristal es, hujan, salju, es',
                    'danger': '⚠️ BAHAYA: Dapat menghasilkan petir, hujan lebat, hujan es, tornado'
                }
            },
            'weather_forecast': '⛈️ PERINGATAN: Cuaca berpotensi ekstrem! Cumulus: Cuaca cerah hingga berawan. Cumulonimbus: Badai petir, hujan sangat deras, angin kencang, kemungkinan hujan es atau tornado',
            'precipitation': 'Cumulus: Biasanya tidak hujan. Cumulonimbus: Hujan sangat lebat (downpour) dengan intensitas tinggi',
            'icon': '⛈️',
            'warning': True
        },
        
        # '7_contrail': {
        #     'category': 'Artificial Clouds',
        #     'types': ['Contrail (Condensation Trail)'],
        #     'altitude': '8,000 - 12,000 meter (26,000 - 40,000 kaki)',
        #     'characteristics': [
        #         '✈️ Jejak kondensasi dari pesawat terbang',
        #         '❄️ Terbentuk dari uap air mesin pesawat yang membeku',
        #         '➖ Berbentuk garis lurus atau sedikit melengkung',
        #         '🌡️ Indikator kelembaban tinggi di atmosfer atas'
        #     ],
        #     'cloud_details': {
        #         'Contrail': {
        #             'description': 'Garis putih panjang yang terbentuk di belakang pesawat',
        #             'appearance': 'Seperti garis lurus putih di langit',
        #             'composition': 'Kristal es dari uap air mesin pesawat',
        #             'persistence': 'Bisa hilang cepat atau bertahan lama tergantung kelembaban'
        #         }
        #     },
        #     'weather_forecast': '✈️ Contrail sendiri tidak memprediksi cuaca, tetapi persistensinya menunjukkan kelembaban tinggi di atmosfer atas yang bisa mengindikasikan perubahan cuaca',
        #     'precipitation': 'Tidak menghasilkan hujan',
        #     'icon': '✈️'
        # },
        
        '4_clearsky': {
            'category': 'Clear Conditions',
            'types': ['Clear Sky (Langit Cerah)'],
            'altitude': 'N/A',
            'characteristics': [
                '☀️ Langit cerah tanpa awan atau dengan awan minimal',
                '🌞 Visibilitas sangat baik',
                '🌡️ Suhu dipengaruhi langsung oleh radiasi matahari',
                '🌤️ Kondisi cuaca stabil'
            ],
            'cloud_details': {
                'Clear Sky': {
                    'description': 'Kondisi langit tanpa tutupan awan signifikan',
                    'appearance': 'Langit biru jernih atau dengan sedikit awan',
                    'composition': 'N/A'
                }
            },
            'weather_forecast': '☀️ Cuaca cerah dan stabil. Tidak ada tanda-tanda hujan. Kondisi bagus untuk aktivitas outdoor',
            'precipitation': 'Tidak ada',
            'icon': '☀️'
        }
    }
    
    # Mapping dari class name lama ke kategori baru
    OLD_TO_NEW_MAPPING = {
        '2_altocumulus_cirrocumulus': 'MID_CLOUD',
        '3_cirrus_cirrostratus': 'HIGH_CLOUD',
        '4_clearsky': '4_clearsky',
        '5_stratocumulus_stratus_altostratus': 'LOW_CLOUD',
        '6_cumulonimbus_nimbostratus': 'CONVECTIVE',
        '7_contrail': '7_contrail'
    }

# ============================================================================
# FISHEYE PREPROCESSING FUNCTIONS
# ============================================================================

class FisheyePreprocessor:
    """Class untuk mendeteksi dan mengoreksi distorsi fisheye"""
    
    @staticmethod
    def detect_fisheye(image):
        """
        Deteksi apakah gambar mengandung distorsi fisheye
        Returns: (is_fisheye: bool, confidence: float)
        """
        # Convert ke grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        
        # Cek aspect ratio dan circular edges
        # Gambar fisheye biasanya memiliki:
        # 1. Edge detection tinggi di area circular
        # 2. Brightness yang lebih gelap di edges
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Hitung edge density di corners vs center
        corner_size = min(h, w) // 6
        
        # Corner regions (4 corners)
        corners_density = 0
        corners_density += np.mean(edges[0:corner_size, 0:corner_size])  # Top-left
        corners_density += np.mean(edges[0:corner_size, -corner_size:])  # Top-right
        corners_density += np.mean(edges[-corner_size:, 0:corner_size])  # Bottom-left
        corners_density += np.mean(edges[-corner_size:, -corner_size:])  # Bottom-right
        corners_density /= 4
        
        # Center region
        cy, cx = h // 2, w // 2
        center_radius = min(h, w) // 4
        center_region = edges[cy-center_radius:cy+center_radius, cx-center_radius:cx+center_radius]
        center_density = np.mean(center_region) if center_region.size > 0 else 0
        
        # Fisheye biasanya punya corner yang lebih gelap/lebih banyak edge
        # karena distorsi dan vignetting
        if corners_density > 0:
            ratio = center_density / corners_density
        else:
            ratio = 1.0
        
        # Cek circular pattern menggunakan brightness
        # Fisheye biasanya lebih gelap di edges (vignetting)
        brightness_corners = 0
        brightness_corners += np.mean(gray[0:corner_size, 0:corner_size])
        brightness_corners += np.mean(gray[0:corner_size, -corner_size:])
        brightness_corners += np.mean(gray[-corner_size:, 0:corner_size])
        brightness_corners += np.mean(gray[-corner_size:, -corner_size:])
        brightness_corners /= 4
        
        center_region_gray = gray[cy-center_radius:cy+center_radius, cx-center_radius:cx+center_radius]
        brightness_center = np.mean(center_region_gray) if center_region_gray.size > 0 else 0
        
        # Calculate vignetting score
        if brightness_center > 0:
            vignetting_score = 1 - (brightness_corners / brightness_center)
        else:
            vignetting_score = 0
        
        # Combine scores
        # Fisheye confidence berdasarkan vignetting dan edge distribution
        fisheye_confidence = (vignetting_score * 0.7) + ((1 - min(ratio, 1.0)) * 0.3)
        fisheye_confidence = max(0, min(1, fisheye_confidence))
        
        is_fisheye = fisheye_confidence > Config.FISHEYE_DETECTION_THRESHOLD
        
        return is_fisheye, fisheye_confidence
    
    @staticmethod
    def apply_fisheye_inverse(image, strength=1.5):
        """
        Mengoreksi distorsi fisheye dengan inverse transformation
        """
        height, width = image.shape[:2]
        
        # Koordinat pusat
        cx, cy = width // 2, height // 2
        
        # Radius maksimum
        max_radius = min(cx, cy)
        
        # Buat mesh grid
        y, x = np.mgrid[0:height, 0:width]
        
        # Hitung jarak dari pusat
        dx = x - cx
        dy = y - cy
        distance = np.sqrt(dx**2 + dy**2)
        
        # Normalisasi jarak
        normalized_distance = distance / max_radius
        
        # INVERSE fisheye transformation
        # Untuk mengoreksi fisheye, kita lakukan kebalikan dari distorsi
        corrected_distance = normalized_distance ** (1.0 / strength)
        
        # Hitung koordinat baru
        angle = np.arctan2(dy, dx)
        new_distance = corrected_distance * max_radius
        
        new_x = (cx + new_distance * np.cos(angle)).astype(np.float32)
        new_y = (cy + new_distance * np.sin(angle)).astype(np.float32)
        
        # Apply transformation
        corrected = cv2.remap(image, new_x, new_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return corrected
    
    @staticmethod
    def process_image(pil_image, auto_correct=True):
        """
        Main preprocessing function
        Args:
            pil_image: PIL Image object
            auto_correct: Jika True, otomatis koreksi jika fisheye terdeteksi
        Returns:
            (processed_image: PIL.Image, metadata: dict)
        """
        # Convert PIL to numpy array
        img_array = np.array(pil_image)
        
        # Deteksi fisheye
        is_fisheye, fisheye_confidence = FisheyePreprocessor.detect_fisheye(img_array)
        
        metadata = {
            'fisheye_detected': is_fisheye,
            'fisheye_confidence': round(fisheye_confidence * 100, 2),
            'correction_applied': False,
            'original_size': pil_image.size
        }
        
        # Jika fisheye terdeteksi dan auto_correct enabled
        if is_fisheye and auto_correct:
            print(f"  📸 Fisheye detected (confidence: {fisheye_confidence*100:.1f}%), applying correction...")
            
            # Tentukan strength koreksi berdasarkan confidence
            # Semakin tinggi confidence fisheye, semakin kuat koreksi
            correction_strength = 1.2 + (fisheye_confidence * 0.8)  # Range: 1.2 - 2.0
            
            corrected = FisheyePreprocessor.apply_fisheye_inverse(img_array, strength=correction_strength)
            
            # Convert back to PIL
            processed_pil = Image.fromarray(corrected.astype('uint8'))
            
            metadata['correction_applied'] = True
            metadata['correction_strength'] = round(correction_strength, 2)
            
            return processed_pil, metadata
        else:
            if is_fisheye:
                print(f"  📸 Fisheye detected but correction disabled")
            
            return pil_image, metadata

# ============================================================================
# LOAD MODELS
# ============================================================================
print("=" * 80)
print("CLOUD WEATHER PREDICTION API WITH FISHEYE PREPROCESSING")
print("=" * 80)

# Load Sky Detector
sky_detector = None
sky_classes = None

if os.path.exists(Config.SKY_DETECTOR_PATH):
    try:
        print(f"\n[Stage 1] Loading Sky Detector: {Config.SKY_DETECTOR_PATH}")
        sky_detector = keras.models.load_model(Config.SKY_DETECTOR_PATH)
        print("✓ Sky Detector loaded successfully")
        sky_classes = {0: 'bukan_langit', 1: 'langit'}
    except Exception as e:
        print(f"⚠ Warning: Sky Detector gagal dimuat: {str(e)}")

# Load Cloud Classifier
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
        Config.OLD_CLASSES = metadata.get('classes', list(Config.OLD_TO_NEW_MAPPING.keys()))
    else:
        Config.OLD_CLASSES = list(Config.OLD_TO_NEW_MAPPING.keys())
        
except Exception as e:
    print(f"✗ Error loading Cloud Model: {str(e)}")

print("\n[Fisheye Preprocessing]")
print(f"  Auto-detection: {'ENABLED ✓' if Config.ENABLE_AUTO_FISHEYE_CORRECTION else 'DISABLED ✗'}")
print(f"  Detection threshold: {Config.FISHEYE_DETECTION_THRESHOLD * 100}%")

print("=" * 80)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def preprocess_image(image_bytes, apply_fisheye_correction=True):
    """
    Preprocess image dengan fisheye correction
    Args:
        image_bytes: Raw image bytes
        apply_fisheye_correction: Enable/disable fisheye correction
    Returns:
        (preprocessed_array, preprocessing_metadata)
    """
    # Load image
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Apply fisheye preprocessing jika enabled
    preprocessing_metadata = {}
    if apply_fisheye_correction and Config.ENABLE_AUTO_FISHEYE_CORRECTION:
        img, fisheye_meta = FisheyePreprocessor.process_image(img, auto_correct=True)
        preprocessing_metadata['fisheye'] = fisheye_meta
    
    # Resize untuk model
    img = img.resize(Config.IMG_SIZE)
    img_array = np.array(img).astype('float32') / 255.0
    
    return np.expand_dims(img_array, axis=0), preprocessing_metadata

def get_specific_cloud_type(old_class, confidence_scores):
    """Menentukan jenis awan spesifik"""
    specific_mapping = {
        '2_altocumulus_cirrocumulus': ['Altocumulus', 'Cirrocumulus'],
        '3_cirrus_cirrostratus': ['Cirrus', 'Cirrostratus'],
        '5_stratocumulus_stratus_altostratus': ['Stratocumulus', 'Stratus', 'Altostratus'],
        '6_cumulonimbus_nimbostratus': ['Cumulonimbus', 'Nimbostratus']
    }
    
    if old_class in specific_mapping:
        return specific_mapping[old_class][0]
    
    return None

def generate_detailed_explanation(old_class, confidence, all_probabilities):
    """Generate detailed explanation"""
    new_category = Config.OLD_TO_NEW_MAPPING.get(old_class, old_class)
    category_info = Config.CLASS_MAPPING.get(new_category, {})
    specific_cloud = get_specific_cloud_type(old_class, all_probabilities)
    
    explanation = {
        'category': new_category,
        'category_name': category_info.get('category', 'Unknown'),
        'specific_cloud_type': specific_cloud,
        'confidence': confidence,
        'cloud_types_in_category': category_info.get('types', []),
        'altitude': category_info.get('altitude', 'Unknown'),
        'characteristics': category_info.get('characteristics', []),
        'cloud_details': category_info.get('cloud_details', {}),
        'weather_forecast': category_info.get('weather_forecast', 'Unknown'),
        'precipitation': category_info.get('precipitation', 'Unknown'),
        'icon': category_info.get('icon', '☁️'),
        'is_warning': category_info.get('warning', False)
    }
    
    types = category_info.get('types', ['Unknown'])
    
    if specific_cloud:
        explanation['main_text'] = f"Awan yang terdeteksi adalah <strong>{specific_cloud}</strong>"
        if len(types) > 1:
            other_types = [t for t in types if t != specific_cloud]
            if other_types:
                explanation['main_text'] += f", yang termasuk dalam kategori <strong>{new_category}</strong> bersama dengan {', '.join(other_types)}"
        else:
            explanation['main_text'] += f" (kategori <strong>{new_category}</strong>)"
    else:
        if len(types) == 1:
            explanation['main_text'] = f"Terdeteksi <strong>{types[0]}</strong>"
        else:
            types_str = ', '.join(types[:-1]) + f" dan {types[-1]}"
            explanation['main_text'] = f"Terdeteksi awan kategori <strong>{new_category}</strong>, yang meliputi {types_str}"
    
    explanation['main_text'] += f" dengan tingkat keyakinan <strong>{confidence:.1f}%</strong>."
    
    return explanation

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'service': 'Cloud Weather Prediction API',
        'version': '2.1',
        'features': {
            'fisheye_preprocessing': Config.ENABLE_AUTO_FISHEYE_CORRECTION,
            'sky_detection': sky_detector is not None,
            'cloud_classification': cloud_model is not None
        },
        'categories': list(Config.CLASS_MAPPING.keys())
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Cloud Classification with Fisheye Preprocessing
    
    Form parameters:
    - image: Image file (required)
    - disable_fisheye_correction: Set to 'true' to disable fisheye correction (optional)
    """
    
    if cloud_model is None:
        return jsonify({
            'status': 'error',
            'message': 'Cloud model belum dimuat'
        }), 500
    
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
        # Check jika user ingin disable fisheye correction
        disable_correction = request.form.get('disable_fisheye_correction', 'false').lower() == 'true'
        apply_correction = not disable_correction
        
        # Read and preprocess image
        image_bytes = file.read()
        processed_image, preprocessing_meta = preprocess_image(image_bytes, apply_fisheye_correction=apply_correction)
        
        print("\n" + "="*50)
        print("PROCESSING NEW REQUEST")
        print("="*50)
        
        if 'fisheye' in preprocessing_meta:
            fisheye_info = preprocessing_meta['fisheye']
            print(f"📸 Fisheye Detection:")
            print(f"  - Detected: {'Yes' if fisheye_info['fisheye_detected'] else 'No'}")
            print(f"  - Confidence: {fisheye_info['fisheye_confidence']}%")
            print(f"  - Correction Applied: {'Yes' if fisheye_info['correction_applied'] else 'No'}")
        
        # ====== STAGE 1: SKY DETECTION ======
        is_sky = True
        sky_confidence = 1.0
        sky_class_name = "unknown"
        
        if sky_detector is not None:
            print("\n[Stage 1] Sky Detection...")
            sky_predictions = sky_detector.predict(processed_image, verbose=0)
            sky_class_idx = np.argmax(sky_predictions[0])
            sky_confidence = float(sky_predictions[0][sky_class_idx])
            sky_class_name = sky_classes.get(sky_class_idx, 'unknown')
            
            print(f"  Result: {sky_class_name} ({sky_confidence*100:.2f}%)")
            
            if sky_class_idx == 0 or sky_confidence < Config.SKY_CONFIDENCE_THRESHOLD:
                is_sky = False
        
        if not is_sky:
            return jsonify({
                'status': 'error',
                'error_type': 'not_sky_image',
                'message': 'Gambar yang diupload bukan gambar langit/awan',
                'detail': {
                    'is_sky': False,
                    'predicted_class': sky_class_name,
                    'sky_confidence': round(sky_confidence * 100, 2),
                    'preprocessing': preprocessing_meta,
                    'suggestion': 'Silakan upload gambar langit atau awan untuk prediksi cuaca'
                }
            }), 400
        
        # ====== STAGE 2: CLOUD CLASSIFICATION ======
        print("\n[Stage 2] Cloud Classification...")
        predictions = cloud_model.predict(processed_image, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx]) * 100
        old_predicted_class = Config.OLD_CLASSES[predicted_idx]
        
        print(f"  Predicted: {old_predicted_class} ({confidence:.2f}%)")
        
        # Generate detailed explanation
        explanation = generate_detailed_explanation(
            old_predicted_class, 
            confidence, 
            predictions[0]
        )
        
        # Category probabilities
        category_probabilities = {}
        for i, old_class in enumerate(Config.OLD_CLASSES):
            new_cat = Config.OLD_TO_NEW_MAPPING.get(old_class, old_class)
            prob = float(predictions[0][i]) * 100
            if new_cat in category_probabilities:
                category_probabilities[new_cat] = max(category_probabilities[new_cat], prob)
            else:
                category_probabilities[new_cat] = prob
        
        sorted_probs = sorted(category_probabilities.items(), key=lambda x: x[1], reverse=True)
        
        print("="*50)
        print("✓ Request completed successfully")
        print("="*50 + "\n")
        
        return jsonify({
            'status': 'success',
            'data': {
                'prediction': explanation['category'],
                'specific_cloud_type': explanation['specific_cloud_type'],
                'confidence': round(confidence, 2),
                'explanation': explanation,
                'probabilities': {k: round(v, 2) for k, v in sorted_probs},
                'validation': {
                    'is_sky_image': True,
                    'sky_confidence': round(sky_confidence * 100, 2)
                },
                'preprocessing': preprocessing_meta,
                'timestamp': datetime.now().isoformat()
            }
        })
    
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Error saat prediksi: {str(e)}'
        }), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get all cloud categories information"""
    return jsonify({
        'status': 'success',
        'categories': Config.CLASS_MAPPING
    })

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current API configuration"""
    return jsonify({
        'status': 'success',
        'config': {
            'fisheye_preprocessing': {
                'enabled': Config.ENABLE_AUTO_FISHEYE_CORRECTION,
                'detection_threshold': Config.FISHEYE_DETECTION_THRESHOLD * 100,
                'description': 'Automatic fisheye lens distortion detection and correction'
            },
            'image_size': Config.IMG_SIZE,
            'sky_confidence_threshold': Config.SKY_CONFIDENCE_THRESHOLD * 100
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'models': {
            'sky_detector': 'loaded' if sky_detector else 'not_loaded',
            'cloud_classifier': 'loaded' if cloud_model else 'not_loaded'
        },
        'features': {
            'fisheye_preprocessing': Config.ENABLE_AUTO_FISHEYE_CORRECTION
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/web')
def web_ui():
    """Web UI"""
    return render_template("index.html")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("Starting Flask Server...")
    print("Fisheye preprocessing: ENABLED" if Config.ENABLE_AUTO_FISHEYE_CORRECTION else "Fisheye preprocessing: DISABLED")
    print("="*80 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)