// Cloud Weather Prediction Frontend - Version 2.1
// Enhanced with fisheye preprocessing support and detailed cloud information

const API_URL = "https://projectcloud-production.up.railway.app/api/predict";

let selectedFile = null;

// Element references
const fileInput = document.getElementById("fileInput");
const cameraInput = document.getElementById("cameraInput");
const previewSection = document.getElementById("previewSection");
const imagePreview = document.getElementById("imagePreview");
const predictBtn = document.getElementById("predictBtn");
const resultSection = document.getElementById("resultSection");
const errorMessage = document.getElementById("errorMessage");

// Event listeners
fileInput.addEventListener("change", handleImageSelect);
cameraInput.addEventListener("change", handleImageSelect);
predictBtn.addEventListener("click", handlePredict);

/**
 * Handle image selection
 */
function handleImageSelect(e) {
  const file = e.target.files[0];

  if (!file) return;

  if (!file.type.startsWith("image/")) {
    showError("File yang dipilih bukan gambar!");
    return;
  }

  if (file.size > 10 * 1024 * 1024) {
    showError("Ukuran file terlalu besar! Maksimal 10MB");
    return;
  }

  selectedFile = file;

  const reader = new FileReader();
  reader.onload = function (event) {
    imagePreview.src = event.target.result;
    imagePreview.style.display = "block";
    previewSection.querySelector(".placeholder").style.display = "none";
    predictBtn.disabled = false;

    resultSection.classList.remove("show");
    hideError();
  };
  reader.readAsDataURL(file);
}

/**
 * Handle prediction
 */
async function handlePredict() {
  if (!selectedFile) {
    showError("Tidak ada gambar yang dipilih!");
    return;
  }

  predictBtn.textContent = "Menganalisis...";
  predictBtn.disabled = true;
  hideError();

  try {
    const formData = new FormData();
    formData.append("image", selectedFile);
    
    // Optional: Disable fisheye correction jika diperlukan
    // formData.append("disable_fisheye_correction", "false");

    const response = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });

    const result = await response.json();

    console.log("API Response:", result);

    if (result.status === "success") {
      showDetailedResult(result.data);
    } else if (result.error_type === "not_sky_image") {
      showNotSkyError(result);
    } else {
      showError(result.message || "Terjadi kesalahan saat prediksi");
    }
  } catch (error) {
    console.error("Error:", error);
    showError("Gagal terhubung ke server: " + error.message);
  } finally {
    predictBtn.textContent = "Prediksi Cuaca";
    predictBtn.disabled = false;
  }
}

/**
 * Show detailed cloud classification result
 */
function showDetailedResult(data) {
  const { 
    prediction, 
    specific_cloud_type, 
    confidence, 
    explanation, 
    probabilities,
    preprocessing  // NEW: Fisheye preprocessing info
  } = data;

  const resultIcon = document.getElementById("resultIcon");
  const resultTitle = document.getElementById("resultTitle");
  const resultDescription = document.getElementById("resultDescription");
  const confidenceText = document.getElementById("confidenceText");

  // Set icon
  resultIcon.textContent = explanation.icon || "☁️";

  // Set title
  if (specific_cloud_type) {
    resultTitle.innerHTML = `${specific_cloud_type.toUpperCase()}`;
  } else {
    resultTitle.innerHTML = prediction.replace(/_/g, " ");
  }

  // Build detailed description
  let descriptionHTML = '';

  // NEW: Show fisheye preprocessing info if detected
  if (preprocessing && preprocessing.fisheye) {
    descriptionHTML += buildPreprocessingInfo(preprocessing.fisheye);
  }

  descriptionHTML += `
    <div style="margin-bottom: 20px;">
      ${explanation.main_text}
    </div>
  `;

  // Warning for dangerous weather
  if (explanation.is_warning) {
    descriptionHTML += `
      <div style="background: rgba(255, 50, 50, 0.2); padding: 12px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #ff3333;">
        <strong>⚠️ PERINGATAN CUACA</strong>
      </div>
    `;
  }

  // Category information
  descriptionHTML += `
    <div style="background: rgba(255,255,255,0.15); padding: 12px; border-radius: 8px; margin-bottom: 15px;">
      <div style="font-weight: bold; margin-bottom: 8px;">📋 Kategori: ${explanation.category_name}</div>
      <div style="font-size: 13px; opacity: 0.95;">
        ${explanation.cloud_types_in_category.join(", ")}
      </div>
    </div>
  `;

  // Specific cloud details
  if (specific_cloud_type && explanation.cloud_details[specific_cloud_type]) {
    const details = explanation.cloud_details[specific_cloud_type];
    
    descriptionHTML += `
      <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
        <div style="font-weight: bold; font-size: 15px; margin-bottom: 10px; color: #fff;">
          ☁️ Detail ${specific_cloud_type}
        </div>
        <div style="font-size: 13px; line-height: 1.6; margin-bottom: 8px;">
          <strong>Deskripsi:</strong> ${details.description}
        </div>
        <div style="font-size: 13px; line-height: 1.6; margin-bottom: 8px;">
          <strong>Penampakan:</strong> ${details.appearance}
        </div>
        <div style="font-size: 13px; line-height: 1.6;">
          <strong>Komposisi:</strong> ${details.composition}
        </div>
        ${details.danger ? `
          <div style="font-size: 13px; line-height: 1.6; margin-top: 8px; color: #ffcccc;">
            ${details.danger}
          </div>
        ` : ''}
      </div>
    `;
  }

  // Characteristics
  if (explanation.characteristics && explanation.characteristics.length > 0) {
    descriptionHTML += `
      <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
        <div style="font-weight: bold; margin-bottom: 10px;">🔍 Karakteristik:</div>
        <div style="font-size: 13px; line-height: 1.8;">
          ${explanation.characteristics.map(char => `<div style="margin-bottom: 5px;">${char}</div>`).join('')}
        </div>
      </div>
    `;
  }

  // Altitude
  descriptionHTML += `
    <div style="font-size: 14px; margin-bottom: 12px; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 6px;">
      📏 <strong>Ketinggian:</strong> ${explanation.altitude}
    </div>
  `;

  // Weather forecast
  descriptionHTML += `
    <div style="background: rgba(100, 200, 255, 0.2); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
      <div style="font-weight: bold; margin-bottom: 8px;">🌤️ Prediksi Cuaca:</div>
      <div style="font-size: 14px; line-height: 1.6;">
        ${explanation.weather_forecast}
      </div>
    </div>
  `;

  // Precipitation info
  descriptionHTML += `
    <div style="font-size: 13px; margin-bottom: 12px; background: rgba(255,255,255,0.08); padding: 10px; border-radius: 6px;">
      💧 <strong>Potensi Hujan:</strong> ${explanation.precipitation}
    </div>
  `;

  resultDescription.innerHTML = descriptionHTML;

  // Confidence section
  let confidenceHTML = `
    <div style="margin-bottom: 15px; padding: 12px; background: rgba(255,255,255,0.1); border-radius: 8px;">
      <div style="font-weight: bold; margin-bottom: 8px;">📊 Tingkat Keyakinan</div>
      <div style="font-size: 24px; font-weight: bold; color: ${getConfidenceColor(confidence)};">
        ${confidence.toFixed(1)}%
      </div>
      <div style="font-size: 12px; opacity: 0.8; margin-top: 5px;">
        ${getConfidenceLabel(confidence)}
      </div>
    </div>
  `;

  // Probabilities
  if (probabilities) {
    confidenceHTML += `
      <div style="font-size: 13px; margin-top: 15px;">
        <div style="font-weight: bold; margin-bottom: 10px;">📈 Kemungkinan Kategori Lain:</div>
    `;

    const sortedProbs = Object.entries(probabilities).slice(0, 4);
    
    sortedProbs.forEach(([category, prob], index) => {
      const isMainPrediction = index === 0;
      confidenceHTML += `
        <div style="margin: 8px 0; padding: 10px; background: rgba(255,255,255,${isMainPrediction ? '0.15' : '0.08'}); border-radius: 6px; ${isMainPrediction ? 'border-left: 3px solid #4CAF50;' : ''}">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-weight: ${isMainPrediction ? 'bold' : 'normal'};">
              ${isMainPrediction ? '✓ ' : ''}${formatCategoryName(category)}
            </span>
            <span style="font-weight: bold; ${isMainPrediction ? 'color: #4CAF50;' : ''}">
              ${prob.toFixed(1)}%
            </span>
          </div>
          <div style="width: 100%; height: 4px; background: rgba(255,255,255,0.2); border-radius: 2px; margin-top: 5px; overflow: hidden;">
            <div style="width: ${prob}%; height: 100%; background: ${isMainPrediction ? '#4CAF50' : '#2196F3'}; transition: width 0.3s;"></div>
          </div>
        </div>
      `;
    });

    confidenceHTML += `</div>`;
  }

  confidenceText.innerHTML = confidenceHTML;

  // Apply styling
  resultSection.classList.remove("clearsky", "rain", "cloud", "warning");

  if (prediction === "4_clearsky") {
    resultSection.classList.add("clearsky");
  } else if (prediction === "CONVECTIVE") {
    resultSection.classList.add("warning");
  } else if (prediction === "LOW_CLOUD") {
    resultSection.classList.add("rain");
  } else {
    resultSection.classList.add("cloud");
  }

  resultSection.classList.add("show");

  // Smooth scroll to result
  setTimeout(() => {
    resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, 100);
}

/**
 * NEW: Build preprocessing info display
 */
function buildPreprocessingInfo(fisheyeInfo) {
  if (!fisheyeInfo.fisheye_detected) {
    // Jika tidak terdeteksi fisheye, tidak perlu tampilkan (optional)
    return '';
  }

  const detected = fisheyeInfo.fisheye_detected;
  const confidence = fisheyeInfo.fisheye_confidence;
  const corrected = fisheyeInfo.correction_applied;
  const strength = fisheyeInfo.correction_strength;

  let html = `
    <div style="background: rgba(100, 150, 255, 0.15); padding: 12px; border-radius: 8px; margin-bottom: 15px; border-left: 3px solid #64b5f6;">
      <div style="font-weight: bold; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
        <span>📸</span>
        <span>Preprocessing Gambar</span>
      </div>
      <div style="font-size: 13px; line-height: 1.6;">
  `;

  if (detected) {
    html += `
      <div style="margin-bottom: 6px;">
        ✓ Distorsi fisheye terdeteksi (${confidence.toFixed(1)}%)
      </div>
    `;

    if (corrected) {
      html += `
        <div style="margin-bottom: 6px; color: #a5d6a7;">
          ✓ Koreksi distorsi diterapkan (strength: ${strength})
        </div>
        <div style="font-size: 12px; opacity: 0.8; margin-top: 8px; font-style: italic;">
          Gambar Anda dari kamera wide-angle telah dioptimalkan untuk akurasi yang lebih baik
        </div>
      `;
    } else {
      html += `
        <div style="opacity: 0.8;">
          ℹ️ Koreksi tidak diperlukan
        </div>
      `;
    }
  }

  html += `
      </div>
    </div>
  `;

  return html;
}

/**
 * Show error for non-sky images
 */
function showNotSkyError(result) {
  const detail = result.detail || {};
  
  let suggestionHTML = '';
  if (detail.suggestion) {
    suggestionHTML = `
      <div style="margin: 15px 0; font-size: 14px; line-height: 1.6;">
        💡 ${detail.suggestion}
      </div>
    `;
  }

  // NEW: Show preprocessing info even in error
  let preprocessingHTML = '';
  if (detail.preprocessing && detail.preprocessing.fisheye) {
    const fisheye = detail.preprocessing.fisheye;
    if (fisheye.fisheye_detected) {
      preprocessingHTML = `
        <div style="margin-top: 15px; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 6px; font-size: 12px;">
          📸 Fisheye terdeteksi dan dikoreksi, namun gambar tetap bukan langit/awan
        </div>
      `;
    }
  }

  errorMessage.innerHTML = `
    <div style="text-align: center; padding: 20px;">
      <div style="font-size: 48px; margin-bottom: 15px;">🚫</div>
      <div style="font-size: 18px; font-weight: bold; margin-bottom: 10px;">
        Bukan Gambar Langit
      </div>
      <div style="margin: 15px 0; font-size: 14px; line-height: 1.6;">
        ${result.message}
      </div>
      ${suggestionHTML}
      ${preprocessingHTML}
      <div style="margin-top: 15px; font-size: 12px; opacity: 0.7;">
        Confidence: ${detail.sky_confidence}%
      </div>
    </div>
  `;
  errorMessage.classList.add("show");

  setTimeout(() => {
    hideError();
  }, 8000);
}

/**
 * Show error message
 */
function showError(message) {
  errorMessage.innerHTML = `
    <div style="text-align: center;">
      <div style="font-size: 18px; font-weight: bold;">❌ Error</div>
      <div style="margin-top: 8px;">${message}</div>
    </div>
  `;
  errorMessage.classList.add("show");

  setTimeout(() => {
    hideError();
  }, 5000);
}

/**
 * Hide error message
 */
function hideError() {
  errorMessage.classList.remove("show");
  errorMessage.innerHTML = "";
}

/**
 * Helper: Format category name for display
 */
function formatCategoryName(category) {
  const names = {
    'HIGH_CLOUD': 'High-Level Clouds',
    'MID_CLOUD': 'Mid-Level Clouds',
    'LOW_CLOUD': 'Low-Level Clouds',
    'CONVECTIVE': 'Convective Clouds',
    '4_clearsky': 'Clear Sky'
  };
  return names[category] || category.replace(/_/g, ' ');
}

/**
 * Helper: Get confidence color
 */
function getConfidenceColor(confidence) {
  if (confidence >= 80) return '#4CAF50';
  if (confidence >= 60) return '#FFC107';
  return '#FF9800';
}

/**
 * Helper: Get confidence label
 */
function getConfidenceLabel(confidence) {
  if (confidence >= 90) return 'Sangat Yakin';
  if (confidence >= 75) return 'Yakin';
  if (confidence >= 60) return 'Cukup Yakin';
  return 'Kurang Yakin';
}

/**
 * Check backend connection and features
 */
async function checkBackendConnection() {
  try {
    const response = await fetch(API_URL.replace("/predict", ""));
    const result = await response.json();
    
    if (result.status === "online") {
      console.log("✓ Backend connected - " + result.service);
      
      // NEW: Check fisheye preprocessing feature
      if (result.features && result.features.fisheye_preprocessing) {
        console.log("✓ Fisheye preprocessing: ENABLED");
      }
      
      // Optional: Get config
      try {
        const configResponse = await fetch(API_URL.replace("/predict", "/config"));
        const config = await configResponse.json();
        
        if (config.status === "success") {
          console.log("Backend Config:", config.config);
        }
      } catch (e) {
        // Config endpoint might not exist in older versions
      }
    }
  } catch (error) {
    console.warn("⚠ Backend not connected");
  }
}

// Initialize
checkBackendConnection();

// Optional: Add visual feedback when camera is being used
if (cameraInput) {
  cameraInput.addEventListener('click', function() {
    console.log("📸 Opening camera...");
  });
}