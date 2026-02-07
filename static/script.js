// Cloud Classification Frontend - Updated for Fisheye Preprocessing
// API URL - Sesuaikan dengan backend Anda
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
const warningMessage = document.getElementById("warningMessage");

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
    hideWarning();
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
  hideWarning();

  try {
    const formData = new FormData();
    formData.append("image", selectedFile);

    const response = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });

    const result = await response.json();

    console.log("API Response:", result);

    if (result.status === "success") {
      showResult(result.data);
    } else if (result.error_type === "not_sky_image") {
      showNotSkyError(result);
    } else {
      showError(result.message || "Terjadi kesalahan saat prediksi");
    }
  } catch (error) {
    console.error("Error:", error);
    showError("Gagal terhubung ke server: " + error.message);
  } finally {
    predictBtn.textContent = "Klasifikasi Awan";
    predictBtn.disabled = false;
  }
}

/**
 * Show classification result
 */
function showResult(data) {
  const { 
    prediction, 
    confidence, 
    cloud_info,
    probabilities,
    preprocessing
  } = data;

  const resultIcon = document.getElementById("resultIcon");
  const resultTitle = document.getElementById("resultTitle");
  const resultDescription = document.getElementById("resultDescription");
  const confidenceText = document.getElementById("confidenceText");

  // Set icon
  resultIcon.textContent = cloud_info.icon || "☁️";

  // Set title (cloud types)
  const cloudTypes = cloud_info.cloud_types || ['Unknown'];
  if (cloudTypes.length === 1) {
    resultTitle.innerHTML = cloudTypes[0].toUpperCase();
  } else {
    resultTitle.innerHTML = cloudTypes.join(" / ").toUpperCase();
  }

  // Build description
  let descriptionHTML = '';

  // Preprocessing info (NEW)
  if (preprocessing && (preprocessing.fisheye_applied || preprocessing.contour_applied)) {
    descriptionHTML += `
      <div style="background: rgba(100, 150, 255, 0.2); padding: 12px; border-radius: 8px; margin-bottom: 15px; border-left: 3px solid #64b5f6;">
        <div style="font-weight: bold; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
          <span>🔧</span>
          <span>Preprocessing Applied</span>
        </div>
        <div style="font-size: 13px; line-height: 1.6;">
          ${preprocessing.fisheye_applied ? '<div>✓ Fisheye Conversion</div>' : ''}
          ${preprocessing.contour_applied ? '<div>✓ Contour Enhancement</div>' : ''}
        </div>
      </div>
    `;
  }

  // Main explanation
  descriptionHTML += `
    <div style="margin-bottom: 20px; font-size: 15px;">
      ${cloud_info.explanation}
    </div>
  `;

  // Characteristics
  if (cloud_info.characteristics && cloud_info.characteristics.length > 0) {
    descriptionHTML += `
      <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
        <div style="font-weight: bold; margin-bottom: 10px;">🔍 Karakteristik:</div>
        <div style="font-size: 13px; line-height: 1.8;">
          ${cloud_info.characteristics.map(char => `<div style="margin-bottom: 5px;">${char}</div>`).join('')}
        </div>
      </div>
    `;
  }

  // Altitude
  descriptionHTML += `
    <div style="font-size: 14px; margin-bottom: 12px; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 6px;">
      📏 <strong>Ketinggian:</strong> ${cloud_info.altitude}
    </div>
  `;

  // Weather forecast
  descriptionHTML += `
    <div style="background: rgba(100, 200, 255, 0.2); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
      <div style="font-weight: bold; margin-bottom: 8px;">🌤️ Prediksi Cuaca:</div>
      <div style="font-size: 14px; line-height: 1.6;">
        ${cloud_info.weather}
      </div>
    </div>
  `;

  // Precipitation
  descriptionHTML += `
    <div style="font-size: 13px; margin-bottom: 12px; background: rgba(255,255,255,0.08); padding: 10px; border-radius: 6px;">
      💧 <strong>Potensi Hujan:</strong> ${cloud_info.precipitation}
    </div>
  `;

  resultDescription.innerHTML = descriptionHTML;

  // Confidence section
  let confidenceHTML = `
    <div style="margin-bottom: 15px; padding: 12px; background: rgba(255,255,255,0.1); border-radius: 8px;">
      <div style="font-weight: bold; margin-bottom: 8px;">📊 Tingkat Keyakinan</div>
      <div style="font-size: 28px; font-weight: bold; color: ${getConfidenceColor(confidence)};">
        ${confidence.toFixed(1)}%
      </div>
      <div style="font-size: 12px; opacity: 0.8; margin-top: 5px;">
        ${getConfidenceLabel(confidence)}
      </div>
    </div>
  `;

  // Probabilities
  if (probabilities) {
    const sortedProbs = Object.entries(probabilities).slice(0, 4);
    
    confidenceHTML += `
      <div style="font-size: 13px; margin-top: 15px;">
        <div style="font-weight: bold; margin-bottom: 10px;">📈 Kemungkinan Lain:</div>
    `;
    
    sortedProbs.forEach(([category, prob], index) => {
      const isMainPrediction = index === 0;
      confidenceHTML += `
        <div style="margin: 8px 0; padding: 10px; background: rgba(255,255,255,${isMainPrediction ? '0.15' : '0.08'}); border-radius: 6px;">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-weight: ${isMainPrediction ? 'bold' : 'normal'};">
              ${isMainPrediction ? '✓ ' : ''}${formatCategoryName(category)}
            </span>
            <span style="font-weight: bold;">
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

  // Apply styling based on prediction
  resultSection.classList.remove("clearsky", "rain", "cloud", "warning");

  if (prediction === "4_clearsky") {
    resultSection.classList.add("clearsky");
  } else if (prediction === "6_cumulonimbus_nimbostratus") {
    resultSection.classList.add("warning");
  } else if (prediction.includes("stratus") || prediction.includes("nimbus")) {
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
 * Show error for non-sky images
 */
function showNotSkyError(result) {
  const detail = result.detail || {};
  
  errorMessage.innerHTML = `
    <div style="text-align: center; padding: 20px;">
      <div style="font-size: 48px; margin-bottom: 15px;">🚫</div>
      <div style="font-size: 18px; font-weight: bold; margin-bottom: 10px;">
        Bukan Gambar Langit/Awan
      </div>
      <div style="margin: 15px 0; font-size: 14px; line-height: 1.6;">
        ${result.message}
      </div>
      ${detail.suggestion ? `
        <div style="margin: 15px 0; font-size: 14px; line-height: 1.6;">
          💡 ${detail.suggestion}
        </div>
      ` : ''}
      <div style="margin-top: 15px; font-size: 12px; opacity: 0.7;">
        Confidence: ${detail.sky_confidence || 0}%
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
}

/**
 * Hide warning message
 */
function hideWarning() {
  warningMessage.classList.remove("show");
}

/**
 * Helper: Format category name
 */
function formatCategoryName(category) {
  // Convert snake_case to readable format
  return category
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
    .replace(/\d+\s+/, '');  // Remove leading numbers
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

// Check backend connection on load
async function checkBackend() {
  try {
    const baseUrl = API_URL.replace('/api/predict', '');
    const response = await fetch(baseUrl);
    const result = await response.json();
    
    if (result.status === "online") {
      console.log("✓ Backend connected:", result.service);
      console.log("  Preprocessing:", result.preprocessing);
    }
  } catch (error) {
    console.warn("⚠ Backend not connected");
  }
}

// Initialize
checkBackend();