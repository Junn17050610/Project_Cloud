// // Cloud Classification Frontend JavaScript
// // Modified untuk menampilkan multiple cloud types explanation

// // Konfigurasi
// const API_URL = "https://projectcloud-production.up.railway.app/api/predict";  // Ganti dengan URL deploy Anda

// // Variabel global
// let selectedFile = null;

// // Element references
// const fileInput = document.getElementById("fileInput");
// const cameraInput = document.getElementById("cameraInput");
// const previewSection = document.getElementById("previewSection");
// const imagePreview = document.getElementById("imagePreview");
// const predictBtn = document.getElementById("predictBtn");
// const resultSection = document.getElementById("resultSection");
// const errorMessage = document.getElementById("errorMessage");

// // Event listeners
// fileInput.addEventListener("change", handleImageSelect);
// cameraInput.addEventListener("change", handleImageSelect);
// predictBtn.addEventListener("click", handlePredict);

// /**
//  * Handle image selection
//  */
// function handleImageSelect(e) {
//   const file = e.target.files[0];

//   if (!file) return;

//   // Validasi
//   if (!file.type.startsWith("image/")) {
//     showError("File yang dipilih bukan gambar!");
//     return;
//   }

//   if (file.size > 10 * 1024 * 1024) {
//     showError("Ukuran file terlalu besar! Maksimal 10MB");
//     return;
//   }

//   selectedFile = file;

//   // Preview
//   const reader = new FileReader();
//   reader.onload = function (event) {
//     imagePreview.src = event.target.result;
//     imagePreview.style.display = "block";
//     previewSection.querySelector(".placeholder").style.display = "none";
//     predictBtn.disabled = false;

//     resultSection.classList.remove("show");
//     hideError();
//   };
//   reader.readAsDataURL(file);
// }

// /**
//  * Handle prediction
//  */
// async function handlePredict() {
//   if (!selectedFile) {
//     showError("Tidak ada gambar yang dipilih!");
//     return;
//   }

//   predictBtn.textContent = "Memproses...";
//   predictBtn.disabled = true;
//   hideError();

//   try {
//     const formData = new FormData();
//     formData.append("image", selectedFile);

//     const response = await fetch(API_URL, {
//       method: "POST",
//       body: formData,
//     });

//     const result = await response.json();

//     console.log("API Response:", result);

//     if (result.status === "success") {
//       const data = result.data;
//       showCloudResult(
//         data.prediction,
//         data.confidence,
//         data.cloud_info,
//         data.probabilities
//       );
//     } else {
//       showError(result.message || "Terjadi kesalahan saat prediksi");
//     }
//   } catch (error) {
//     console.error("Error:", error);
//     showError("Error: " + error.message);
//   } finally {
//     predictBtn.textContent = "Klasifikasi Awan";
//     predictBtn.disabled = false;
//   }
// }

// /**
//  * Show cloud classification result
//  */
// function showCloudResult(prediction, confidence, cloudInfo, probabilities) {
//   const resultIcon = document.getElementById("resultIcon");
//   const resultTitle = document.getElementById("resultTitle");
//   const resultDescription = document.getElementById("resultDescription");
//   const confidenceText = document.getElementById("confidenceText");

//   console.log("Cloud Info:", cloudInfo);

//   // Set icon
//   resultIcon.textContent = cloudInfo.icon || "☁️";

//   // Set title
//   const cloudTypes = cloudInfo.cloud_types || ["Unknown"];
//   if (cloudTypes.length === 1) {
//     resultTitle.textContent = cloudTypes[0].toUpperCase();
//   } else {
//     resultTitle.textContent = "MULTIPLE CLOUD TYPES";
//   }

//   // Set description (HTML dengan penjelasan)
//   let descriptionHTML = `
//     <div style="margin-bottom: 15px;">
//       ${cloudInfo.explanation || "Cloud type detected"}
//     </div>
//   `;

//   // Add altitude info
//   if (cloudInfo.altitude) {
//     descriptionHTML += `
//       <div style="font-size: 14px; opacity: 0.9; margin-bottom: 10px;">
//         📏 <strong>Ketinggian:</strong> ${cloudInfo.altitude}
//       </div>
//     `;
//   }

//   // Add weather info
//   if (cloudInfo.weather) {
//     descriptionHTML += `
//       <div style="font-size: 14px; opacity: 0.9; background: rgba(255,255,255,0.2); padding: 10px; border-radius: 5px;">
//         🌤️ <strong>Prediksi Cuaca:</strong> ${cloudInfo.weather}
//       </div>
//     `;
//   }

//   resultDescription.innerHTML = descriptionHTML;

//   // Confidence
//   let confidenceHTML = `
//     <div style="margin-bottom: 10px;">
//       <strong>Tingkat Kepercayaan:</strong> ${confidence.toFixed(2)}%
//     </div>
//   `;

//   // Show top 3 probabilities
//   if (probabilities) {
//     // Sort probabilities
//     const sortedProbs = Object.entries(probabilities)
//       .sort((a, b) => b[1] - a[1])
//       .slice(0, 3);

//     confidenceHTML += `
//       <div style="font-size: 12px; margin-top: 10px; text-align: left;">
//         <strong>Top 3 Kemungkinan:</strong><br>
//     `;

//     sortedProbs.forEach(([className, prob], index) => {
//       // Clean class name untuk display
//       const cleanName = className
//         .replace(/^\d+_/, "")
//         .replace(/_/g, " ")
//         .split(" ")
//         .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
//         .join(" ");

//       confidenceHTML += `
//         <div style="margin: 5px 0; padding: 5px; background: rgba(255,255,255,0.1); border-radius: 3px;">
//           ${index + 1}. ${cleanName}: ${prob.toFixed(2)}%
//         </div>
//       `;
//     });

//     confidenceHTML += `</div>`;
//   }

//   confidenceText.innerHTML = confidenceHTML;

//   // Show result dengan styling
//   resultSection.classList.remove("clearsky", "rain", "cloud");

//   // Apply styling berdasarkan tipe awan
//   if (prediction.includes("clearsky")) {
//     resultSection.classList.add("clearsky");
//   } else if (
//     prediction.includes("cumulonimbus") ||
//     prediction.includes("nimbostratus")
//   ) {
//     resultSection.classList.add("rain");
//   } else {
//     resultSection.classList.add("cloud");
//   }

//   resultSection.classList.add("show");
// }

// /**
//  * Show error message
//  */
// function showError(message) {
//   errorMessage.innerHTML = `
//     <div style="text-align: center;">
//       <div style="font-size: 18px; font-weight: bold;">❌ Error</div>
//       <div style="margin-top: 8px;">${message}</div>
//     </div>
//   `;
//   errorMessage.classList.add("show");

//   setTimeout(() => {
//     hideError();
//   }, 5000);
// }

// /**
//  * Hide error message
//  */
// function hideError() {
//   errorMessage.classList.remove("show");
//   errorMessage.innerHTML = "";
// }

// // Check backend connection
// async function checkBackendConnection() {
//   try {
//     const response = await fetch(API_URL.replace("/predict", ""));
//     const result = await response.json();
//     if (result.status === "online") {
//       console.log("✓ Backend connected");
//     }
//   } catch (error) {
//     console.warn("⚠ Backend not connected");
//   }
// }

// checkBackendConnection();

// Cloud Weather Prediction Frontend - Version 2.0
// Enhanced with detailed cloud information and weather forecasting

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
  const { prediction, specific_cloud_type, confidence, explanation, probabilities } = data;

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
  let descriptionHTML = `
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
 * Show error for non-sky images
 */
function showNotSkyError(result) {
  errorMessage.innerHTML = `
    <div style="text-align: center; padding: 20px;">
      <div style="font-size: 48px; margin-bottom: 15px;">🚫</div>
      <div style="font-size: 18px; font-weight: bold; margin-bottom: 10px;">
        Bukan Gambar Langit
      </div>
      <div style="margin: 15px 0; font-size: 14px; line-height: 1.6;">
        ${result.message}
      </div>
      <div style="margin-top: 15px; font-size: 12px; opacity: 0.7;">
        Confidence: ${result.detail.sky_confidence}%
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
    '7_contrail': 'Contrail',
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
 * Check backend connection
 */
async function checkBackendConnection() {
  try {
    const response = await fetch(API_URL.replace("/predict", ""));
    const result = await response.json();
    if (result.status === "online") {
      console.log("✓ Backend connected - " + result.service);
    }
  } catch (error) {
    console.warn("⚠ Backend not connected");
  }
}

checkBackendConnection();