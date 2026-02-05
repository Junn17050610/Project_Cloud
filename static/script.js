// Cloud Classification Frontend JavaScript
// Modified untuk menampilkan multiple cloud types explanation

// Konfigurasi
const API_URL = "http://localhost:5000/api/predict";  // Ganti dengan URL deploy Anda

// Variabel global
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

  // Validasi
  if (!file.type.startsWith("image/")) {
    showError("File yang dipilih bukan gambar!");
    return;
  }

  if (file.size > 10 * 1024 * 1024) {
    showError("Ukuran file terlalu besar! Maksimal 10MB");
    return;
  }

  selectedFile = file;

  // Preview
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

  predictBtn.textContent = "Memproses...";
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
      const data = result.data;
      showCloudResult(
        data.prediction,
        data.confidence,
        data.cloud_info,
        data.probabilities
      );
    } else {
      showError(result.message || "Terjadi kesalahan saat prediksi");
    }
  } catch (error) {
    console.error("Error:", error);
    showError("Error: " + error.message);
  } finally {
    predictBtn.textContent = "Klasifikasi Awan";
    predictBtn.disabled = false;
  }
}

/**
 * Show cloud classification result
 */
function showCloudResult(prediction, confidence, cloudInfo, probabilities) {
  const resultIcon = document.getElementById("resultIcon");
  const resultTitle = document.getElementById("resultTitle");
  const resultDescription = document.getElementById("resultDescription");
  const confidenceText = document.getElementById("confidenceText");

  console.log("Cloud Info:", cloudInfo);

  // Set icon
  resultIcon.textContent = cloudInfo.icon || "☁️";

  // Set title
  const cloudTypes = cloudInfo.cloud_types || ["Unknown"];
  if (cloudTypes.length === 1) {
    resultTitle.textContent = cloudTypes[0].toUpperCase();
  } else {
    resultTitle.textContent = "MULTIPLE CLOUD TYPES";
  }

  // Set description (HTML dengan penjelasan)
  let descriptionHTML = `
    <div style="margin-bottom: 15px;">
      ${cloudInfo.explanation || "Cloud type detected"}
    </div>
  `;

  // Add altitude info
  if (cloudInfo.altitude) {
    descriptionHTML += `
      <div style="font-size: 14px; opacity: 0.9; margin-bottom: 10px;">
        📏 <strong>Ketinggian:</strong> ${cloudInfo.altitude}
      </div>
    `;
  }

  // Add weather info
  if (cloudInfo.weather) {
    descriptionHTML += `
      <div style="font-size: 14px; opacity: 0.9; background: rgba(255,255,255,0.2); padding: 10px; border-radius: 5px;">
        🌤️ <strong>Prediksi Cuaca:</strong> ${cloudInfo.weather}
      </div>
    `;
  }

  resultDescription.innerHTML = descriptionHTML;

  // Confidence
  let confidenceHTML = `
    <div style="margin-bottom: 10px;">
      <strong>Tingkat Kepercayaan:</strong> ${confidence.toFixed(2)}%
    </div>
  `;

  // Show top 3 probabilities
  if (probabilities) {
    // Sort probabilities
    const sortedProbs = Object.entries(probabilities)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3);

    confidenceHTML += `
      <div style="font-size: 12px; margin-top: 10px; text-align: left;">
        <strong>Top 3 Kemungkinan:</strong><br>
    `;

    sortedProbs.forEach(([className, prob], index) => {
      // Clean class name untuk display
      const cleanName = className
        .replace(/^\d+_/, "")
        .replace(/_/g, " ")
        .split(" ")
        .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
        .join(" ");

      confidenceHTML += `
        <div style="margin: 5px 0; padding: 5px; background: rgba(255,255,255,0.1); border-radius: 3px;">
          ${index + 1}. ${cleanName}: ${prob.toFixed(2)}%
        </div>
      `;
    });

    confidenceHTML += `</div>`;
  }

  confidenceText.innerHTML = confidenceHTML;

  // Show result dengan styling
  resultSection.classList.remove("clearsky", "rain", "cloud");

  // Apply styling berdasarkan tipe awan
  if (prediction.includes("clearsky")) {
    resultSection.classList.add("clearsky");
  } else if (
    prediction.includes("cumulonimbus") ||
    prediction.includes("nimbostratus")
  ) {
    resultSection.classList.add("rain");
  } else {
    resultSection.classList.add("cloud");
  }

  resultSection.classList.add("show");
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

// Check backend connection
async function checkBackendConnection() {
  try {
    const response = await fetch(API_URL.replace("/predict", ""));
    const result = await response.json();
    if (result.status === "online") {
      console.log("✓ Backend connected");
    }
  } catch (error) {
    console.warn("⚠ Backend not connected");
  }
}

checkBackendConnection();
