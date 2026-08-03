const backendUrl = "https://omnicataract-x-api.onrender.com";

const imageInput = document.getElementById("imageInput");
const uploadHint = document.getElementById("uploadHint");
const preview = document.getElementById("preview");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultBox = document.getElementById("resultBox");
const resultText = document.getElementById("resultText");

let selectedFile = null;

imageInput.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (!file) return;

  selectedFile = file;
  uploadHint.textContent = file.name;

  const reader = new FileReader();
  reader.onload = () => {
    preview.src = reader.result;
    preview.hidden = false;
  };
  reader.readAsDataURL(file);
});

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) {
    alert("Please choose an image first.");
    return;
  }

  analyzeBtn.disabled = true;
  resultBox.hidden = true;
  resultText.textContent = "Analyzing...";

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch(`${backendUrl}/predict`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(error || "Request failed");
    }

    const result = await response.json();
    resultBox.hidden = false;
    resultText.innerHTML = `
      <p><strong>Cataract detected:</strong> ${result.cataract_detected ? "Yes" : "No"}</p>
      <p><strong>Confidence:</strong> ${(result.cataract_confidence * 100).toFixed(1)}%</p>
      <p><strong>Quality:</strong> ${result.quality_status}</p>
      <p><strong>Severity:</strong> ${result.severity_grade}</p>
      <p><strong>Message:</strong> ${result.message}</p>
    `;
  } catch (error) {
    resultBox.hidden = false;
    resultText.textContent = `Error: ${error.message}`;
  } finally {
    analyzeBtn.disabled = false;
  }
});
