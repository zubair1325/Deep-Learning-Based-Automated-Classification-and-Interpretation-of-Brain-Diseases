import { useMemo, useState } from "react";
import "./App.css";

const diseaseOptions = [
  "Alzheimer MRI",
  "Brain Stroke CT Scan",
  "Brain Tumor CT Scan",
  "Brain Tumor MRI",
  "Brain Tumor MRI and CT Scan",
  "Parkinsons MRI",
];

function App() {
  const [selectedDisease, setSelectedDisease] = useState("Alzheimer MRI");
  const [imageFile, setImageFile] = useState(null);
  const [secondaryImageFile, setSecondaryImageFile] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");

  const selectedLabel = useMemo(
    () => selectedDisease.replace(/\s+/g, "_").toLowerCase(),
    [selectedDisease],
  );
  const needsDualUpload = selectedDisease === "Brain Tumor MRI and CT Scan";

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!imageFile) {
      setErrorMessage("Please select an image to continue.");
      return;
    }

    if (needsDualUpload && !secondaryImageFile) {
      setErrorMessage(
        "Please upload both MRI and CT images for the combined tumor workflow.",
      );
      return;
    }

    setErrorMessage("");
    setIsSubmitting(true);
    setResult(null);

    const formData = new FormData();
    formData.append("disease", selectedDisease);

    if (needsDualUpload) {
      formData.append("mri_image", imageFile);
      formData.append("ct_image", secondaryImageFile);
    } else {
      formData.append("image", imageFile);
    }

    try {
      const response = await fetch("/api/predict/", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }

      setResult(data);
    } catch (submissionError) {
      setErrorMessage(
        submissionError.message ||
          "Something went wrong while processing the image.",
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <p className="eyebrow">BrainScan</p>
        <h1>Brain disease classification and interpretation</h1>
        <p className="hero-copy">
          Select a disease type, upload one scan for standard categories, or
          upload both MRI and CT scans for the combined Brain Tumor MRI + CT
          workflow. The backend then runs the winner-model pipeline, returns the
          confidence score, and visualizes the Grad-CAM evidence.
        </p>
      </section>

      <form className="upload-card" onSubmit={handleSubmit}>
        <label className="field-label" htmlFor="disease-select">
          Disease category
        </label>
        <select
          id="disease-select"
          value={selectedDisease}
          onChange={(event) => setSelectedDisease(event.target.value)}
        >
          {diseaseOptions.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>

        <label className="field-label" htmlFor="image-input">
          {needsDualUpload
            ? "Upload MRI image"
            : `Upload image for ${selectedDisease}`}
        </label>
        <input
          id="image-input"
          type="file"
          accept="image/*"
          onChange={(event) => setImageFile(event.target.files?.[0] || null)}
        />

        {needsDualUpload ? (
          <>
            <label className="field-label" htmlFor="secondary-image-input">
              Upload CT image
            </label>
            <input
              id="secondary-image-input"
              type="file"
              accept="image/*"
              onChange={(event) =>
                setSecondaryImageFile(event.target.files?.[0] || null)
              }
            />
          </>
        ) : null}

        <button type="submit" disabled={isSubmitting}>
          {isSubmitting ? "Processing image…" : "Submit"}
        </button>

        {errorMessage ? <p className="error-text">{errorMessage}</p> : null}
      </form>

      <section className="result-card">
        <div className="status-bar">
          <span className="status-pill">{selectedDisease}</span>
          <span className="status-pill">Mode: {selectedLabel}</span>
        </div>

        {isSubmitting ? (
          <div className="progress-box">
            <div className="spinner" aria-hidden="true" />
            <p>Your image is being processed in the background.</p>
          </div>
        ) : null}

        {result ? (
          <div className="result-grid">
            <article>
              <h2>Prediction</h2>
              <p className="prediction-name">{result.prediction}</p>
              <p className="confidence-score">
                Confidence score: {result.confidence_score}
              </p>
              <div className="explanation-block">
                <h3>AI explanation</h3>
                <p>{result.explanation}</p>
              </div>
            </article>

            <article>
              <h2>Grad-CAM</h2>
              {result.grad_cam ? (
                <img
                  className="gradcam-image"
                  src={`data:image/png;base64,${result.grad_cam}`}
                  alt="Grad-CAM activation map"
                />
              ) : (
                <p>No Grad-CAM generated.</p>
              )}
            </article>
          </div>
        ) : null}
      </section>
    </main>
  );
}

export default App;
