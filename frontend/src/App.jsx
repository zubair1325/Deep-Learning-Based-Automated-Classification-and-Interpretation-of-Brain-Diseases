import { useMemo, useState } from "react";
import { z } from "zod";
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

  const explanationSchema = z.object({
    summary: z.string(),
    detailed_visual_description: z.string(),
    imaging_findings: z.string(),
    measurements: z.string().optional(),
    differential_diagnosis: z.string(),
    interpretation: z.string(),
    suggested_urgency: z.string().optional(),
    recommended_next_steps: z.string().optional(),
    treatment_options: z.string().optional(),
    confidence: z.union([z.number(), z.string()]).optional(),
    limitations: z.string().optional(),
    image_references: z.record(z.any()).optional(),
    references: z.string().optional(),
    notes_for_provider: z.string().optional(),
  });

  const tryParseExplanation = (rawValue) => {
    if (rawValue == null) {
      return null;
    }

    if (typeof rawValue === "string") {
      const sanitized = (text) => {
        const firstBrace = text.indexOf("{");
        const lastBrace = text.lastIndexOf("}");
        if (firstBrace !== -1 && lastBrace !== -1 && firstBrace < lastBrace) {
          const candidate = text.slice(firstBrace, lastBrace + 1);
          try {
            return JSON.parse(candidate);
          } catch {
            // fall through
          }
        }

        try {
          return JSON.parse(text);
        } catch {
          return null;
        }
      };

      return sanitized(rawValue);
    }

    if (typeof rawValue === "object") {
      return rawValue;
    }

    return null;
  };

  const getImageSrc = (value) => {
    if (!value || typeof value !== "string") {
      return undefined;
    }

    if (value.startsWith("data:image/")) {
      return value;
    }

    if (/^[A-Za-z0-9+/=]+$/.test(value)) {
      return `data:image/png;base64,${value}`;
    }

    return value;
  };

  const toTitle = (key) =>
    key.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());

  const renderList = (text) => {
    if (!text) return [];
    return text
      .split(/\r?\n/)
      .map((item) => item.trim())
      .filter(Boolean);
  };

  const renderExplanationValue = (value) => {
    if (Array.isArray(value)) {
      return (
        <ul className="detail-list">
          {value.map((item, index) => (
            <li key={index}>{String(item)}</li>
          ))}
        </ul>
      );
    }

    if (typeof value === "object" && value !== null) {
      return (
        <pre className="explanation-json">{JSON.stringify(value, null, 2)}</pre>
      );
    }

    return <p>{String(value)}</p>;
  };

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

  const parsedExplanation = result
    ? tryParseExplanation(result.explanation)
    : null;

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
              {result.explanation_source ? (
                <p className="explanation-source">
                  Explanation source:{" "}
                  {result.explanation_source === "AI"
                    ? "AI-generated"
                    : "Rule-based fallback"}
                </p>
              ) : null}
              <div className="warning-block">
                <strong>Note:</strong> These results are for informational
                purposes only and may not always be correct. Please consult a
                healthcare professional for final interpretation.
              </div>
              <div className="explanation-block">
                <h3>AI explanation</h3>
                {parsedExplanation?.retried || result?.explanation?.retried ? (
                  <div className="continuation-warning">
                    Partial response recovered via continuation retry.
                  </div>
                ) : null}

                {/* Show uploaded image(s) and Grad-CAM when image_references present */}
                {parsedExplanation?.image_references ||
                (result?.explanation &&
                  typeof result.explanation === "object" &&
                  result.explanation.image_references) ? (
                  <section className="explanation-section">
                    <h4>{toTitle("image_references")}</h4>
                    <div className="split-columns">
                      <div>
                        <h5>Uploaded image</h5>
                        {needsDualUpload ? (
                          <>
                            {getImageSrc(
                              parsedExplanation?.image_references?.original_mri,
                            ) ||
                            getImageSrc(
                              result?.explanation?.image_references
                                ?.original_mri,
                            ) ||
                            (imageFile
                              ? URL.createObjectURL(imageFile)
                              : undefined) ? (
                              <figure>
                                <img
                                  className="gradcam-image"
                                  src={
                                    getImageSrc(
                                      parsedExplanation?.image_references
                                        ?.original_mri,
                                    ) ||
                                    getImageSrc(
                                      result?.explanation?.image_references
                                        ?.original_mri,
                                    ) ||
                                    (imageFile
                                      ? URL.createObjectURL(imageFile)
                                      : undefined)
                                  }
                                  alt="Uploaded MRI"
                                />
                                <figcaption>MRI upload</figcaption>
                              </figure>
                            ) : (
                              <p>No MRI uploaded</p>
                            )}

                            {getImageSrc(
                              parsedExplanation?.image_references?.original_ct,
                            ) ||
                            getImageSrc(
                              result?.explanation?.image_references
                                ?.original_ct,
                            ) ||
                            (secondaryImageFile
                              ? URL.createObjectURL(secondaryImageFile)
                              : undefined) ? (
                              <figure>
                                <img
                                  className="gradcam-image"
                                  src={
                                    getImageSrc(
                                      parsedExplanation?.image_references
                                        ?.original_ct,
                                    ) ||
                                    getImageSrc(
                                      result?.explanation?.image_references
                                        ?.original_ct,
                                    ) ||
                                    (secondaryImageFile
                                      ? URL.createObjectURL(secondaryImageFile)
                                      : undefined)
                                  }
                                  alt="Uploaded CT"
                                />
                                <figcaption>CT upload</figcaption>
                              </figure>
                            ) : (
                              <p>No CT uploaded</p>
                            )}
                          </>
                        ) : getImageSrc(
                            parsedExplanation?.image_references?.original,
                          ) ||
                          getImageSrc(
                            result?.explanation?.image_references?.original,
                          ) ||
                          (imageFile
                            ? URL.createObjectURL(imageFile)
                            : undefined) ? (
                          <figure>
                            <img
                              className="gradcam-image"
                              src={
                                getImageSrc(
                                  parsedExplanation?.image_references?.original,
                                ) ||
                                getImageSrc(
                                  result?.explanation?.image_references
                                    ?.original,
                                ) ||
                                (imageFile
                                  ? URL.createObjectURL(imageFile)
                                  : undefined)
                              }
                              alt="Uploaded image"
                            />
                            <figcaption>Uploaded scan</figcaption>
                          </figure>
                        ) : (
                          <p>No uploaded image available</p>
                        )}
                      </div>

                      <div>
                        <h5>Grad-CAM</h5>
                        {needsDualUpload ? (
                          <>
                            {getImageSrc(
                              parsedExplanation?.image_references?.gradcam_mri,
                            ) ||
                            getImageSrc(
                              result?.explanation?.image_references
                                ?.gradcam_mri,
                            ) ||
                            getImageSrc(result?.grad_cam_mri) ? (
                              <figure>
                                <img
                                  className="gradcam-image"
                                  src={
                                    getImageSrc(
                                      parsedExplanation?.image_references
                                        ?.gradcam_mri,
                                    ) ||
                                    getImageSrc(
                                      result?.explanation?.image_references
                                        ?.gradcam_mri,
                                    ) ||
                                    getImageSrc(result?.grad_cam_mri)
                                  }
                                  alt="MRI Grad-CAM"
                                />
                                <figcaption>MRI Grad-CAM</figcaption>
                              </figure>
                            ) : (
                              <p>No MRI Grad-CAM</p>
                            )}

                            {getImageSrc(
                              parsedExplanation?.image_references?.gradcam_ct,
                            ) ||
                            getImageSrc(
                              result?.explanation?.image_references?.gradcam_ct,
                            ) ||
                            getImageSrc(result?.grad_cam_ct) ? (
                              <figure>
                                <img
                                  className="gradcam-image"
                                  src={
                                    getImageSrc(
                                      parsedExplanation?.image_references
                                        ?.gradcam_ct,
                                    ) ||
                                    getImageSrc(
                                      result?.explanation?.image_references
                                        ?.gradcam_ct,
                                    ) ||
                                    getImageSrc(result?.grad_cam_ct)
                                  }
                                  alt="CT Grad-CAM"
                                />
                                <figcaption>CT Grad-CAM</figcaption>
                              </figure>
                            ) : (
                              <p>No CT Grad-CAM</p>
                            )}
                          </>
                        ) : getImageSrc(
                            parsedExplanation?.image_references?.gradcam,
                          ) ||
                          getImageSrc(
                            result?.explanation?.image_references?.gradcam,
                          ) ||
                          getImageSrc(result?.grad_cam) ? (
                          <figure>
                            <img
                              className="gradcam-image"
                              src={
                                getImageSrc(
                                  parsedExplanation?.image_references?.gradcam,
                                ) ||
                                getImageSrc(
                                  result?.explanation?.image_references
                                    ?.gradcam,
                                ) ||
                                getImageSrc(result?.grad_cam)
                              }
                              alt="Grad-CAM"
                            />
                            <figcaption>Grad-CAM</figcaption>
                          </figure>
                        ) : (
                          <p>No Grad-CAM available</p>
                        )}
                      </div>
                    </div>
                  </section>
                ) : null}

                {parsedExplanation ? (
                  Object.keys(parsedExplanation)
                    .filter((k) => k !== "retried" && k !== "image_references")
                    .map((key) => (
                      <section className="explanation-section" key={key}>
                        <h4>{toTitle(key)}</h4>
                        {renderExplanationValue(parsedExplanation[key])}
                      </section>
                    ))
                ) : typeof result.explanation === "object" &&
                  result.explanation !== null ? (
                  <pre className="explanation-json">
                    {JSON.stringify(
                      Object.fromEntries(
                        Object.entries(result.explanation).filter(
                          ([k]) => k !== "retried",
                        ),
                      ),
                      null,
                      2,
                    )}
                  </pre>
                ) : (
                  <p>{result.explanation}</p>
                )}
              </div>
            </article>

            <article>
              <h2>Grad-CAM</h2>
              {needsDualUpload ? (
                <div className="dual-gradcam-grid">
                  {result.grad_cam_mri ? (
                    <figure>
                      <figcaption>MRI Grad-CAM</figcaption>
                      <img
                        className="gradcam-image"
                        src={`data:image/png;base64,${result.grad_cam_mri}`}
                        alt="MRI Grad-CAM activation map"
                      />
                    </figure>
                  ) : null}
                  {result.grad_cam_ct ? (
                    <figure>
                      <figcaption>CT Grad-CAM</figcaption>
                      <img
                        className="gradcam-image"
                        src={`data:image/png;base64,${result.grad_cam_ct}`}
                        alt="CT Grad-CAM activation map"
                      />
                    </figure>
                  ) : null}
                </div>
              ) : result.grad_cam ? (
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
