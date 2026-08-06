# BrainScan

BrainScan is a full-stack medical imaging application for brain disease classification and interpretation using MRI and CT scans. The project combines a React + Vite frontend with a Django backend and supports both single-image inference and a dual-image MRI + CT workflow for brain tumor analysis.

The current system can:

- classify brain disease categories from uploaded scans
- generate a confidence score
- create a Grad-CAM visualization
- return an explanation for the prediction
- optionally use the Google Gemini API for structured radiology-style JSON output

Supported disease categories:

- Alzheimer MRI
- Brain Stroke CT Scan
- Brain Tumor CT Scan
- Brain Tumor MRI
- Brain Tumor MRI and CT Scan
- Parkinsons MRI

---

## 1. Project purpose

BrainScan was built to demonstrate an end-to-end medical image analysis pipeline that mirrors a simplified clinical workflow:

1. A user selects a disease category in the frontend.
2. A scan image is uploaded.
3. The frontend sends the file to the Django API.
4. The backend preprocesses the image.
5. The appropriate saved PyTorch model is loaded from the repository.
6. The model predicts the class and confidence.
7. A Grad-CAM heatmap is generated for visual explanation.
8. A plain-language or Gemini-generated explanation is returned to the UI.

This project is intended for educational, research, and demonstration purposes and should not be used as a substitute for professional medical diagnosis.

---

## 1.1 Full request-to-response workflow

The complete BrainScan workflow includes every step from user upload to final frontend rendering:

1. The user chooses a disease category in the React UI.
2. The user uploads one image for single-image categories or two files for the MRI + CT workflow.
3. The frontend performs local validation and then sends a multipart HTTP POST request to `/api/predict/`.
4. The Django backend receives the request in `brainscan_backend/brainscan_api/views.py`.
5. The backend validates the selected disease label and the required uploaded files.
6. Uploaded files are opened as PIL images and converted to RGB.
7. The backend applies artifact removal to the raw scan image using OpenCV-based cleaning logic.
8. The cleaned scan is resized to 224x224 pixels.
9. The cleaned image is transformed to a PyTorch tensor and normalized with ImageNet statistics.
10. The selected model is loaded from `Saved Model/` and matched to the disease label.
11. The cleaned tensor is passed through the model to generate logits, probabilities, a predicted class, and a confidence score.
12. Grad-CAM is computed from the model’s feature maps and fused with the cleaned input image.
13. The backend generates a human-readable explanation, either from a rule-based fallback or via the Gemini API when configured.
14. The backend returns a JSON response containing prediction, confidence, Grad-CAM image data, explanation content, and optional image references.
15. The frontend receives the JSON response and renders the prediction card, the Grad-CAM overlay, and the explanation sections in the UI.

This makes the workflow explicit for both single-image and dual-image inference modes.

---

## 2. High-level architecture

### Frontend

- Framework: React + Vite
- Main UI: `frontend/src/App.jsx`
- Styling: `frontend/src/App.css`
- Package manager: npm
- Dev server: Vite

### Backend

- Framework: Django
- API app: `brainscan_backend/brainscan_api`
- Project settings: `brainscan_backend/brainscan_backend/settings.py`
- API routing: `brainscan_backend/brainscan_api/urls.py`
- Request handler: `brainscan_backend/brainscan_api/views.py`
- Inference engine: `brainscan_backend/brainscan_api/inference.py`

### Model storage

Saved model weights are stored under the `Saved Model/` directory.

---

## 3. Technology stack

### Python backend

- Python 3.10+
- Django
- Django REST Framework
- django-cors-headers
- python-dotenv
- PyTorch
- TorchVision
- OpenCV (`cv2`)
- Pillow (`PIL`)
- NumPy
- Google GenAI SDK (`google.genai`)

### Frontend

- React 19
- Vite
- Zod

---

## 4. Project structure

```text
.
├── README.md
├── manage.py
├── .env
├── backend/
├── brainscan_backend/
│   ├── manage.py
│   ├── db.sqlite3
│   ├── brainscan_api/
│   │   ├── inference.py
│   │   ├── views.py
│   │   ├── urls.py
│   │   └── models.py
│   └── brainscan_backend/
│       ├── settings.py
│       ├── urls.py
│       └── wsgi.py
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── App.jsx
│       ├── App.css
│       └── main.jsx
├── Saved Model/
├── Colab Files/
└── scripts/
```

---

## 5. Prerequisites

Before installing the project, make sure you have the following tools available:

- Python 3.10 or newer
- Node.js 18+ and npm
- A terminal with PowerShell, bash, or cmd
- Optional: a Google Gemini API key for AI explanations

If you are using Windows PowerShell, the commands below are written for that shell.

---

## 6. Installation

### 6.1 Clone the repository

```powershell
git clone <your-repo-url>
cd "Deep Learning-Based Automated Classification and Interpretation of Brain Diseases Using MRI and CT scan"
```

### 6.2 Create and activate a Python virtual environment

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

### 6.3 Install Python dependencies

Upgrade pip first:

```powershell
python -m pip install --upgrade pip
```

Install the required packages:

```powershell
pip install django djangorestframework django-cors-headers python-dotenv google-genai opencv-python pillow numpy torch torchvision
```

> Note: `torch` and `torchvision` are large packages. If you are using a GPU, install the CUDA-enabled build that matches your machine. The current project works with the CPU build on most systems.

### 6.4 Install frontend dependencies

```powershell
cd frontend
npm install
```

### 6.5 Create the environment file

Create a `.env` file in the project root if it does not already exist.

Example:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

- If `GEMINI_API_KEY` is not set, the backend will fall back to a safe rule-based explanation.
- If the key is present but invalid or inaccessible, the system still falls back gracefully.

---

## 7. How to run the project

### 7.1 Start the Django backend

From the project root:

```powershell
cd brainscan_backend
python manage.py migrate
python manage.py runserver 8000
```

The API will be available at:

- http://127.0.0.1:8000/api/predict/

### 7.2 Start the React frontend

Open a second terminal:

```powershell
cd frontend
npm run dev
```

The frontend will usually be available at:

- http://localhost:5173

The Vite dev server is configured to proxy `/api` requests to the Django backend.

---

## 8. API usage

### Endpoint

```http
POST /api/predict/
```

### Request format

The API accepts multipart form data.

#### Single-image disease categories

Send:

- `disease`: disease label
- `image`: uploaded image file

Example fields:

- `disease=Alzheimer MRI`
- `image=@scan.png`

#### Dual-image workflow

For `Brain Tumor MRI and CT Scan`, send:

- `disease=Brain Tumor MRI and CT Scan`
- `mri_image`: MRI file
- `ct_image`: CT file

This is the dual-input path used by the combined tumor workflow. The backend receives both files, processes them through the shared dual-head inference pipeline, and returns the combined result.

### Example with curl

Single-image example:

```powershell
curl -X POST "http://127.0.0.1:8000/api/predict/" -F "disease=Alzheimer MRI" -F "image=@C:\path\to\scan.png"
```

Dual-image example:

```powershell
curl -X POST "http://127.0.0.1:8000/api/predict/" -F "disease=Brain Tumor MRI and CT Scan" -F "mri_image=@C:\path\to\mri.png" -F "ct_image=@C:\path\to\ct.png"
```

### Response structure

The backend returns a JSON object similar to:

```json
{
  "prediction": "VeryMildDemented",
  "confidence_score": "0.9243",
  "grad_cam": "base64-encoded-png-string",
  "explanation": "Human-readable explanation or fallback text",
  "explanation_source": "AI",
  "grad_cam_mri": "base64-encoded-png-string",
  "grad_cam_ct": "base64-encoded-png-string"
}
```

Important response details:

- `prediction`: the predicted disease class or stage label
- `confidence_score`: numeric or string confidence output from the backend
- `grad_cam`: single-image Grad-CAM output for standard workflows
- `grad_cam_mri` and `grad_cam_ct`: Grad-CAM outputs for the MRI + CT dual workflow
- `explanation`: plain-text fallback explanation or structured AI result
- `explanation_source`: indicates whether the explanation came from AI generation or fallback logic

Depending on the backend configuration, the `explanation` field may be a plain string or a structured object containing AI-generated keys such as `summary`, `imaging_findings`, `interpretation`, and `image_references`.

---

## 9. Model details

The project uses saved PyTorch checkpoints stored in the [Saved Model](Saved%20Model) directory.

### Available model artifacts

| Disease category            | Model file                                                                                                                                                     | Architecture family       |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- |
| Alzheimer MRI               | [Saved Model/Alzheimer MRI/ConvNeXt_Tiny_best.pth](Saved%20Model/Alzheimer%20MRI/ConvNeXt_Tiny_best.pth)                                                       | ConvNeXt Tiny legacy      |
| Brain Stroke CT Scan        | [Saved Model/Brain Stroke CT Scan/DenseNet121_best.pth](Saved%20Model/Brain%20Stroke%20CT%20Scan/DenseNet121_best.pth)                                         | DenseNet 121              |
| Brain Tumor CT Scan         | [Saved Model/Brain Tumor/Brain Tumor CT Scan/DenseNet121_best.pth](Saved%20Model/Brain%20Tumor/Brain%20Tumor%20CT%20Scan/DenseNet121_best.pth)                 | DenseNet 121              |
| Brain Tumor MRI             | [Saved Model/Brain Tumor/Brain Tumor MRI/DenseNet121_best.pth](Saved%20Model/Brain%20Tumor/Brain%20Tumor%20MRI/DenseNet121_best.pth)                           | DenseNet 121              |
| Brain Tumor MRI and CT Scan | [Saved Model/Brain Tumor/Brain Tumor MRI + CT Scan/DenseNet121_best.pth](Saved%20Model/Brain%20Tumor/Brain%20Tumor%20MRI%20+%20CT%20Scan/DenseNet121_best.pth) | Shared DenseNet dual-head |
| Parkinsons MRI              | [Saved Model/Parkinson’s Disease MRI/ConvNeXt_Tiny_best.pth](Saved%20Model/Parkinson%E2%80%99s%20Disease%20MRI/ConvNeXt_Tiny_best.pth)                         | ConvNeXt Tiny legacy      |

### How the backend chooses the model

The inference script resolves the correct architecture based on the selected disease:

- `Alzheimer MRI` and `Parkinsons MRI` use a ConvNeXt-based legacy model.
- Single-image stroke and tumor categories use a DenseNet 121 classifier.
- `Brain Tumor MRI and CT Scan` uses a custom dual-head DenseNet architecture that accepts both MRI and CT images.

### Preprocessing used for inference

Each uploaded image follows a strict backend preprocessing pipeline:

- the backend receives the image file through Django’s request handler
- the file is opened as a PIL image and converted to RGB
- the raw scan image is cleaned to reduce visible artifacts and text overlays using OpenCV-based artifact detection and inpainting
- the cleaned image is resized to 224x224 pixels
- the cleaned image is converted to a PyTorch tensor
- the tensor is normalized using ImageNet mean and standard deviation

Note: This cleaning step is applied before model inference and before Grad-CAM generation, so the model and heatmap are produced from the cleaned scan rather than the raw upload.

### Grad-CAM generation

The backend creates a Grad-CAM heatmap from the model’s deeper feature maps and overlays it onto the input image so the UI can show the visual explanation.

---

## 10. Explanation generation

The backend can produce explanations in two ways:

1. Rule-based fallback explanation
   - Used when no Gemini API key is available or when the API call fails.
   - Always returns a safe, human-readable explanation.
   - This is the default behavior for local runs when the environment is not configured for Gemini.

2. Gemini-assisted explanation
   - Enabled when `GEMINI_API_KEY` is present.
   - The backend asks Gemini for a structured JSON response with fields such as:
     - `summary`
     - `detailed_visual_description`
     - `imaging_findings`
     - `interpretation`
     - `confidence`
     - `image_references`
   - The frontend renders these explanation sections and image references when they are returned by the backend.

### Grad-CAM and Gemini flow

The inference pipeline follows this sequence:

1. The uploaded image is preprocessed and passed through the selected model.
2. The model produces a prediction and confidence score.
3. A Grad-CAM heatmap is generated to highlight important image regions.
4. The backend either returns a fallback explanation or sends the image context to Gemini for a richer report.
5. The frontend displays the prediction, confidence, Grad-CAM, and explanation in the result panel.

---

## 11. Notes about the frontend

The React UI supports:

- selecting the disease category
- uploading one image or two images for the combined brain tumor workflow
- showing prediction results
- displaying confidence
- rendering Grad-CAM images
- displaying AI explanation sections when available

The frontend relies on the Django backend for all inference and explanation generation.

---

## 12. Common troubleshooting

### Model file missing

If you get an error like `Model artifact missing`, verify that the corresponding `.pth` file exists under [Saved Model](Saved%20Model).

### Backend cannot connect to the frontend

If the frontend cannot talk to Django, ensure:

- the Django server is running
- the frontend is using the Vite proxy correctly
- CORS is enabled in the Django settings
- the API is reachable at `http://127.0.0.1:8000/api/predict/`

### Import errors

If Python throws package import errors, reinstall the backend dependencies in the active virtual environment:

```powershell
pip install django djangorestframework django-cors-headers python-dotenv google-genai opencv-python pillow numpy torch torchvision
```

### Gemini errors

If the explanation generation fails:

- verify that `GEMINI_API_KEY` exists in the `.env` file
- confirm the key is valid
- check the backend console output for the provider error
- remember that the app will still work with the fallback explanation if Gemini access is unavailable

### Frontend build or dev-server issues

If the frontend fails to start:

- run `npm install` inside the `frontend` folder
- confirm Node.js 18+ is installed
- restart the Vite dev server after installing new packages

---

## 13. Important usage note

This project is a demonstration of an AI-assisted medical image analysis workflow. It is useful for development, experimentation, and educational purposes, but it should not be considered a certified clinical tool or a substitute for diagnosis by a qualified medical professional.

---

## 14. Suggested next steps

You can extend this project by:

- adding more disease categories
- improving the model training pipeline
- adding image preprocessing quality checks
- adding user authentication
- deploying the frontend and backend separately
- adding automated tests for the API and inference flow

---

## Environment and runtime details

### Python environment

The workspace uses a local Python virtual environment:

- `.venv`

### Important runtime packages used by the inference stack

- `torch`
- `torchvision`
- `opencv-python`
- `Pillow`
- `numpy`
- `google-genai`
- `python-dotenv`
- `Django`
- `djangorestframework`
- `django-cors-headers`

### Frontend runtime packages

- `react`
- `react-dom`
- `vite`

---

## Local setup instructions

### 1. Activate the project Python environment

Use the workspace virtual environment:

```powershell
.\.venv\Scripts\python.exe
```

### 2. Install missing runtime dependencies if needed

If the environment is missing a package, install it into the active environment:

```powershell
.\.venv\Scripts\python.exe -m pip install torchvision python-dotenv google-genai
```

### 3. Run Django backend

From the project root:

```powershell
cd brainscan_backend
..\.venv\Scripts\python.exe manage.py runserver 127.0.0.1:8000
```

### 4. Run React frontend

From the project root:

```powershell
cd frontend
npm install
npm run dev
```

### 5. Use the application

Open the Vite frontend in the browser and upload a scan image for one of the supported diseases.

---

## Gemini configuration note

The backend was wired to pick up a repository-level `.env` file containing the Gemini API key.

Environment variable:

```env
GEMINI_API_KEY=your_api_key_here
```

Current verification status of the external Gemini provider:

- the key exists in the runtime environment
- the code path is correctly wired to the live Gemini client
- the provider returned a `403 PERMISSION_DENIED` response for the current project access

This means the implementation works structurally, but the external provider currently blocks live generation for this project ID.

---

## Known current status

### Working now

- frontend build path is functional
- Django API route is live
- multipart image upload route returns structured JSON
- prediction + confidence output is real
- Grad-CAM data is returned as a base64 image string
- explanation path is wired with fallback-safe behavior

### Still controlled by external provider access

- live Gemini explanation text generation

The backend is already coded to switch over to the live model once project access is allowed.

---

## Summary

BrainScan is now a functioning full-stack MRI/CT classification demo that:

- serves the UI via React + Vite
- receives image uploads in Django
- loads real saved PyTorch weights from the `Saved Model/` directory
- performs model inference
- returns prediction + confidence
- generates a Grad-CAM overlay
- attempts live AI explanation generation through Gemini

This README captures the implementation path from the initial scaffold through the current verified runtime behavior.
