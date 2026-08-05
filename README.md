# BrainScan

BrainScan is a full-stack medical image classification and interpretation project for brain diseases using MRI and CT scans. The project combines a React + Vite frontend with a Django backend, and it is designed to receive a user-uploaded scan image, run disease-specific model inference, compute a confidence score, generate a Grad-CAM visualization, and return an AI-generated explanation.

The system currently supports the following disease categories:

- Alzheimer MRI
- Brain Stroke CT Scan
- Brain Tumor CT Scan
- Brain Tumor MRI
- Brain Tumor MRI and CT Scan
- Parkinsons MRI

---

## Project purpose

The application was built to demonstrate an end-to-end clinical-style prediction pipeline:

1. User selects a disease category in the frontend.
2. User uploads an MRI or CT image.
3. React sends the request to Django.
4. Django receives the multipart request.
5. The backend preprocesses the image.
6. A saved PyTorch `.pth` model is loaded.
7. The model predicts the label and confidence.
8. A Grad-CAM heatmap is generated.
9. An explanation is returned to the frontend.

---

## High-level architecture

### Frontend

- Framework: React + Vite
- Main UI file: `frontend/src/App.jsx`
- Styling: `frontend/src/App.css`
- Build and dev server setup: `frontend/package.json`

### Backend

- Framework: Django
- API app: `brainscan_backend/brainscan_api`
- Project settings: `brainscan_backend/brainscan_backend/settings.py`
- API route: `brainscan_backend/brainscan_api/urls.py`
- Request handler: `brainscan_backend/brainscan_api/views.py`
- Inference engine: `brainscan_backend/brainscan_api/inference.py`

### Model storage

The saved model artifacts are kept under the `Saved Model/` folder.

Included winner-model artifacts:

- `Saved Model/Alzheimer MRI/ConvNeXt_Tiny_best.pth`
- `Saved Model/Brain Stroke CT Scan/DenseNet121_best.pth`
- `Saved Model/Brain Tumor/Brain Tumor CT Scan/DenseNet121_best.pth`
- `Saved Model/Brain Tumor/Brain Tumor MRI/DenseNet121_best.pth`
- `Saved Model/Brain Tumor/Brain Tumor MRI + CT Scan/DenseNet121_best.pth`
- `Saved Model/Parkinson’s Disease MRI/ConvNeXt_Tiny_best.pth`

---

## What was implemented from the start to the current state

### 1. Frontend UI

The frontend was replaced with a BrainScan medical dashboard that includes:

- disease category dropdown
- image file input
- submit button
- processing/loading state
- result card showing:
  - predicted label
  - confidence score
  - AI explanation
  - Grad-CAM image

The frontend logic is centralized in `frontend/src/App.jsx`.

### 2. API proxy setup for local development

Vite is configured to proxy `/api` requests to Django:

- file: `frontend/vite.config.js`

This allows the React app to keep the API call path clean while the Django backend serves the actual inference route.

### 3. Django project runtime configuration

The Django project was updated with:

- `rest_framework`
- `corsheaders`
- `brainscan_api` app registration
- `ALLOWED_HOSTS` configured for local testing
- CORS enabled for frontend-to-backend communication

Relevant settings file:

- `brainscan_backend/brainscan_backend/settings.py`

### 4. Django API route

The API endpoint is exposed as:

- `POST /api/predict/`

The route is declared in:

- `brainscan_backend/brainscan_api/urls.py`

### 5. Upload handler

The view receives:

- the selected disease label from `request.POST['disease']`
- the uploaded image file from `request.FILES['image']`

Then it delegates the work to the real inference function.

Relevant file:

- `brainscan_backend/brainscan_api/views.py`

### 6. Real inference pipeline

The backend inference engine in `brainscan_backend/brainscan_api/inference.py` now performs the following steps:

- resolves the appropriate model path from the disease label
- loads the saved `.pth` state dictionary on CPU
- reconstructs the correct model architecture family
- preprocesses the uploaded image into the expected tensor format
- runs the forward pass
- extracts predicted class and confidence score
- builds a Grad-CAM heatmap overlay
- returns a JSON payload to the frontend

### 7. Exact model-family distinction

The model loader resolves the correct architecture choice based on the disease label, and the file structure precisely reflects that split.

Implemented architecture behavior:

- ConvNeXt Tiny legacy family for the Alzheimer MRI and Parkinsons MRI checkpoints
- DenseNet 121 family for the single-image stroke and tumor checkpoints
- a shared-backbone dual-head DenseNet 121 family for the combined Brain Tumor MRI and CT Scan workflow

#### Single-image disease families

- `Alzheimer MRI` → ConvNeXt Tiny legacy checkpoint
- `Parkinsons MRI` → ConvNeXt Tiny legacy checkpoint
- `Brain Stroke CT Scan` → DenseNet 121 checkpoint
- `Brain Tumor CT Scan` → DenseNet 121 checkpoint
- `Brain Tumor MRI` → DenseNet 121 checkpoint

#### Combined MRI + CT family

- `Brain Tumor MRI and CT Scan` → `SharedDenseNetDualHead`
  - one upload for the MRI image
  - one upload for the CT image
  - the model runs through a common DenseNet backbone and then uses separate MRI and CT prediction heads
  - the current backend produces the final prediction from the MRI head path for the combined-class workflow

### 8. Grad-CAM generation

The backend builds a Grad-CAM activation map using the chosen model’s final high-level convolutional layer, then blends it with the input image to produce a visualization.

### 9. Explanation generation path

The project is designed to return a natural-language explanation for the prediction.

Behavior:

- If no Gemini key exists, the backend returns a safe fallback explanation.
- If the key exists, the backend attempts Gemini-based explanation generation.
- If the provider rejects the key or project access, the code falls back to the safe static explanation text.

### 10. Final runtime verification

The project was verified through the real Django test client with a multipart upload request against the actual `/api/predict/` route.

Verified result from the last runtime proof:

- status = `200`
- prediction = `VeryMildDemented`
- confidence score = `0.9243`
- explanation field returned in JSON payload

This confirms that the endpoint is now returning a real structured prediction response from the backend.

---

## Current request/response contract

### Frontend request

The frontend sends a multipart form request with:

- `disease`
- `image` for the standard single-input disease categories
- `mri_image` and `ct_image` for `Brain Tumor MRI and CT Scan`

For the combined tumor choice, the request contract is intentionally different because that model is a multi-head shared-backbone system rather than a standard single-input DenseNet 121 classifier.

### Backend response shape

The backend returns a JSON payload like this:

```json
{
  "prediction": "VeryMildDemented",
  "confidence_score": "0.9243",
  "grad_cam": "base64-encoded-png-string",
  "explanation": "Human-readable explanation or fallback text"
}
```

---

## Project files notable at a glance

### Root project

- `manage.py`
- `.env`
- `README.md`

### Backend root

- `brainscan_backend/manage.py`
- `brainscan_backend/brainscan_backend/settings.py`
- `brainscan_backend/brainscan_backend/urls.py`
- `brainscan_backend/brainscan_api/views.py`
- `brainscan_backend/brainscan_api/inference.py`
- `brainscan_backend/brainscan_api/urls.py`

### Frontend root

- `frontend/package.json`
- `frontend/vite.config.js`
- `frontend/src/App.jsx`
- `frontend/src/App.css`

### Saved models

- `Saved Model/`

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
