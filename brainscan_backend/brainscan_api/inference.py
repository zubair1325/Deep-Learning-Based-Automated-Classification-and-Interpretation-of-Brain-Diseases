import base64
import io
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_ROOT = BASE_DIR / 'Saved Model'

CLASS_LABELS = {
    'Alzheimer MRI': [
        'MildDemented',
        'ModerateDemented',
        'NonDemented',
        'VeryMildDemented',
    ],
    'Brain Stroke CT Scan': [
        'Class 0',
        'Class 1',
        'Class 2',
    ],
    'Brain Tumor CT Scan': [
        'Class 0',
        'Class 1',
        'Class 2',
        'Class 3',
    ],
    'Brain Tumor MRI': [
        'Class 0',
        'Class 1',
        'Class 2',
    ],
    'Brain Tumor MRI and CT Scan': [
        'Class 0',
        'Class 1',
        'Class 2',
        'Class 3',
    ],
    'Parkinsons MRI': [
        'Class 0',
        'Class 1',
        'Class 2',
    ],
}

MODEL_PATHS = {
    'Alzheimer MRI': MODEL_ROOT / 'Alzheimer MRI' / 'ConvNeXt_Tiny_best.pth',
    'Brain Stroke CT Scan': MODEL_ROOT / 'Brain Stroke CT Scan' / 'DenseNet121_best.pth',
    'Brain Tumor CT Scan': MODEL_ROOT / 'Brain Tumor' / 'Brain Tumor CT Scan' / 'DenseNet121_best.pth',
    'Brain Tumor MRI': MODEL_ROOT / 'Brain Tumor' / 'Brain Tumor MRI' / 'DenseNet121_best.pth',
    'Brain Tumor MRI and CT Scan': MODEL_ROOT / 'Brain Tumor' / 'Brain Tumor MRI + CT Scan' / 'DenseNet121_best.pth',
    'Parkinsons MRI': MODEL_ROOT / 'Parkinson’s Disease MRI' / 'ConvNeXt_Tiny_best.pth',
}


def _resolve_model(disease_label: str):
    if disease_label not in MODEL_PATHS:
        raise ValueError(f'Unsupported disease label: {disease_label}')

    model_path = MODEL_PATHS[disease_label]
    if not model_path.exists():
        raise FileNotFoundError(f'Model artifact missing: {model_path}')

    state_dict = torch.load(model_path, map_location='cpu')
    model_name = model_path.name

    if 'ConvNeXt' in model_name:
        num_classes = 4 if 'Alzheimer' in disease_label else 3
        model = ConvNeXtTinyLegacy(num_classes=num_classes)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.module_name = 'convnext'
        return model, 'convnext'

    if 'DenseNet121' in model_name:
        if disease_label == 'Brain Tumor MRI and CT Scan':
            model = SharedDenseNetDualHead(num_mri_classes=4, num_ct_classes=2)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            model.module_name = 'densenet_shared'
            return model, 'densenet_shared'

        num_classes = _infer_dense_output_from_state_dict(state_dict)
        hidden_dim = _infer_dense_hidden_from_state_dict(state_dict)
        model = models.densenet121(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier.in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.module_name = 'densenet'
        return model, 'densenet'

    raise ValueError(f'Could not resolve a matching architecture for {disease_label}')


def _infer_dense_output_from_state_dict(state_dict):
    for key in state_dict:
        if key.endswith('classifier.3.weight'):
            return int(state_dict[key].shape[0])
    return 2


def _infer_dense_hidden_from_state_dict(state_dict):
    for key in state_dict:
        if key.endswith('classifier.0.weight'):
            return int(state_dict[key].shape[0])
    return 512


def _build_preprocessor():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert('RGB')
    tensor = _build_preprocessor()(image)
    return tensor.unsqueeze(0)


def build_gradcam(model, tensor):
    model.eval()
    tensor.requires_grad_(True)

    feature_map = None
    gradients = None

    def _forward_hook(module, inputs, outputs):
        nonlocal feature_map
        feature_map = outputs.detach()

    def _backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()

    if model.module_name == 'convnext':
        target_layer = model.stages[-1].blocks[-1].conv_dw
    elif model.module_name == 'densenet_shared':
        target_layer = model.backbone.features.denseblock4.denselayer16.conv2
    else:
        target_layer = model.features.denseblock4.denselayer16.conv2

    target_layer.register_forward_hook(_forward_hook)
    target_layer.register_full_backward_hook(_backward_hook)

    logits = model(tensor)
    if model.module_name == 'densenet_shared':
        logits = logits[0]

    predicted = int(torch.argmax(logits, dim=1).item())
    score = logits[:, predicted].sum()
    model.zero_grad(set_to_none=True)
    score.backward()

    if feature_map is None or gradients is None:
        return None

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * feature_map, dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam.squeeze(0).squeeze(0).detach().cpu().numpy()
    cam = cv2.resize(cam, (224, 224))
    cam = cam - cam.min()
    cam = cam / max(cam.max(), 1e-8)
    cam = (cam * 255).astype('uint8')
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    input_image = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    input_image = (input_image * np.array([0.229, 0.224, 0.225], dtype=np.float64)
                   + np.array([0.485, 0.456, 0.406], dtype=np.float64)) * 255
    input_image = np.clip(input_image, 0, 255).astype(np.float32)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    input_image = input_image.astype(np.uint8)
    fused = (0.4 * heatmap + 0.6 * input_image).astype(np.float32)
    fused = np.clip(fused, 0, 255).astype(np.uint8)
    fused = cv2.cvtColor(fused, cv2.COLOR_BGR2RGB)

    buffer = io.BytesIO()
    Image.fromarray(fused).save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def _generate_gemini_explanation(disease_label: str, predicted_label: str, confidence: float):
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return (
            f'The uploaded {disease_label} scan was processed by BrainScan using the winner-model pipeline. '
            f'Prediction: {predicted_label}. Confidence: {confidence:.4f}. '
            'Grad-CAM was generated to highlight the most influential image regions.'
        )

    prompt = (
        f'You are BrainScan, an MRI/CT classification assistant. '
        f'Selected disease: {disease_label}. '
        f'Predicted diagnosis: {predicted_label}. '
        f'Confidence score: {confidence:.4f}. '
        'Provide a concise, safe explanation for the user in clinical-style language.'
    )

    try:
        try:
            from google import generativeai as genai
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            response = gemini_model.generate_content(prompt)
            if getattr(response, 'text', None):
                return response.text
        except Exception:
            from google import genai as google_genai
            client = google_genai.Client(api_key=api_key)
            for model_name in ('gemini-2.5-flash', 'gemini-1.5-flash'):
                try:
                    response = client.models.generate_content(model=model_name, contents=prompt)
                    text = getattr(response, 'text', None)
                    if text:
                        return text
                except Exception:
                    continue
    except Exception:
        pass

    return (
        f'The uploaded {disease_label} scan was processed using the winner-model pipeline. '
        f'Prediction: {predicted_label}. Confidence: {confidence:.4f}. '
        'Grad-CAM highlighting was produced for visual localization.'
    )


def predict_image(disease_label: str, uploaded_files):
    model, architecture = _resolve_model(disease_label)
    file_list = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]

    if disease_label == 'Brain Tumor MRI and CT Scan':
        if len(file_list) < 2:
            raise ValueError('Brain Tumor MRI and CT Scan requires both MRI and CT image uploads.')

        mri_tensor = preprocess_image(file_list[0])
        ct_tensor = preprocess_image(file_list[1])

        with torch.no_grad():
            mri_logits, ct_logits = model(mri_tensor, ct_tensor)
            mri_probabilities = torch.softmax(mri_logits, dim=1)
            top_class = int(torch.argmax(mri_probabilities, dim=1).item())
            confidence = float(mri_probabilities[0, top_class].item())

        grad_cam = build_gradcam(model, mri_tensor)
        labels = CLASS_LABELS.get(disease_label, [f'Class {idx}' for idx in range(mri_probabilities.shape[1])])
        predicted_label = labels[top_class] if top_class < len(labels) else f'Class {top_class}'

        explanation = _generate_gemini_explanation(disease_label, predicted_label, confidence)
        return {
            'prediction': predicted_label,
            'confidence_score': f'{confidence:.4f}',
            'grad_cam': grad_cam,
            'explanation': explanation,
        }

    tensor = preprocess_image(file_list[0])

    with torch.no_grad():
        logits = model(tensor)
        if architecture == 'densenet_shared':
            logits = logits[0]
        probabilities = torch.softmax(logits, dim=1)
        top_class = int(torch.argmax(probabilities, dim=1).item())
        confidence = float(probabilities[0, top_class].item())

    grad_cam = build_gradcam(model, tensor)
    labels = CLASS_LABELS.get(disease_label, [f'Class {idx}' for idx in range(probabilities.shape[1])])
    predicted_label = labels[top_class] if top_class < len(labels) else f'Class {top_class}'

    explanation = _generate_gemini_explanation(disease_label, predicted_label, confidence)
    return {
        'prediction': predicted_label,
        'confidence_score': f'{confidence:.4f}',
        'grad_cam': grad_cam,
        'explanation': explanation,
    }


class ConvNeXtMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.gelu(x)
        x = self.fc2(x)
        return x


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = ConvNeXtMLP(dim)

    def forward(self, x):
        residual = x
        x = self.conv_dw(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)
        return residual + self.gamma.view(1, -1, 1, 1) * x


class ConvNeXtStage(nn.Module):
    def __init__(self, dim, depth, downsample=False, in_dim=None):
        super().__init__()
        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(
                nn.LayerNorm(in_dim, eps=1e-6),
                nn.Conv2d(in_dim, dim, kernel_size=2, stride=2),
            )
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(dim=dim)
            for _ in range(depth)
        ])

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample[0](x.permute(0, 2, 3, 1))
            x = x.permute(0, 3, 1, 2)
            x = self.downsample[1](x)
        for block in self.blocks:
            x = block(x)
        return x


class ConvNeXtTinyLegacy(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=4, stride=4),
            nn.LayerNorm(96, eps=1e-6),
        )
        self.stages = nn.ModuleList([
            ConvNeXtStage(dim=96, depth=3),
            ConvNeXtStage(dim=192, depth=3, downsample=True, in_dim=96),
            ConvNeXtStage(dim=384, depth=9, downsample=True, in_dim=192),
            ConvNeXtStage(dim=768, depth=3, downsample=True, in_dim=384),
        ])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.ModuleDict({
            'norm': nn.LayerNorm(768, eps=1e-6),
            'fc': nn.Linear(768, num_classes),
        })

    def forward(self, x):
        x = self.stem[0](x)
        x = x.permute(0, 2, 3, 1)
        x = self.stem[1](x)
        x = x.permute(0, 3, 1, 2)
        x = self.stages[0](x)
        x = self.stages[1](x)
        x = self.stages[2](x)
        x = self.stages[3](x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head['norm'](x)
        x = self.head['fc'](x)
        return x


class SharedDenseNetDualHead(nn.Module):
    def __init__(self, num_mri_classes=4, num_ct_classes=2):
        super().__init__()
        base_model = models.densenet121(weights=None)
        self.backbone = nn.Module()
        self.backbone.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mri_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_mri_classes),
        )
        self.ct_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_ct_classes),
        )

    def forward(self, mri_x, ct_x=None):
        if ct_x is None:
            x = self.backbone.features(mri_x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.mri_head(x), self.ct_head(x)

        mri_features = self.backbone.features(mri_x)
        mri_features = self.avgpool(mri_features)
        mri_features = torch.flatten(mri_features, 1)

        ct_features = self.backbone.features(ct_x)
        ct_features = self.avgpool(ct_features)
        ct_features = torch.flatten(ct_features, 1)

        return self.mri_head(mri_features), self.ct_head(ct_features)
