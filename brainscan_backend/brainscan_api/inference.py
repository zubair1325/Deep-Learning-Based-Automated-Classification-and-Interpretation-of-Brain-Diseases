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
from google.genai import types

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_ROOT = BASE_DIR / 'Saved Model'

STAGE_LABELS = {
    'Alzheimer MRI': [
        'Mild dementia features detected. This may indicate early-stage Alzheimer’s disease.',
        'Moderate dementia features detected. Please consider specialist follow-up.',
        'No dementia detected on this scan.',
        'Very mild dementia features detected. This may represent very early cognitive change.',
    ],
    'Brain Stroke CT Scan': [
        'No stroke signs detected.',
        'Ischemic stroke signs detected. This may indicate reduced blood flow in part of the brain.',
        'Bleeding stroke signs detected. This may indicate hemorrhage in the brain.',
    ],
    'Brain Tumor CT Scan': [
        'No tumor signs detected.',
        'Early-stage tumor features detected.',
        'Moderate-stage tumor features detected.',
        'Advanced-stage tumor features detected.',
    ],
    'Brain Tumor MRI': [
        'No tumor signs detected.',
        'Early-stage tumor features detected.',
        'Moderate-stage tumor features detected.',
    ],
    'Brain Tumor MRI and CT Scan': [
        'No tumor signs detected on MRI.',
        'Early-stage tumor features detected on MRI.',
        'Moderate-stage tumor features detected on MRI.',
        'Advanced-stage tumor features detected on MRI.',
    ],
    'Parkinsons MRI': [
        'No Parkinson’s disease features detected.',
        'Possible early-stage Parkinson’s disease features detected.',
        'Parkinson’s disease features likely detected.',
    ],
}

COMBINED_CT_STAGE_LABELS = {
    'Brain Tumor MRI and CT Scan': [
        'No tumor signs detected on CT.',
        'Tumor-related CT changes detected.',
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


def _pil_image_to_bytes(image: Image.Image, format: str = 'PNG') -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def build_gradcam_pil(model, tensor):
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

    return Image.fromarray(fused)


def build_gradcam(model, tensor):
    pil_image = build_gradcam_pil(model, tensor)
    if pil_image is None:
        return None

    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def _resolve_stage_label(disease_label: str, class_index: int, is_ct: bool = False):
    if is_ct and disease_label in COMBINED_CT_STAGE_LABELS:
        labels = COMBINED_CT_STAGE_LABELS[disease_label]
    else:
        labels = STAGE_LABELS.get(disease_label)

    if labels is None:
        return f'Class {class_index}'

    return labels[class_index] if class_index < len(labels) else f'Class {class_index}'


def _load_pil_image(uploaded_file):
    uploaded_file.seek(0)
    return Image.open(uploaded_file).convert('RGB')


def preprocess_image(uploaded_file):
    image = _load_pil_image(uploaded_file)
    tensor = _build_preprocessor()(image)
    return tensor.unsqueeze(0)


def _build_default_explanation(
    disease_label: str,
    predicted_label: str,
    confidence: float,
    extra_context: str = '',
):
    clean_label = predicted_label.rstrip('. ').strip()
    explanation_parts = [
        f'The uploaded {disease_label} scan was analyzed by BrainScan.',
        f'Result: {clean_label}.',
    ]

    if 'Alzheimer' in disease_label:
        if 'No dementia' in predicted_label:
            explanation_parts.append(
                'This means the image does not show the typical signs of Alzheimer’s-related brain changes, and the scan appears largely normal.'
            )
        elif 'Very mild dementia' in predicted_label:
            explanation_parts.append(
                'This suggests very subtle dementia-related changes that may correspond with early cognitive decline. '
                'These changes are mild and may not yet cause obvious symptoms.'
            )
        elif 'Mild dementia' in predicted_label:
            explanation_parts.append(
                'This indicates early-stage dementia features that merit follow-up with a healthcare provider.'
            )
        elif 'Moderate dementia' in predicted_label:
            explanation_parts.append(
                'This indicates more pronounced dementia-related changes and should be reviewed by a specialist.'
            )
    elif 'Stroke' in disease_label:
        if 'No stroke' in predicted_label:
            explanation_parts.append(
                'The scan does not show clear evidence of stroke, which suggests the brain tissue appears stable in this image.'
            )
        elif 'Ischemic' in predicted_label:
            explanation_parts.append(
                'This suggests reduced blood flow to a part of the brain, which is commonly seen in ischemic stroke. '
                'Immediate medical review is advised when these changes are present.'
            )
        elif 'Bleeding' in predicted_label:
            explanation_parts.append(
                'This suggests there may be bleeding within the brain, which is a serious finding that should be evaluated promptly by a doctor.'
            )
    elif 'Tumor' in disease_label:
        if 'No tumor' in predicted_label:
            explanation_parts.append(
                'The scan does not show obvious tumor-related findings, suggesting there are no clear suspicious masses in this image.'
            )
        elif 'Early-stage tumor' in predicted_label:
            explanation_parts.append(
                'This indicates a small or early-stage tumor-related area that should be monitored and confirmed by a specialist.'
            )
        elif 'Moderate-stage tumor' in predicted_label:
            explanation_parts.append(
                'This indicates a more developed tumor-related change that requires specialist evaluation.'
            )
        elif 'Advanced-stage tumor' in predicted_label:
            explanation_parts.append(
                'This suggests a larger or more advanced tumor-related finding that should be reviewed by a medical professional as soon as possible.'
            )
    elif 'Parkinson' in disease_label:
        if 'No Parkinson' in predicted_label:
            explanation_parts.append(
                'The image does not show clear signs of Parkinson’s disease on this scan.'
            )
        elif 'Possible early-stage Parkinson' in predicted_label:
            explanation_parts.append(
                'This suggests early Parkinson’s-related changes, which are often subtle and may require clinical correlation.'
            )
        elif 'Parkinson’s disease features likely' in predicted_label:
            explanation_parts.append(
                'This indicates scan features that are more consistent with Parkinson’s disease and should be reviewed by a neurologist.'
            )

    explanation_parts.append(f'Confidence score: {confidence:.4f}.')
    explanation_parts.append(
        'This result is informational only and may not always be correct. '
        'Please consult a qualified healthcare professional for full interpretation.'
    )

    if extra_context:
        explanation_parts.append(extra_context)

    return ' '.join(explanation_parts)


def _generate_gemini_explanation(
    disease_label: str,
    predicted_label: str,
    confidence: float,
    original_image: Image.Image | None = None,
    grad_image: Image.Image | None = None,
    extra_context: str = '',
):
    api_key = os.getenv('GEMINI_API_KEY')
    role_description = (
        'You are BrainScan, a medical imaging assistant. Provide a concise radiology-style report in strict JSON only. '
        'Include: summary, detailed_visual_description, imaging_findings, measurements (if any), differential_diagnosis, interpretation, suggested_urgency, recommended_next_steps, treatment_options (optional), confidence, limitations, and image_references. '
        'State this is informational only.'
    )

    # Strong instruction for structured output: request strict JSON only.
    json_output_instructions = (
        'IMPORTANT: Respond only with a single valid JSON object and nothing else. '
        'Do not include any surrounding explanation, markdown, or bullet formatting outside the JSON object. The JSON must include the keys below with these exact names and types:\n'
        '  - summary (string)\n'
        '  - detailed_visual_description (string)\n'
        '  - imaging_findings (string)\n'
        '  - measurements (string|null)\n'
        '  - differential_diagnosis (string)\n'
        '  - interpretation (string)\n'
        '  - suggested_urgency (string)\n'
        '  - recommended_next_steps (string)\n'
        '  - treatment_options (string|null)\n'
        '  - confidence (number)\n'
        '  - limitations (string)\n'
        '  - image_references (object)\n'
        'Optional keys allowed: references (string), notes_for_provider (string).'
    )

    # Provide a compact example JSON line to reduce token usage.
    example_json = (
        '{"summary":"Brief summary.","detailed_visual_description":"Short visual description.","imaging_findings":"- finding1\\n- finding2","measurements":null,"differential_diagnosis":"1) A 2) B","interpretation":"Short interpretation.","suggested_urgency":"routine","recommended_next_steps":"Follow-up","treatment_options":null,"confidence":0.80,"limitations":"Informational only.","image_references":{"original":"full_scan","gradcam":"heatmap_overlay"}}'
    )

    contents = [
        types.Content(parts=[
            types.Part.from_text(
                text=role_description,
            ),
            types.Part.from_text(
                text=json_output_instructions,
            ),
            types.Part.from_text(
                text='Example JSON (use EXACT keys and types):',
            ),
            types.Part.from_text(
                text=example_json,
            ),
            types.Part.from_text(
                text=f'Scan type: {disease_label}.',
            ),
            types.Part.from_text(
                text=f'Predicted result: {predicted_label}.',
            ),
            types.Part.from_text(
                text=f'Confidence score: {confidence:.4f}.',
            ),
        ])
    ]

    if extra_context:
        contents[0].parts.append(
            types.Part.from_text(
                text=f'Additional context: {extra_context}',
            )
        )

    # Always print the prepared Gemini request contents so the user can inspect them,
    # even if GEMINI_API_KEY is not set and we will skip the external call.
    try:
        print('\n=== PREPARED GEMINI REQUEST (pre-check) ===')
        for ci, content in enumerate(contents):
            for pi, part in enumerate(getattr(content, 'parts', [])):
                ptext = getattr(part, 'text', None)
                if ptext is not None:
                    preview = ptext if len(ptext) < 1000 else ptext[:1000] + '...'
                    print(f'content[{ci}].part[{pi}].text:', preview)
                elif getattr(part, 'inline_data', None) is not None:
                    bd = getattr(part.inline_data, 'data', b'')
                    mtype = getattr(part.inline_data, 'mime_type', None)
                    print(f'content[{ci}].part[{pi}].inline_data: mime={mtype} size={len(bd)} bytes')
                elif getattr(part, 'file_data', None) is not None:
                    fd = getattr(part.file_data, 'file_uri', None)
                    print(f'content[{ci}].part[{pi}].file_data: uri={fd}')
                else:
                    print(f'content[{ci}].part[{pi}] unknown part repr:', repr(part)[:400])
        print('=== END PREPARED REQUEST ===\n')
    except Exception as e:
        print('Error printing prepared Gemini request:', repr(e))

    if original_image is not None:
        # Downsample original image to limit token cost of multimodal input
        try:
            img_copy = original_image.copy()
            img_copy.thumbnail((512, 512))
            contents[0].parts.append(
                types.Part.from_text(
                    text='Original uploaded image (resized 512x512): see attached image below.',
                )
            )
            contents[0].parts.append(
                types.Part.from_bytes(
                    data=_pil_image_to_bytes(img_copy, format='JPEG'),
                    mime_type='image/jpeg',
                )
            )
        except Exception:
            # fallback to original if resizing fails
            contents[0].parts.append(
                types.Part.from_text(text='Original uploaded image: see attached image below.')
            )
            contents[0].parts.append(
                types.Part.from_bytes(data=_pil_image_to_bytes(original_image, format='JPEG'), mime_type='image/jpeg')
            )

    if grad_image is not None:
        try:
            g_copy = grad_image.copy()
            g_copy.thumbnail((512, 512))
            contents[0].parts.append(
                types.Part.from_text(text='Grad-CAM overlay image (resized 512x512): see attached image below.')
            )
            contents[0].parts.append(types.Part.from_bytes(data=_pil_image_to_bytes(g_copy, format='PNG'), mime_type='image/png'))
        except Exception:
            contents[0].parts.append(types.Part.from_text(text='Grad-CAM overlay image: see attached image below.'))
            contents[0].parts.append(types.Part.from_bytes(data=_pil_image_to_bytes(grad_image, format='PNG'), mime_type='image/png'))

    if not api_key:
        return _build_default_explanation(
            disease_label,
            predicted_label,
            confidence,
            extra_context=extra_context,
        ), 'fallback'

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        # Use a config to request deterministic, longer output (temperature=0, increased token limit)
        config = types.GenerateContentConfig(temperature=0, maxOutputTokens=1200)
        response_schema = {
            'type': 'object',
            'properties': {
                'summary': {'type': 'string'},
                'detailed_visual_description': {'type': 'string'},
                'imaging_findings': {'type': 'string'},
                'measurements': {'type': ['string', 'null']},
                'differential_diagnosis': {'type': 'string'},
                'interpretation': {'type': 'string'},
                'suggested_urgency': {'type': 'string'},
                'recommended_next_steps': {'type': 'string'},
                'treatment_options': {'type': ['string', 'null']},
                'confidence': {'type': 'number'},
                'limitations': {'type': 'string'},
                'image_references': {'type': 'object'},
            },
            'required': [
                'summary',
                'detailed_visual_description',
                'imaging_findings',
                'measurements',
                'differential_diagnosis',
                'interpretation',
                'suggested_urgency',
                'recommended_next_steps',
                'treatment_options',
                'confidence',
                'limitations',
                'image_references',
            ],
            'additionalProperties': False,
        }
        # Prefer the flash-lite variant first (more relaxed rate limits), then fall back to flash models.
        for model_name in (
            'models/gemini-3.5-flash-lite',
            'models/gemini-3.5-flash',
            'models/gemini-3.6-flash',
            'models/gemini-3.1-flash',
            'models/gemini-2.5-flash',
            'models/gemini-2.0-flash',
        ):
            try:
                # Debug: print the outgoing Gemini request contents (text parts and image sizes)
                try:
                    print('\n=== GEMINI REQUEST START ===')
                    for ci, content in enumerate(contents):
                        for pi, part in enumerate(getattr(content, 'parts', [])):
                            ptext = getattr(part, 'text', None)
                            if ptext is not None:
                                preview = ptext if len(ptext) < 1000 else ptext[:1000] + '...'
                                print(f'content[{ci}].part[{pi}].text:', preview)
                            elif getattr(part, 'inline_data', None) is not None:
                                bd = getattr(part.inline_data, 'data', b'')
                                mtype = getattr(part.inline_data, 'mime_type', None)
                                print(f'content[{ci}].part[{pi}].inline_data: mime={mtype} size={len(bd)} bytes')
                            elif getattr(part, 'file_data', None) is not None:
                                fd = getattr(part.file_data, 'file_uri', None)
                                print(f'content[{ci}].part[{pi}].file_data: uri={fd}')
                            else:
                                print(f'content[{ci}].part[{pi}] unknown part repr:', repr(part)[:400])
                    print('=== GEMINI REQUEST END ===\n')
                except Exception as e:
                    print('Error printing Gemini request contents:', repr(e))

                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        maxOutputTokens=1200,
                        responseMimeType='application/json',
                        responseJsonSchema=response_schema,
                    ),
                )
                # Debug: print the raw response object
                try:
                    print('\n=== GEMINI RESPONSE START ===')
                    print('response repr:', repr(response))
                    if getattr(response, 'candidates', None):
                        for ci, cand in enumerate(response.candidates):
                            content = getattr(cand, 'content', None)
                            print(f'candidate[{ci}].content repr:', repr(content)[:1000])
                            ctext = getattr(content, 'text', None)
                            if ctext:
                                preview = ctext if len(ctext) < 2000 else ctext[:2000] + '...'
                                print(f'candidate[{ci}].content.text:', preview)
                            parts = getattr(content, 'parts', None)
                            if parts:
                                for pi, part in enumerate(parts):
                                    ptext = getattr(part, 'text', None)
                                    if ptext is not None:
                                        print(f'candidate[{ci}].part[{pi}].text:', ptext[:1000])
                                    elif getattr(part, 'inline_data', None) is not None:
                                        bd = getattr(part.inline_data, 'data', b'')
                                        mtype = getattr(part.inline_data, 'mime_type', None)
                                        print(f'candidate[{ci}].part[{pi}].inline_data mime={mtype} size={len(bd)} bytes')
                    print('=== GEMINI RESPONSE END ===\n')
                except Exception as e:
                    print('Error printing Gemini response:', repr(e))
                if response and response.candidates:
                    text = getattr(response.candidates[0].content, 'text', None)
                    if not text and getattr(response.candidates[0].content, 'parts', None):
                        text_parts = [
                            getattr(part, 'text', None) for part in response.candidates[0].content.parts
                            if getattr(part, 'text', None) is not None
                        ]
                        text = ' '.join(text_parts).strip()

                    if text:
                            # Prefer structured JSON per instructions. Try to parse and return JSON.
                            import json, re
                            try:
                                parsed = json.loads(text)
                                return (parsed, 'AI')
                            except Exception:
                                # attempt to recover malformed JSON and parse again
                                first_brace = text.find('{')
                                if first_brace != -1:
                                    candidate = text[first_brace:]
                                    candidate = re.sub(r",\s*([\]}])", r"\1", candidate)
                                    open_braces = candidate.count('{')
                                    close_braces = candidate.count('}')
                                    if close_braces < open_braces:
                                        candidate += '}' * (open_braces - close_braces)
                                    try:
                                        parsed = json.loads(candidate)
                                        return (parsed, 'AI')
                                    except Exception:
                                        pass

                            # If the model stopped because of token limits or parsing failed,
                            # attempt a short auto-retry to continue the JSON object.
                            try:
                                finish_reason = getattr(response.candidates[0], 'finish_reason', None)
                            except Exception:
                                finish_reason = None

                            # Retry up to 2 times to continue the JSON
                            for attempt in range(2):
                                try:
                                    continuation_prompt = (
                                        'The previous response was truncated. '
                                        'Here is the partial JSON response below. '
                                        'Continue the JSON object so it becomes a single valid JSON object. '
                                        'Return ONLY the remaining JSON text needed to complete the object, with no explanation.'
                                    )
                                    cont_contents = [
                                        types.Content(parts=[
                                            types.Part.from_text(text=continuation_prompt),
                                            types.Part.from_text(text=text),
                                        ])
                                    ]

                                    cont_response = client.models.generate_content(
                                        model=model_name,
                                        contents=cont_contents,
                                        config=types.GenerateContentConfig(
                                            temperature=0,
                                            maxOutputTokens=800,
                                            responseMimeType='text/plain',
                                        ),
                                    )

                                    cont_text = None
                                    if cont_response and getattr(cont_response, 'candidates', None):
                                        cont_text = getattr(cont_response.candidates[0].content, 'text', None)
                                        if not cont_text and getattr(cont_response.candidates[0].content, 'parts', None):
                                            cont_text = ' '.join([
                                                getattr(p, 'text', '') for p in cont_response.candidates[0].content.parts
                                                if getattr(p, 'text', None)
                                            ]).strip()

                                    if cont_text:
                                        # Append continuation and attempt to parse full JSON
                                        combined = text + cont_text
                                        try:
                                            parsed = json.loads(combined)
                                            return (parsed, 'AI')
                                        except Exception:
                                            # try the same repair approach on combined
                                            m = re.search(r"\{[\s\S]*\}", combined)
                                            if m:
                                                candidate = m.group(0)
                                                candidate = re.sub(r",\s*([\]}])", r"\1", candidate)
                                                open_braces = candidate.count('{')
                                                close_braces = candidate.count('}')
                                                if close_braces < open_braces:
                                                    candidate += '}' * (open_braces - close_braces)
                                                try:
                                                    parsed = json.loads(candidate)
                                                    return (parsed, 'AI')
                                                except Exception:
                                                    pass
                                    # if continuation didn't help, try next attempt
                                except Exception:
                                    continue

                            # If parsing still fails, fall back to raw text but label as unstructured AI
                            return (text, 'AI-unstructured')
            except Exception:
                continue
    except Exception:
        pass

    return _build_default_explanation(
        disease_label,
        predicted_label,
        confidence,
        extra_context=extra_context,
    ), 'fallback'


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
            ct_probabilities = torch.softmax(ct_logits, dim=1)
            mri_class = int(torch.argmax(mri_probabilities, dim=1).item())
            ct_class = int(torch.argmax(ct_probabilities, dim=1).item())
            mri_confidence = float(mri_probabilities[0, mri_class].item())
            ct_confidence = float(ct_probabilities[0, ct_class].item())

        mri_image = _load_pil_image(file_list[0])
        ct_image = _load_pil_image(file_list[1])
        mri_tensor = preprocess_image(file_list[0])
        ct_tensor = preprocess_image(file_list[1])

        with torch.no_grad():
            mri_logits, ct_logits = model(mri_tensor, ct_tensor)
            mri_probabilities = torch.softmax(mri_logits, dim=1)
            ct_probabilities = torch.softmax(ct_logits, dim=1)
            mri_class = int(torch.argmax(mri_probabilities, dim=1).item())
            ct_class = int(torch.argmax(ct_probabilities, dim=1).item())
            mri_confidence = float(mri_probabilities[0, mri_class].item())
            ct_confidence = float(ct_probabilities[0, ct_class].item())

        mri_grad_cam = build_gradcam(model, mri_tensor)
        ct_grad_cam = build_gradcam(model, ct_tensor)
        mri_prediction = _resolve_stage_label(disease_label, mri_class)
        ct_prediction = _resolve_stage_label(disease_label, ct_class, is_ct=True)
        predicted_label = f'MRI: {mri_prediction}; CT: {ct_prediction}'
        confidence = mri_confidence
        explanation, explanation_source = _generate_gemini_explanation(
            disease_label,
            predicted_label,
            confidence,
            original_image=mri_image,
            grad_image=build_gradcam_pil(model, mri_tensor),
            extra_context=(
                f'MRI confidence: {mri_confidence:.4f}. '
                f'CT confidence: {ct_confidence:.4f}. '
                f'MRI result: {mri_prediction}. CT result: {ct_prediction}.'
            ),
        )
        return {
            'prediction': predicted_label,
            'confidence_score': f'MRI {mri_confidence:.4f}, CT {ct_confidence:.4f}',
            'grad_cam_mri': mri_grad_cam,
            'grad_cam_ct': ct_grad_cam,
            'grad_cam': mri_grad_cam,
            'explanation': explanation,
            'explanation_source': explanation_source,
        }

    image = _load_pil_image(file_list[0])
    tensor = preprocess_image(file_list[0])

    with torch.no_grad():
        logits = model(tensor)
        if architecture == 'densenet_shared':
            logits = logits[0]
        probabilities = torch.softmax(logits, dim=1)
        top_class = int(torch.argmax(probabilities, dim=1).item())
        confidence = float(probabilities[0, top_class].item())

    grad_cam = build_gradcam(model, tensor)
    predicted_label = _resolve_stage_label(disease_label, top_class)
    explanation, explanation_source = _generate_gemini_explanation(
        disease_label,
        predicted_label,
        confidence,
        original_image=image,
        grad_image=build_gradcam_pil(model, tensor),
    )
    return {
        'prediction': predicted_label,
        'confidence_score': f'{confidence:.4f}',
        'grad_cam': grad_cam,
        'explanation': explanation,
        'explanation_source': explanation_source,
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
