from django.core.files.uploadedfile import UploadedFile
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .inference import predict_image


@csrf_exempt
def predict_view(request):
    if request.method != 'POST':
        return JsonResponse({'detail': 'Only POST requests are supported.'}, status=405)

    disease = request.POST.get('disease', 'Alzheimer MRI')

    if disease == 'Brain Tumor MRI and CT Scan':
        mri_image = request.FILES.get('mri_image')
        ct_image = request.FILES.get('ct_image')

        if not isinstance(mri_image, UploadedFile) or not isinstance(ct_image, UploadedFile):
            return JsonResponse({'detail': 'Please upload both MRI and CT images for Brain Tumor MRI and CT Scan.'}, status=400)

        try:
            result = predict_image(disease, [mri_image, ct_image])
        except Exception as error:
            return JsonResponse({'detail': str(error)}, status=500)

        return JsonResponse(result)

    uploaded_image = request.FILES.get('image')
    if not isinstance(uploaded_image, UploadedFile):
        return JsonResponse({'detail': 'No image file was uploaded.'}, status=400)

    try:
        result = predict_image(disease, uploaded_image)
    except Exception as error:
        return JsonResponse({'detail': str(error)}, status=500)

    return JsonResponse(result)
