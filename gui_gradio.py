import gradio as gr
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import joblib
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import tempfile
import os

# Load models
models_dict = joblib.load("compression_model_quantile.pkl")
pca_model = joblib.load("pca_model.pkl")

# Load ResNet
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
feature_extractor.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def compress_image_gui(input_img, uncertainty_threshold=35):
    """Process image and return compressed version with metrics"""
    
    # Convert PIL to OpenCV
    img_cv = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)
    
    # Save temporarily
    temp_path = "temp_input.png"
    cv2.imwrite(temp_path, img_cv)
    
    # Extract features
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    img_t = transform(img_pil).unsqueeze(0)
    
    with torch.no_grad():
        features = feature_extractor(img_t)
    features = features.squeeze().numpy()
    
    # Apply PCA
    features_reduced = pca_model.transform([features])
    
    # Predict
    pred_median = models_dict[0.5].predict(features_reduced)[0]
    pred_lower = models_dict[0.1].predict(features_reduced)[0]
    pred_upper = models_dict[0.9].predict(features_reduced)[0]
    uncertainty = pred_upper - pred_lower
    
    # Determine quality
    if uncertainty > uncertainty_threshold:
        final_quality = (pred_median * 0.7) + (pred_lower * 0.3)
    else:
        final_quality = pred_median
    
    final_quality = int(np.clip(final_quality, 50, 95))
    
    # Compress
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), final_quality]
    result, encimg = cv2.imencode('.jpg', img_cv, encode_param)
    compressed_img = cv2.imdecode(encimg, 1)
    
    # Calculate metrics
    original_size = os.path.getsize(temp_path) / 1024
    compressed_size = len(encimg) / 1024
    compression_ratio = original_size / compressed_size
    size_reduction = (1 - compressed_size/original_size) * 100
    
    ssim_score, _ = ssim(img_cv, compressed_img, channel_axis=2, full=True)
    
    # Clean up
    os.remove(temp_path)
    
    # Convert back to RGB for display
    compressed_img_rgb = cv2.cvtColor(compressed_img, cv2.COLOR_BGR2RGB)
    
    # Create info text
    info = f"""
    **Compression Results:**
    - Predicted Quality: {pred_median:.1f} ± {uncertainty/2:.1f}
    - Used Quality: {final_quality}
    - Original Size: {original_size:.2f} KB
    - Compressed Size: {compressed_size:.2f} KB
    - Compression Ratio: {compression_ratio:.2f}x
    - Size Reduction: {size_reduction:.1f}%
    - SSIM Score: {ssim_score:.4f}
    """
    
    return compressed_img_rgb, info

# Create Gradio interface
demo = gr.Interface(
    fn=compress_image_gui,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(20, 50, value=35, label="Uncertainty Threshold")
    ],
    outputs=[
        gr.Image(label="Compressed Image"),
        gr.Textbox(label="Metrics")
    ],
    title="🎨 AI-Powered Image Compression",
    description="**Graduate-Level ML Project** | Uses ResNet18 features + Quantile Regression for optimal JPEG compression",
    examples=[
        ["/Users/admin/Documents/ml_images/data/div2k/DIV2K_train_HR/DIV2K_train_HR/0001.png", 35]
    ]
)

demo.launch()