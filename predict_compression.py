import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import joblib
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim
import os
import sys
import subprocess
import platform

# ==================== Load Models ====================

print("Loading trained models...")
models_dict = joblib.load("compression_model_quantile.pkl")
pca_model = joblib.load("pca_model.pkl")
print("Models loaded successfully!")

# Load ResNet for feature extraction
print("Loading ResNet18...")
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
feature_extractor.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_deep_features(image_path):
    """Extract 512-dimensional features from ResNet"""
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        features = feature_extractor(img_t)
    
    return features.squeeze().numpy()


def compress_image(input_path, output_path, quality):
    """Compress image with specified JPEG quality"""
    img = cv2.imread(input_path)
    
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    
    cv2.imwrite(output_path, decimg)
    
    original_size = os.path.getsize(input_path) / 1024
    compressed_size = os.path.getsize(output_path) / 1024
    compression_ratio = original_size / compressed_size
    
    ssim_score, _ = ssim(img, decimg, channel_axis=2, full=True)
    
    return original_size, compressed_size, compression_ratio, ssim_score


def create_comparison_preview(input_path, output_path, preview_path="comparison_preview.jpg"):
    """Create and open a side-by-side original/compressed preview."""
    original = Image.open(input_path).convert("RGB")
    compressed = Image.open(output_path).convert("RGB")

    max_height = 700
    if original.height > max_height:
        ratio = max_height / original.height
        original = original.resize((int(original.width * ratio), max_height))

    compressed = compressed.resize(original.size)

    padding = 20
    label_height = 40
    preview_width = original.width * 2 + padding * 3
    preview_height = original.height + padding * 2 + label_height

    preview = Image.new("RGB", (preview_width, preview_height), "white")
    draw = ImageDraw.Draw(preview)

    left_label = "Original Image"
    right_label = "Compressed Image"
    draw.text((padding, padding), left_label, fill="black")
    draw.text((original.width + padding * 2, padding), right_label, fill="black")

    preview.paste(original, (padding, padding + label_height))
    preview.paste(compressed, (original.width + padding * 2, padding + label_height))
    preview.save(preview_path)

    if platform.system() == "Darwin":
        subprocess.run(["open", preview_path], check=False)
    elif platform.system() == "Windows":
        os.startfile(preview_path)
    else:
        subprocess.run(["xdg-open", preview_path], check=False)

    return preview_path


def predict_and_compress(input_path, output_path, uncertainty_threshold=35):
    """Predict optimal quality and compress image"""
    print(f"\nProcessing: {input_path}")
    
    print("Extracting features...")
    # ✅ FIX: Use input_path (not image_path)
    features = get_deep_features(input_path)
    print(f"  Raw features shape: {features.shape}")
    
    print("Applying PCA transformation...")
    features_reduced = pca_model.transform([features])
    print(f"  Reduced features shape: {features_reduced.shape}")
    
    pred_median = models_dict[0.5].predict(features_reduced)[0]
    pred_lower = models_dict[0.1].predict(features_reduced)[0]
    pred_upper = models_dict[0.9].predict(features_reduced)[0]
    uncertainty = pred_upper - pred_lower
    
    print(f"\nPrediction:")
    print(f"  Predicted quality: {pred_median:.1f} ± {uncertainty/2:.1f}")
    print(f"  Confidence interval: [{pred_lower:.1f}, {pred_upper:.1f}]")
    
    if uncertainty > uncertainty_threshold:
        print(f"\n⚠️  High uncertainty ({uncertainty:.1f})!")
        final_quality = (pred_median * 0.7) + (pred_lower * 0.3)
    else:
        print(f"\n✓ Uncertainty acceptable ({uncertainty:.1f})")
        final_quality = pred_median
    
    final_quality = np.clip(final_quality, 50, 95)
    
    print(f"\nUsing JPEG quality: {final_quality:.1f}")
    
    orig_size, comp_size, ratio, ssim_score = compress_image(
        input_path, output_path, final_quality
    )
    
    print(f"\n{'='*60}")
    print(f"COMPRESSION RESULTS")
    print(f"{'='*60}")
    print(f"  Original size:      {orig_size:.2f} KB")
    print(f"  Compressed size:    {comp_size:.2f} KB")
    print(f"  Compression ratio:  {ratio:.2f}x")
    print(f"  Size reduction:     {(1 - comp_size/orig_size)*100:.1f}%")
    print(f"  SSIM (quality):     {ssim_score:.4f}")
    
    if ssim_score >= 0.92:
        print(f"  Quality Status:     ✅ EXCELLENT (SSIM ≥ 0.92)")
    elif ssim_score >= 0.90:
        print(f"  Quality Status:     ✅ GOOD (SSIM ≥ 0.90)")
    else:
        print(f"  Quality Status:     ⚠️  ACCEPTABLE (SSIM < 0.90)")
    
    print(f"  Output saved to:    {output_path}")
    print(f"{'='*60}")

    preview_path = create_comparison_preview(input_path, output_path)
    print(f"  Preview saved to:   {preview_path}")
    
    return final_quality, uncertainty, ssim_score


# ==================== Main ====================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python predict_compression.py <image_path>")
        print("Example: python predict_compression.py test_image.png\n")
        sys.exit(1)
    
    input_image = sys.argv[1]
    output_image = "compressed_output.jpg"
    
    if not os.path.exists(input_image):
        print(f"Error: File '{input_image}' not found!")
        sys.exit(1)
    
    predict_and_compress(input_image, output_image)
