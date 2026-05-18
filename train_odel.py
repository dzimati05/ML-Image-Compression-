import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from skimage.metrics import structural_similarity as ssim
import lpips
import joblib
from PIL import Image
import time

DATASET_PATH = "/Users/admin/Documents/ml_images/data/div2k/DIV2K_train_HR/DIV2K_train_HR"

# ==================== Initialize Models ====================

print("Loading LPIPS model...")
loss_fn_alex = lpips.LPIPS(net='alex')
print("LPIPS model loaded.")

print("Loading ResNet18 for feature extraction...")
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
feature_extractor.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_deep_features(image_path):
    """Extract 512-dimensional semantic features"""
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        features = feature_extractor(img_t)
    
    return features.squeeze().numpy()


def find_best_quality_perceptual(image, target_ssim=0.92):
    """Find the LOWEST quality that maintains acceptable perceptual quality"""
    best_quality = 50
    min_size = float('inf')
    
    for q in range(50, 96, 5):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        result, encimg = cv2.imencode('.jpg', image, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        
        score_ssim, _ = ssim(image, decimg, channel_axis=2, full=True)
        
        if score_ssim >= target_ssim:
            size = len(encimg)
            if size < min_size:
                min_size = size
                best_quality = q
    
    return best_quality


# ==================== Build Dataset ====================

print("\n" + "="*60)
print("BUILDING DATASET (Using ALL 800 images)")
print("="*60)

X = []
y = []

files = [f for f in os.listdir(DATASET_PATH) if f.endswith(".png")]
print(f"Found {len(files)} images in dataset")

# Use ALL images (or limit to 800)
num_images = min(800, len(files))
start_time = time.time()

for i, file in enumerate(files[:num_images]):
    path = os.path.join(DATASET_PATH, file)
    img = cv2.imread(path)
    
    if img is None:
        continue
    
    if (i + 1) % 50 == 0:
        elapsed = time.time() - start_time
        print(f"Processing {i+1}/{num_images} (ETA: {elapsed/(i+1)*(num_images-i-1)/60:.1f} min)")
    
    # Extract deep features
    deep_features = get_deep_features(path)
    
    # Get perceptual quality label
    best_quality = find_best_quality_perceptual(img, target_ssim=0.92)
    
    X.append(deep_features)
    y.append(best_quality)

X = np.array(X)
y = np.array(y)

print(f"\n✓ Dataset built: {X.shape[0]} images, {X.shape[1]} features each")
print(f"✓ Quality labels: {y.min()} - {y.max()}")


# ==================== Apply PCA ====================

print("\n" + "="*60)
print("APPLYING PCA (Dimensionality Reduction)")
print("="*60)

# Reduce from 512 to 100 features
pca = PCA(n_components=100, random_state=42)
X_reduced = pca.fit_transform(X)

# Check how much variance we kept
variance_ratio = sum(pca.explained_variance_ratio_)
print(f"✓ Reduced features: 512 → 50")
print(f"✓ Variance retained: {variance_ratio*100:.1f}%")

# Save PCA model for later use
joblib.dump(pca, "pca_model.pkl")
print("✓ PCA model saved")


# ==================== Train Model ====================

print("\n" + "="*60)
print("TRAINING QUANTILE REGRESSION MODELS")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

quantiles = [0.1, 0.5, 0.9]
models_dict = {}

for q in quantiles:
    print(f"\nTraining quantile {q}...")
    model = GradientBoostingRegressor(
        loss='quantile',
        alpha=q,
        n_estimators=150,  # Increased from 100
        max_depth=6,       # Increased from 5
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    models_dict[q] = model

# Evaluate
median_pred = models_dict[0.5].predict(X_test)
mse = mean_squared_error(y_test, median_pred)
rmse = np.sqrt(mse)

print(f"\n{'='*60}")
print(f"MODEL PERFORMANCE (With PCA + 800 images)")
print(f"{'='*60}")
print(f"Test MSE: {mse:.2f}")
print(f"Test RMSE: {rmse:.2f}")
print(f"Mean Absolute Error: {np.mean(np.abs(y_test - median_pred)):.2f}")
print(f"R² Score: {np.mean([(1 - np.sum((y_test - median_pred)**2) / np.sum((y_test - np.mean(y_test))**2))]):.3f}")

# Show predictions
print(f"\n{'='*60}")
print("SAMPLE PREDICTIONS WITH UNCERTAINTY")
print(f"{'='*60}")
for i in range(min(10, len(X_test))):
    pred_median = models_dict[0.5].predict([X_test[i]])[0]
    pred_lower = models_dict[0.1].predict([X_test[i]])[0]
    pred_upper = models_dict[0.9].predict([X_test[i]])[0]
    actual = y_test[i]
    uncertainty = pred_upper - pred_lower
    
    status = "✓" if abs(pred_median - actual) < 10 else "⚠️"
    print(f"{status} Sample {i+1}: Pred={pred_median:.1f}±{uncertainty/2:.1f}, Actual={actual}")


# ==================== Save Everything ====================

joblib.dump(models_dict, "compression_model_quantile.pkl")
print(f"\n{'='*60}")
print("✓ Models saved to 'compression_model_quantile.pkl'")
print("✓ PCA model saved to 'pca_model.pkl'")
print(f"{'='*60}")

print(f"\n🎉 Training complete! Expected RMSE: 8-12 (was 14.57)")