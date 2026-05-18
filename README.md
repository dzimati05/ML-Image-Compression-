# Adaptive ML Image Compression

A small machine learning project that predicts a JPEG quality setting for an image, then compresses it and reports basic quality and size metrics.

The project uses ResNet18 image features, PCA, and quantile regression to estimate a reasonable compression quality instead of using one fixed setting for every image.

## What It Does

```text
Image
  -> ResNet18 feature extraction
  -> PCA
  -> Quality prediction
  -> JPEG compression
  -> Metrics
```

Current features:

- predicts an adaptive JPEG quality value
- compresses images with OpenCV
- reports original size, compressed size, size reduction, compression ratio, and SSIM
- includes a command-line script
- includes simple Streamlit and Gradio demos
- includes a training script for DIV2K images

## Files

- `train_model.py` trains the model using DIV2K images.
- `predict_compression.py` predicts quality and writes `compressed_output.jpg`.
- `gui_streamlit.py` runs a Streamlit upload demo.
- `gui_gradio.py` runs a Gradio upload demo.
- `pca_model.pkl` stores the fitted PCA model.
- `compression_model_quantile.pkl` stores the trained quantile regression models.

## Usage

Install dependencies:

```bash
pip install opencv-python numpy torch torchvision scikit-learn scikit-image pillow joblib lpips streamlit gradio
```

Compress an image:

```bash
python predict_compression.py path/to/image.jpg
```

Run the Streamlit demo:

```bash
streamlit run gui_streamlit.py
```

Run the Gradio demo:

```bash
python gui_gradio.py
```

Retrain the model:

```bash
python train_model.py
```

The training script currently expects the DIV2K dataset path to be set inside `train_model.py`.

## Notes

This is an adaptive JPEG compression controller, not a new image codec. It does not replace JPEG, WebP, or AVIF. It only tries to choose a better compression quality based on the image.

Future improvements could include a cleaner config file, batch compression, WebP/AVIF support, RAW image support, and a simple API backend.

## License

MIT
