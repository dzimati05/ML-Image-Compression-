import os

DATASET_PATH = "/Users/admin/Documents/ml_images/data/div2k/DIV2K_train_HR/DIV2K_train_HR"

files = os.listdir(DATASET_PATH)

print("Number of images:", len(files))
print("First 5 images:", files[:5])

