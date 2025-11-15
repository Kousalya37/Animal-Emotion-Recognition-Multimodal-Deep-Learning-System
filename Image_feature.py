import os
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from multiprocessing.dummy import Pool as ThreadPool

device = "cuda" if torch.cuda.is_available() else "cpu"
weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)
model = torch.nn.Sequential(*list(model.children())[:-1])  
model = model.to(device)
model.eval()
preprocess = weights.transforms()

def load_preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        return preprocess(img)
    except Exception as e:
        print(f"Warning: Failed loading {image_path}: {e}")
        return None

def extract_features(image_folder, batch_size=64, num_workers=8):
    image_paths = []
    for root, _, files in os.walk(image_folder):
        for file_name in files:
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file_name))

    print(f"Total {len(image_paths)} images found.")
    pool = ThreadPool(num_workers)
    images = pool.map(load_preprocess_image, image_paths)
    pool.close()
    pool.join()
    valid_items = [(img, path) for img, path in zip(images, image_paths) if img is not None]
    if not valid_items:
        print("No valid images found.")
        return np.array([])
    imgs, valid_paths = zip(*valid_items)

    features = []
    file_list = []
    for i in range(0, len(imgs), batch_size):
        batch_imgs = torch.stack(imgs[i:i+batch_size]).to(device)
        with torch.no_grad():
            batch_feats = model(batch_imgs).squeeze(-1).squeeze(-1).cpu().numpy()
        features.append(batch_feats)
        file_list.extend(valid_paths[i:i+batch_size])
        print(f"Processed batch {i//batch_size + 1} of {(len(imgs)-1)//batch_size + 1}")

    features = np.vstack(features)
    np.save("image_features_fast.npy", features)
    with open("image_files_fast.txt", "w") as f:
        for path in file_list:
            f.write(path + "\n")

    print(f"Extracted features for {len(features)} images and saved.")
    return features

if __name__ == "__main__":
    image_folder = "./image_data" 
    extract_features(image_folder, batch_size=64, num_workers=8)
