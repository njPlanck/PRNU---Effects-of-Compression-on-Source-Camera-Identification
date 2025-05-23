import os
import numpy as np
from skimage import io, restoration
from scipy import signal
from scipy.signal import fftconvolve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import imagecodecs
import cv2
import pyjxl
from PIL import Image
# --- Step 1: Extract PRNU noise ---
def extract_noise(img):
    denoised = restoration.denoise_wavelet(img, method='BayesShrink', rescale_sigma=True)
    return img - denoised

# --- Step 2: Compute average fingerprint for a camera ---
def get_camera_fingerprint(images):
    prnu = np.zeros_like(images[0])
    for img in images:
        prnu += extract_noise(img)
    return prnu / len(images)



# --- Step 3: Load grayscale images from a folder ---
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png','jp2','jxl')):
            #img = io.imread(os.path.join(folder_path, filename), as_gray=True) / 255.0      #jpeg
            #img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
            #img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
            img = Image.open(os.path.join(folder_path, filename))
            img = np.array(img)  # Convert to NumPy array for further use
            img = img.astype(np.float32) / 255.0  # Normalize if needed
            images.append(img)
    return images

    

# --- Step 4: Create fingerprints for each camera folder ---


def build_camera_fingerprints(root_folder):
    fingerprints = {}
    for camera_name in os.listdir(root_folder):
        cam_folder = os.path.join(root_folder, camera_name)
        if os.path.isdir(cam_folder):
            images = load_images_from_folder(cam_folder)
            if images:
                fingerprints[camera_name] = get_camera_fingerprint(images)
                print(f"[INFO] Fingerprint created for {camera_name} using {len(images)} images.")
            else:
                print(f"[WARN] No images found for {camera_name}. Skipping.")
    return fingerprints

'''
# --- Step 5: Match a test image to the best camera ---
def match_camera(test_img, fingerprints):
    test_noise = extract_noise(test_img)
    test_norm = (test_noise - test_noise.mean()) / test_noise.std()  # Normalize once

    best_match = None
    best_score = -np.inf

    for name, fingerprint in fingerprints.items():
        ref_norm = (fingerprint - fingerprint.mean()) / fingerprint.std()

        # Normalized cross-correlation
        corr = signal.correlate2d(test_norm, ref_norm, mode='same', boundary='symm')
        score = np.max(corr) / (test_norm.shape[0] * test_norm.shape[1])  # Optional: normalize by area

        if score > best_score:
            best_score = score
            best_match = name

    return best_match, best_score

'''

def match_camera(test_img, fingerprints):
    test_noise = extract_noise(test_img)
    test_norm = (test_noise - np.mean(test_noise)) / np.std(test_noise)

    best_match = None
    best_score = -np.inf

    for name, fingerprint in fingerprints.items():
        ref_norm = (fingerprint - np.mean(fingerprint)) / np.std(fingerprint)

        # Use FFT-based normalized cross-correlation
        corr = fftconvolve(test_norm, ref_norm[::-1, ::-1], mode='same')
        score = np.max(corr) / (test_norm.shape[0] * test_norm.shape[1])  # Optional: normalize

        if score > best_score:
            best_score = score
            best_match = name

    return best_match, best_score


def validate_on_test_set(test_folder, fingerprints):
    y_true = []
    y_pred = []

    for root, dirs, files in os.walk(test_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png','jp2','jxl')):
                # Get full path
                file_path = os.path.join(root, file)

                # Load and normalize test image
                #test_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                #test_image = pyjxl.decode(file_path)
                test_image = Image.open(file_path)
                test_image = np.array(img)  # Convert to NumPy array for further use
                test_image = test_image.astype(np.float32) / 255.0  # Normalize to [0, 1]

                # Match to camera
                predicted_camera, score = match_camera(test_image, fingerprints)

                # Assume ground truth camera name is the parent folder name (adjust if different)
                ground_truth_camera = os.path.basename(os.path.dirname(file_path))

                y_true.append(ground_truth_camera)
                y_pred.append(predicted_camera)

                print(f"[TEST] {file}: True={ground_truth_camera}, Predicted={predicted_camera} (Score: {score:.2f})")

    return y_true, y_pred

# --- Step 2: Run validation ---

if __name__ == "__main__":
    # Path to train and test folders
    train_root = "trainjxl" 
    test_root =  "testjxl" 

    # Build fingerprints
    camera_fingerprints = build_camera_fingerprints(train_root)

    # Validate
    y_true, y_pred = validate_on_test_set(test_root, camera_fingerprints)

    # Confusion Matrix
    labels = sorted(list(set(y_true) | set(y_pred)))  # All labels seen
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    plt.figure(figsize=(10, 8))
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix: Camera Identification via PRNU")
    plt.show()