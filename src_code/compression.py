import os
import io 
import cv2
import subprocess
import torch
import numpy as np
from PIL import Image
import imagecodecs
from piq import psnr, ssim, brisque
import torchvision.transforms as transforms
import csv
import pyjxl
import glymur
import math

cjxl_path = r"C:/Program Files/jxl-x64-windows-static/cjxl.exe"  # Replace with actual path to cjxl.exe
djxl_path = r"C:/Program Files/jxl-x64-windows-static/djxl.exe"


# Global transform
transform = transforms.ToTensor()

def evaluate_metrics(img1, img2):
    if img1.mode != "RGB":
        img1 = img1.convert("RGB")
    if img2.mode != "RGB":
        img2 = img2.convert("RGB")

    x = transform(img1).unsqueeze(0)
    y = transform(img2).unsqueeze(0)

    if x.shape[1] != y.shape[1]:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.shape[1] == 1:
            y = y.repeat(1, 3, 1, 1)

    return {
        "PSNR": psnr(x, y, data_range=1.0).item(),
        "SSIM": ssim(x, y).item(),
        "BRISQUE": brisque(y).item()
    }


def compress_jpeg(input_path, output_path, target_size_kb, min_quality=1, max_quality=25):
    img1 = Image.open(input_path).convert("RGB")
    target_bytes = target_size_kb * 1024

    low = min_quality
    high = max_quality
    best_quality = low
    best_data = None

    while low <= high:
        quality = (low + high) // 2
        buffer = io.BytesIO()
        img1.save(buffer, "JPEG",
                  quality=quality,
                  subsampling=2,        # 4:2:0 chroma subsampling (most compressed)
                  optimize=True,        # Huffman table optimization
                  progressive=True)     # Progressive encoding
        size = buffer.tell()

        if size <= target_bytes:
            best_quality = quality
            best_data = buffer.getvalue()
            low = quality + 1  # Try higher quality next
        else:
            high = quality - 1  # Try lower quality next

    if best_data is not None:
        with open(output_path, "wb") as f:
            f.write(best_data)

        img2 = Image.open(output_path).convert("RGB")
        return evaluate_metrics(img1, img2)  # Assumes this is defined

    return {"error": f"Could not compress to target size. Best achieved: {size // 1024}KB"}


'''
def compress_jpeg(input_path, output_path, target_size_kb, max_attempts=3000):
    img1 = Image.open(input_path).convert("RGB")
    target_bytes = target_size_kb * 1024
    
    low = 1
    high = 95
    best_quality = low
    best_size = float('inf')
    
    for attempt in range(max_attempts):
        if low > high:
            break
            
        quality = (low + high) // 2
        buffer = io.BytesIO()
        img1.save(buffer, "JPEG", quality=quality)
        size = buffer.tell()
        
        if size <= target_bytes:
            best_quality = quality
            best_size = size
            low = quality + 1  # Try higher quality
        else:
            high = quality - 1  # Try lower quality
    
    if best_size <= target_bytes:
        # Save the best found version
        buffer = io.BytesIO()
        img1.save(buffer, "JPEG", quality=best_quality)
        with open(output_path, "wb") as f:
            f.write(buffer.getvalue())
        
        img2 = Image.open(output_path).convert("RGB")
        return evaluate_metrics(img1, img2)
    
    return {"error": f"Could not compress to target size. Best achieved: {best_size//1024}KB"}
'''

'''
def compress_jp2(input_path, output_path, target_size_kb, min_q=0, max_q=1000):
    img1 = Image.open(input_path).convert("RGB")
    img_cv = cv2.imread(input_path)
    target_bytes = target_size_kb * 1024
    best_q = -1

    while min_q <= max_q:
        mid_q = (min_q + max_q) // 2
        cv2.imwrite(output_path, img_cv, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, mid_q])
        size = os.path.getsize(output_path)

        if size <= target_bytes:
            best_q = mid_q
            min_q = mid_q + 1
        else:
            max_q = mid_q - 1

    if best_q != -1:
        cv2.imwrite(output_path, img_cv, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, best_q])
        img2 = Image.open(output_path).convert("RGB")
        return evaluate_metrics(img1, img2)
    return {"error": "JP2 compression failed"}


def compress_jp2(
    input_path,
    output_path,
    target_size_kb,
    min_q=1,
    max_q=1000,
    max_attempts=12,  # Reduced from 15 (faster convergence)
):
    """Faster JPEG2000 compression with binary search (uses in-memory buffers)."""
    # Read image once (faster than reopening)
    img_cv = cv2.imread(input_path)
    if img_cv is None:
        return {"error": "Failed to read input image"}

    target_bytes = target_size_kb * 1024
    best_q = -1
    best_size = float('inf')

    # Start with a reasonable mid-range guess (faster convergence)
    low, high = min_q, max_q

    for _ in range(max_attempts):
        if low > high:
            break

        mid_q = (low + high) // 2

        # Use imencode (in-memory, no disk I/O)
        success, encoded_img = cv2.imencode(
            ".jp2",
            img_cv,
            [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, mid_q],
        )

        if not success:
            continue  # Skip failed attempts

        size = len(encoded_img)  # Get size from buffer

        if size <= target_bytes:
            best_q = mid_q
            best_size = size
            low = mid_q + 1  # Try higher quality
        else:
            high = mid_q - 1  # Try lower quality

        # Early exit if exact match
        if size == target_bytes:
            break

    # Save the best result (only 1 disk write)
    if best_q != -1:
        cv2.imwrite(
            output_path,
            img_cv,
            [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, best_q],
        )
        
        # Optional: Compute metrics (if needed)
        img_original = Image.open(input_path).convert("RGB")
        img_compressed = Image.open(output_path).convert("RGB")
        metrics = evaluate_metrics(img_original, img_compressed)
        metrics.update({
            "achieved_size_kb": best_size // 1024,
            "quality": best_q,
        })
        return metrics

    return {
        "error": "Could not reach target size",
        "best_size_kb": best_size // 1024,
        "best_quality": best_q,
    }
'''

def compress_jp2(input_path, output_path, target_size_kb, min_q=1, max_q=1000, max_attempts=1000):
    """Optimized JPEG2000 compression using OpenCV with in-memory binary search."""
    target_bytes = target_size_kb * 1024

    # Check if original is already small enough
    original_size = os.path.getsize(input_path)
    if original_size <= target_bytes:
        #shutil.copy2(input_path, output_path)
        return {
            "note": "Original image already smaller than target",
            "achieved_size_kb": original_size // 1024,
            "quality": "original"
        }

    # Read input image once
    img_cv = cv2.imread(input_path)
    if img_cv is None:
        return {"error": "Failed to read input image"}

    best_q = -1
    best_buf = None
    best_size = float("inf")

    low, high = min_q, max_q

    for _ in range(max_attempts):
        if low > high:
            break

        mid_q = (low + high) // 2

        success, encoded_img = cv2.imencode(
            ".jp2", img_cv, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, mid_q]
        )
        if not success:
            continue

        size = len(encoded_img)

        if size <= target_bytes:
            best_q = mid_q
            best_buf = encoded_img
            best_size = size
            low = mid_q + 1  # Try higher quality
        else:
            high = mid_q - 1

        # Close enough (within 1%)
        if abs(size - target_bytes) < 0.01 * target_bytes:
            best_q = mid_q
            best_buf = encoded_img
            best_size = size
            break

    # Save best result to disk
    if best_buf is not None:
        with open(output_path, "wb") as f:
            f.write(best_buf.tobytes())

        # Optional: compute metrics
        img_original = Image.open(input_path).convert("RGB")
        img_compressed = Image.open(output_path).convert("RGB")
        metrics = evaluate_metrics(img_original, img_compressed)
        metrics.update({
            "achieved_size_kb": best_size // 1024,
            "quality": best_q,
        })
        return metrics

    return {
        "error": "Could not reach target size",
        "best_size_kb": best_size // 1024,
        "best_quality": best_q,
    }

def compress_jxr(input_path, output_path, target_size_kb, min_quality=0, max_quality=100):
    img1 = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    img1 =  cv2.resize(img1, (0, 0), fx=0.9, fy=0.9, interpolation=cv2.INTER_AREA)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = Image.fromarray(img1)
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (0, 0), fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
    if img is None:
        return {"error": "Image read failed"}

    target_bytes = target_size_kb * 1024
    low, high = min_quality, max_quality
    best_data = None

    while low <= high:
        q = (low + high) // 2
        jxr_data = imagecodecs.jpegxr_encode(img, level=q / 100.0)

        if len(jxr_data) <= target_bytes:
            best_data = jxr_data
            low = q + 1
        else:
            high = q - 1

    if best_data:
        with open(output_path, "wb") as f:
            f.write(best_data)
        img2 = imagecodecs.jxr_decode(best_data)
        img2_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        return evaluate_metrics(img1, img2_pil)
    return {"error": "JXR compression failed"}



def compress_jxl(input_path, output_path, target_size_kb, start_quality=5, max_quality=100, step=2):
    img1 = Image.open(input_path).convert("RGB")
    best_quality = start_quality
    best_size = None

    for quality in range(start_quality, max_quality + 1, step):
        subprocess.run(["magick", input_path, "-quality", str(quality), output_path], check=True)

        if not os.path.exists(output_path):
            break

        size_kb = os.path.getsize(output_path) / 1024
       # print(f"Quality {quality} -> {size_kb:.1f} KB")

        if size_kb <= target_size_kb:
            best_quality = quality
            best_size = size_kb
        else:
            break

    #print(f"Using best quality {best_quality} ({best_size:.1f} KB)")
    subprocess.run(["magick", input_path, "-quality", str(best_quality), output_path], check=True)

    # Evaluate metrics
    result = subprocess.run(
        ["magick", output_path, "png:-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )
    img2 = Image.open(io.BytesIO(result.stdout))
    return evaluate_metrics(img1, img2)




def compress_all_images_by_camera(main_input_dir, main_output_dir, target_size_kb):
    formats = ["jpeg", "jp2", "jxr", "jxl"]
    compressors = {
        "jpeg": compress_jpeg,
        "jp2": compress_jp2,
        "jxr": compress_jxr,
        "jxl": compress_jxl
    }

    os.makedirs(main_output_dir, exist_ok=True)

    # Loop through each camera folder in the main input directory
    for camera_name in os.listdir(main_input_dir):
        camera_input_path = os.path.join(main_input_dir, camera_name)
        if not os.path.isdir(camera_input_path):
            continue  # Skip files

        print(f"\n Processing Camera: {camera_name}")

        # Create base output path for this camera
        camera_output_path = os.path.join(main_output_dir, camera_name)
        os.makedirs(camera_output_path, exist_ok=True)

        # Create subfolders for each compression format under this camera
        for fmt in formats:
            os.makedirs(os.path.join(camera_output_path, fmt), exist_ok=True)

        # Store metrics for CSV
        csv_data = []

        # Loop through all images in this camera's folder
        for filename in os.listdir(camera_input_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                input_path = os.path.join(camera_input_path, filename)
                print(f"\n Compressing: {filename}")
                image_name = os.path.splitext(filename)[0]

                for fmt in formats:
                    output_path = os.path.join(camera_output_path, fmt, f"{image_name}.{fmt}")
                    metrics = compressors[fmt](input_path, output_path, target_size_kb)

                    if "error" in metrics:
                        print(f" {fmt.upper()} - {metrics['error']}")
                        csv_data.append({
                            "Image": filename,
                            "Format": fmt.upper(),
                            "PSNR": None,
                            "SSIM": None,
                            "BRISQUE": None,
                            "Status": metrics["error"]
                        })
                    else:
                        print(f"{fmt.upper()} - PSNR: {metrics['PSNR']:.2f}, SSIM: {metrics['SSIM']:.4f}, BRISQUE: {metrics['BRISQUE']:.2f}")
                        csv_data.append({
                            "Image": filename,
                            "Format": fmt.upper(),
                            "PSNR": round(metrics["PSNR"], 2),
                            "SSIM": round(metrics["SSIM"], 4),
                            "BRISQUE": round(metrics["BRISQUE"], 2),
                            "Status": "OK"
                        })

        # Save per-camera CSV
        csv_path = os.path.join(camera_output_path, f"{camera_name}_metrics.csv")
        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = ["Image", "Format", "PSNR", "SSIM", "BRISQUE", "Status"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)

        print(f"\n Metrics saved for {camera_name}: {csv_path}")


# Example usage
main_input_folder = "train_images"
main_output_folder = "compressed_results_10kb/train_images"
target_kb = 10

compress_all_images_by_camera(main_input_folder, main_output_folder, target_kb)
