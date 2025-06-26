import cv2
import os
import numpy as np
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm


# ======================
# STEP 1: Frame Extraction with Logo Detection
# ======================


def extract_and_label_frames(video_path, output_folder, logo_template_path, num_frames=1000):
    """Extract frames and generate REAL labels using template matching"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_frames)

    # Load and prepare logo template (30x15)
    logo_template = cv2.imread(logo_template_path, 0)
    assert logo_template is not None, "Logo template not found!"
    logo_template = cv2.resize(logo_template, (30, 15))
    logo_template = cv2.threshold(logo_template, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(f"{output_folder}/labels", exist_ok=True)

    count = 0
    frame_id = 0
    pbar = tqdm(total=num_frames, desc="Extracting frames")

    while cap.isOpened() and count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % step == 0:
            frame_file = f"{output_folder}/frame_{count:04d}.jpg"
            cv2.imwrite(frame_file, frame)

            # Template matching with dynamic threshold
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(gray, logo_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > 0.7:  # Confidence threshold
                x, y = max_loc
                # Convert to YOLO format (normalized)
                x_center = (x + 15) / gray.shape[1]  # 15 = width/2
                y_center = (y + 7.5) / gray.shape[0]  # 7.5 = height/2
                width = 30 / gray.shape[1]
                height = 15 / gray.shape[0]

                with open(f"{output_folder}/labels/frame_{count:04d}.txt", 'w') as f:
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            else:
                print(f"⚠️ Logo not found in frame {count} (confidence={max_val:.2f})")

            count += 1
            pbar.update(1)

        frame_id += 1

    cap.release()
    pbar.close()
    print(f"✅ Extracted {count} frames with labels to {output_folder}")


# ======================
# STEP 2: Prepare YOLO Dataset
# ======================


def prepare_yolo_dataset(raw_folder, output_base="dataset"):
    """Create properly structured YOLO dataset"""
    os.makedirs(f"{output_base}/images/train", exist_ok=True)
    os.makedirs(f"{output_base}/images/val", exist_ok=True)
    os.makedirs(f"{output_base}/labels/train", exist_ok=True)
    os.makedirs(f"{output_base}/labels/val", exist_ok=True)

    # Get only frames with successful detections
    valid_images = [f for f in os.listdir(f"{raw_folder}")
                    if f.endswith('.jpg') and
                    os.path.exists(f"{raw_folder}/labels/{f.replace('.jpg', '.txt')}")]

    train_files, val_files = train_test_split(valid_images, test_size=0.2, random_state=42)

    def copy_files(files, subset):
        for img in files:
            # Copy image
            shutil.copy(
                f"{raw_folder}/{img}",
                f"{output_base}/images/{subset}/{img}"
            )
            # Copy label
            shutil.copy(
                f"{raw_folder}/labels/{img.replace('.jpg', '.txt')}",
                f"{output_base}/labels/{subset}/{img.replace('.jpg', '.txt')}"
            )

    copy_files(train_files, "train")
    copy_files(val_files, "val")

    # Create dataset.yaml
    yaml_content = f"""path: {os.path.abspath(output_base)}
train: images/train
val: images/val
names:
  0: logo

# Small object detection parameters
small_object_scale: 4.0
flipud: 0.0  # Disable vertical flips if logo orientation matters
"""
    with open(f"{output_base}/dataset.yaml", 'w') as f:
        f.write(yaml_content)

    print(f"✅ YOLO dataset prepared with {len(train_files)} train + {len(val_files)} val images")


# ======================
# STEP 3: Train Optimized YOLO Model
# ======================

def train_yolo_model(data_yaml):
    """Train with small-object optimizations"""
    model = YOLO("yolov8n.pt")

    # Custom training configuration
    model.train(
        data=data_yaml,
        epochs=15,
        imgsz=1280,  # Higher resolution for small logos
        batch=4,
        patience=20,  # Early stopping
        single_cls=True,
        augment=True,
        hsv_h=0.015,  # Minimal hue variation
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,  # No rotation if logo orientation is fixed
        translate=0.1,
        scale=0.1,
        shear=0.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.5,
        copy_paste=0.1,  # Synthetic logo augmentation
        name="logo_remover_v2"
    )
    return model


# ======================
# STEP 4: Enhanced Logo Removal
# ======================
def remove_logo_from_video(input_path, model_path, output_path):
    """Advanced removal with hybrid template+YOLO detection and save box-frames"""
    model = YOLO(model_path)
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer (HEVC for better quality)
    fourcc = cv2.VideoWriter_fourcc(*'hev1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create output folder for box frames
    box_frame_folder = "box-frame-folder"
    os.makedirs(box_frame_folder, exist_ok=True)

    # Progress tracking
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Processing video")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect with low confidence threshold
        results = model(frame, conf=0.25, imgsz=1280)

        # Make a copy of frame for saving with boxes
        box_frame = frame.copy()

        # Create mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Ensure minimum 30x15 size
            w, h = x2 - x1, y2 - y1
            if w < 30 or h < 15:
                x1 = max(0, x1 - (30 - w) // 2)
                x2 = min(width, x2 + (30 - w) // 2)
                y1 = max(0, y1 - (15 - h) // 2)
                y2 = min(height, y2 + (15 - h) // 2)

            # Draw box on box_frame
            cv2.rectangle(box_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw mask rectangle
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # Save box_frame
        cv2.imwrite(f"{box_frame_folder}/frame_{frame_idx:05d}.jpg", box_frame)

        # Refine mask
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

        # Inpaint
        inpainted = cv2.inpaint(frame, mask, 3, cv2.INPAINT_NS)
        out.write(inpainted)

        frame_idx += 1
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()
    print(f"✅ Logo removal complete! Saved to {output_path}")
    print(f"✅ Box frames saved to {box_frame_folder}")



# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    # Step 1: Extract and label frames
    extract_and_label_frames(
        video_path="video_with_logo.mp4",
        output_folder="raw_frames",
        logo_template_path="logo.png"  # Your 30x15 logo image
    )

    # Step 2: Prepare YOLO dataset
    prepare_yolo_dataset("raw_frames")

    # Step 3: Train model
    model = train_yolo_model("dataset/dataset.yaml")

    # Step 4: Process video
    remove_logo_from_video(
        input_path="video_with_logo.mp4",
        model_path=model.trainer.best,  # Use best saved weights
        output_path="video_no_logo.mp4"
    )