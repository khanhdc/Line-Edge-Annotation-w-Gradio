import tempfile
import uuid
from datetime import datetime
import zipfile

import os
import cv2
import json


def find_coordinate(img):
    mask = cv2.inRange(img, (1, 1, 1), (255, 255, 255))
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    center = coords.mean(axis=0)[0]  # mean returns [[x, y]]
    return int(center[0]), int(center[1])


# Save the keypoints text into a file
def save_kp_to_file(keypoints_text):
    uploaded_filename = ""
    keypoints_text = keypoints_text[1:-1]
    if uploaded_filename != "":
        base_name = os.path.splitext(os.path.basename(uploaded_filename))[0]
    else:
        base_name = "annotated"

    temp_path = os.path.join(tempfile.gettempdir(), f"{base_name}.txt")
    with open(temp_path, "w") as f:
        f.write(str(keypoints_text))
    return temp_path


def get_active_layer_index(im):
    layers = im.get("layers", [])
    if not layers:
        return None
    return len(layers) - 1


def generate_PID():
    return str(uuid.uuid4())[:8]


def load_working_session(PID):

    return PID


def retrive_PID_func(PID):
    return PID


def convert_to_target_format(current_PID, label_format):
    subfolders = ["images", "labels"]
    if label_format.upper() == "YOLO":
        label_txt_path = f"workspace/{current_PID}/labels"
        output_yolo_path = f"workspace/{current_PID}/yolo_labels"
        img_path = f"workspace/{current_PID}/images"
        convert_to_yolo_pose(label_txt_path, output_yolo_path, img_path)
        subfolders = ["images", "yolo_labels"]

    elif label_format.upper() == "COCO":
        # Develop the convert to coco format here
        subfolders = ["images", "labels"]

    return subfolders


def download_zip_file(current_PID, label_format):
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_zip_folder_path = f"workspace/{current_PID}"
    output_zip_path = f"{base_zip_folder_path}/annotated_{timestamp_str}.zip"

    subfolders = convert_to_target_format(current_PID, label_format)

    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for subfolder in subfolders:
            folder_path = os.path.join(base_zip_folder_path, subfolder)
            for root, _, files in os.walk(folder_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, base_zip_folder_path)
                    zipf.write(full_path, rel_path)
    return output_zip_path

def download_zip_file_LDC(current_PID):
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_zip_folder_path = f"workspace/{current_PID}"
    output_zip_path = f"{base_zip_folder_path}/annotated_{timestamp_str}.zip"

    # Absolute paths
    img_path = os.path.join(base_zip_folder_path, "images")
    pnb_label_path = os.path.join(base_zip_folder_path, "labels_ldc")
    drawns_path = os.path.join(base_zip_folder_path, "drawns")

    subfolders = [img_path, pnb_label_path, drawns_path]

    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for folder_path in subfolders:
            if not os.path.exists(folder_path):
                continue
            for root, _, files in os.walk(folder_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    # Make path relative to base so zip structure is clean
                    rel_path = os.path.relpath(full_path, base_zip_folder_path)
                    zipf.write(full_path, rel_path)

    return output_zip_path


def handle_zip(current_PID, zip_file):
    EXTRACT_FOLDER = f"workspace/{current_PID}/images"
    os.makedirs(EXTRACT_FOLDER, exist_ok=True)

    if zip_file is None:
        return "No file uploaded."
    try:
        # Extract the uploaded ZIP file
        with zipfile.ZipFile(zip_file.name, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_FOLDER)
            extracted_files = zip_ref.namelist()

        return f"Extracted to {EXTRACT_FOLDER}:\n" + "\n".join(extracted_files)

    except zipfile.BadZipFile:
        return "Not a valid ZIP file."

    except Exception as e:
        return f"Error: {str(e)}"


def get_images(current_PID):
    IMAGE_DIR = f"workspace/{current_PID}/images"
    valid_exts = {".png", ".jpg", ".jpeg"}

    image_files = sorted([
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if os.path.splitext(f)[1].lower() in valid_exts
    ])
    print("return image_files: ", image_files)
    return image_files


def draw_points_on_images(
    image_dir,
    txt_dir,
    output_dir="drawn",
    point_color=(0, 0, 255),
    text_color=(0, 255, 0),
    point_radius=5,
):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_dir, filename)
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(txt_dir, txt_filename)

            if not os.path.exists(txt_path):
                print(f"[Warning] No txt file for {filename}")
                continue

            # Load image
            image = cv2.imread(image_path)

            # Load point data
            with open(txt_path, "r") as f:
                try:
                    points = json.load(f)  # expects dict like {"0": [x, y], ...}
                except json.JSONDecodeError:
                    print(f"[Error] Cannot decode {txt_path}")
                    continue

            # Draw points and keys
            for key, (x, y) in points.items():
                cv2.circle(image, (x, y), point_radius, point_color, -1)
                cv2.putText(
                    image,
                    key,
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )

            # Save the drawn image
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, image)
            #print(f"[Saved] {output_path}")


def draw_points_on_single_image(
    image_path,
    txt_path,
    output_path="drawn_output.png",
    point_color=(0, 0, 255),
    text_color=(0, 255, 0),
    point_radius=5,
):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load txt annotation
    with open(txt_path, "r") as f:
        try:
            points = json.load(f)  # e.g., {"0": [x, y], "1": [x, y]}
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in file: {txt_path}")

    # Draw points
    for key, (x, y) in points.items():
        cv2.circle(image, (x, y), point_radius, point_color, -1)
        cv2.putText(
            image,
            key,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            text_color,
            2,
            cv2.LINE_AA,
        )

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    # print(f"[Saved] Annotated image saved to: {output_path}")


def find_corresponding_image(txt_path, img_path):
    folder = img_path
    base_name = os.path.splitext(os.path.basename(txt_path))[0]
    for ext in [".jpg", ".jpeg", ".png"]:
        img_path = os.path.join(folder, base_name + ext)
        if os.path.isfile(img_path):
            return img_path
    return None  # No image found


def convert_to_yolo_pose(folder_path, output_path, img_path):
    os.makedirs(output_path, exist_ok=True)
    for f in os.listdir(folder_path):
        if f.endswith(".txt"):

            corres_img_path = find_corresponding_image(
                os.path.join(folder_path, f), img_path
            )

            image = cv2.imread(corres_img_path)
            h, w, _ = image.shape
            d = json.load(open(os.path.join(folder_path, f)))
            yolo = [0, 0.5, 0.5, 1.0, 1.0]
            for k in sorted(d, key=int):
                x, y = d[k]
                yolo += [x / w, y / h, 2]
            out_file = os.path.join(output_path, f)
            open(out_file, "w").write(" ".join(map(str, yolo)) + "\n")
