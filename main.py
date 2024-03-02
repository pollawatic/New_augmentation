import os
import cv2
import albumentations as A
from tqdm import tqdm

INPUT_IMAGES_DIR = "C:/Users/User/Documents/GitHub/New_augmentation/input/images"
INPUT_LABELS_DIR = "C:/Users/User/Documents/GitHub/New_augmentation/input/labels"
OUTPUT_IMAGES_DIR = "C:/Users/User/Documents/GitHub/New_augmentation/output/images"
OUTPUT_LABELS_DIR = "C:/Users/User/Documents/GitHub/New_augmentation/output/labels"

os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

transform = A.Compose([
    A.Rotate(limit=90, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT, always_apply=True, p=1),
], bbox_params=A.BboxParams(format='yolo', label_fields=['category'], min_visibility=1))

image_files = [f for f in os.listdir(INPUT_IMAGES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]

count = 1
num_images_to_rotate = 1

for image_file in tqdm(image_files, desc="Processing Images"):
    image_path = os.path.join(INPUT_IMAGES_DIR, image_file)
    image = cv2.imread(image_path)
    if image is None:
        continue
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    label_file = os.path.join(INPUT_LABELS_DIR, image_file.replace('.jpg', '.txt'))
    if not os.path.exists(label_file):
        continue

    with open(label_file, 'r') as f:
        yolo_bbox = [float(coord) for coord in f.readline().strip().split()]

    normalized_yolo_bbox = [
        yolo_bbox[1],
        yolo_bbox[2],
        yolo_bbox[3],
        yolo_bbox[4],
    ]

    data = {"image": image, "bboxes": [normalized_yolo_bbox], "category": [int(yolo_bbox[0])]}

    for i in range(num_images_to_rotate):
        transformed = transform(**data)

        augmented_image_path = os.path.join(OUTPUT_IMAGES_DIR, f"{count}_augmented_{i}.jpg")
        cv2.imwrite(augmented_image_path, cv2.cvtColor(transformed["image"], cv2.COLOR_RGB2BGR))

        augmented_label_path = os.path.join(OUTPUT_LABELS_DIR, f"{count}_augmented_{i}.txt")
        with open(augmented_label_path, 'w') as f:
            final_yolo_bbox = [
                transformed["bboxes"][0][0],
                transformed["bboxes"][0][1],    
                transformed["bboxes"][0][2],
                transformed["bboxes"][0][3],
            ]
            f.write(f"{data['category'][0]} {' '.join(map(str, final_yolo_bbox))}")

    count += 1

print("Augmentation complete.")
