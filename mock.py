import tkinter as tk
from tkinter import filedialog
import os
import subprocess
from functools import partial
import cv2
import albumentations as A
from tqdm import tqdm


def browse_button(input_entry):
    filename = filedialog.askdirectory()
    input_entry.delete(0, tk.END)
    input_entry.insert(0, filename)


def augment_images(input_images_dir, input_labels_dir, output_images_dir, output_labels_dir):
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    transform = A.Compose([
        A.Rotate(limit=90, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT, always_apply=True, p=1),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category'], min_visibility=1))

    image_files = [f for f in os.listdir(input_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    count = 1
    num_images_to_rotate = 1

    for image_file in tqdm(image_files, desc="Processing Images"):
        image_path = os.path.join(input_images_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_file = os.path.join(input_labels_dir, image_file.replace('.jpg', '.txt'))
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

            augmented_image_path = os.path.join(output_images_dir, f"{count}_augmented_{i}.jpg")
            cv2.imwrite(augmented_image_path, cv2.cvtColor(transformed["image"], cv2.COLOR_RGB2BGR))

            augmented_label_path = os.path.join(output_labels_dir, f"{count}_augmented_{i}.txt")
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
    subprocess.Popen(["python", "ui.py"])


def start_augmentation():
    input_images_dir = input_images_entry.get()
    input_labels_dir = input_labels_entry.get()
    output_images_dir = output_images_entry.get()
    output_labels_dir = output_labels_entry.get()

    augment_images(input_images_dir, input_labels_dir, output_images_dir, output_labels_dir)


root = tk.Tk()
root.title("Image Augmentation")

input_images_label = tk.Label(root, text="Input Images Directory:")
input_images_label.grid(row=0, column=0, sticky="W")
input_images_entry = tk.Entry(root, width=50)
input_images_entry.grid(row=0, column=1, padx=5, pady=5)
input_images_button = tk.Button(root, text="Browse", command=lambda: browse_button(input_images_entry))
input_images_button.grid(row=0, column=2)

input_labels_label = tk.Label(root, text="Input Labels Directory:")
input_labels_label.grid(row=1, column=0, sticky="W")
input_labels_entry = tk.Entry(root, width=50)
input_labels_entry.grid(row=1, column=1, padx=5, pady=5)
input_labels_button = tk.Button(root, text="Browse", command=lambda: browse_button(input_labels_entry))
input_labels_button.grid(row=1, column=2)

output_images_label = tk.Label(root, text="Output Images Directory:")
output_images_label.grid(row=2, column=0, sticky="W")
output_images_entry = tk.Entry(root, width=50)
output_images_entry.grid(row=2, column=1, padx=5, pady=5)
output_images_button = tk.Button(root, text="Browse", command=lambda: browse_button(output_images_entry))
output_images_button.grid(row=2, column=2)

output_labels_label = tk.Label(root, text="Output Labels Directory:")
output_labels_label.grid(row=3, column=0, sticky="W")
output_labels_entry = tk.Entry(root, width=50)
output_labels_entry.grid(row=3, column=1, padx=5, pady=5)
output_labels_button = tk.Button(root, text="Browse", command=lambda: browse_button(output_labels_entry))
output_labels_button.grid(row=3, column=2)

start_button = tk.Button(root, text="Start Augmentation", command=start_augmentation)
start_button.grid(row=4, column=1, pady=10)

root.mainloop()
