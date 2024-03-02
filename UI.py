import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from main import OUTPUT_IMAGES_DIR, OUTPUT_LABELS_DIR

def read_yolo_annotation(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    annotations = []

    for line in lines:
        parts = line.strip().split()

        # ตรวจสอบว่ามีข้อมูลอย่างน้อย 5 ส่วนหรือไม่
        if len(parts) >= 5:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            annotation = {
                'class_id': class_id,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            }

            annotations.append(annotation)

    return annotations

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.images_dir = OUTPUT_IMAGES_DIR
        self.labels_dir = OUTPUT_LABELS_DIR
        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.current_index = 0

        # สร้าง Frame สำหรับกราฟ
        self.plot_frame = tk.Frame(root)
        self.plot_frame.pack()

        # สร้าง Button สำหรับ Next Image
        self.next_button = tk.Button(root, text="Next Image", command=self.next_image)
        self.next_button.pack()

        # โหลดและแสดงรูปแรก
        self.load_image()

    def load_image(self):
        # ดึงข้อมูลรูปภาพและป้ายกำกับ
        image_file = self.image_files[self.current_index]
        image_path = os.path.join(self.images_dir, image_file)
        label_path = os.path.join(self.labels_dir, image_file.replace('.jpg', '.txt'))
        annotations = read_yolo_annotation(label_path)

        # สร้างกราฟ
        fig, ax = plt.subplots(1, frameon=False)
        ax.set_axis_off()
        img = Image.open(image_path)
        ax.imshow(img)

        # แสดงป้ายกำกับบนกราฟ
        for annotation in annotations:
            x_center = annotation['x_center']
            y_center = annotation['y_center']
            width = annotation['width']
            height = annotation['height']

            x_top_left = x_center - width / 2
            y_top_left = y_center - height / 2

            rect = patches.Rectangle((x_top_left * img.width, y_top_left * img.height),
                                     width * img.width, height * img.height,
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        # แปลงกราฟเป็นรูปภาพและแสดงใน Tkinter
        tk_img = ImageTk.PhotoImage(self.fig_to_img(fig))

        # สร้าง Label แสดงรูปภาพบน Tkinter
        img_label = tk.Label(self.plot_frame, image=tk_img)
        img_label.img = tk_img
        img_label.pack()

    def next_image(self):
        # เลื่อนไปที่รูปภาพถัดไป
        self.current_index += 1
        if self.current_index < len(self.image_files):
            # ลบ Label เก่า
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
            # โหลดรูปภาพใหม่
            self.load_image()

    def fig_to_img(self, fig):
        """แปลงกราฟเป็นรูปภาพ"""
        fig.canvas.draw()
        img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        return img

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
