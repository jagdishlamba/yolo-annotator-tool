# 🖼️ YOLO Image Annotation Tool

A lightweight, interactive annotation tool for labeling images in YOLO format using OpenCV and PyQt5 Designed and developed by **Jagdish Lamba**, this tool allows you to easily draw, delete, and modify bounding boxes for object detection datasets.

---

## 🚀 Features

- Draw bounding boxes with the mouse
- Annotate multiple classes from a `classes.txt` file
- Save annotations in YOLO format (`.txt`)
- Navigate through images using keyboard
- Save empty annotation files when no objects exist
- Set minimum bounding box size for valid annotations
- Proper GUI for all tasks


---

## 🗂️ Folder Structure

Your image dataset folder should look like this:

```
📂 your_dataset_folder/
├── image1.jpg
├── image2.jpg
└── ...                 # More images
```

Example `classes.txt`:

```
person
car
bicycle
```

---

## 🛠️ How to Use

### 1. Install Requirements

Make sure you have Python 3 installed. Then install dependencies:

```bash
pip install opencv-python numpy pyqt5
```

### 2. Run the Tool

```bash
python annotator.py
```

You will be prompted:

```
Enter the path to the folder containing images:
Enter the bounding box thickness:
Enter the minimum size of bounding box (default 10):
```

> 📌 You can simply press **Enter** at the last prompt to use the default minimum size.

---

## 🎮 Controls

| Action                       | Key / Mouse Input     |
|-----------------------------|-----------------------|
| Draw bounding box           | Left-click & drag     |
| Delete bounding box         | Right-click inside box|
| Change class of box         | Middle-click or scroll|
| Next image                  | `d` key               |
| Previous image              | `a` key               |
| Change current class        | `w` key               |
| Quit the tool               | `s` key               |

---

## 💾 Output Format

- Output files are saved in a subfolder: `annotation/` inside your dataset folder.
- Each image gets a corresponding `.txt` file with YOLO format:
  ```
  <class_id> <x_center> <y_center> <width> <height>
  ```
- If no objects are annotated in an image, an **empty `.txt` file is still saved**.

---

## 📦 Requirements

- Python 3.x
- OpenCV
- NumPy

Install via:

```bash
pip install opencv-python numpy
```

---

## 🧾 License
Developed by Jagdish Lamba  
For research and training purposes.
Feel free to use it.

---

## 🙌 Credits

Special thanks to all open-source contributors, and to the Machine Learning community.
