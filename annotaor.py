"""
Developer : Jagdish Lamba
YOLO Image Annotation Tool with PyQt5 GUI
A comprehensive tool for creating YOLO format annotations with modern GUI
"""

import sys
import os
import cv2
import json
import random
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QLineEdit, 
                            QListWidget, QSpinBox, QFileDialog, QMessageBox,
                            QSplitter, QGroupBox, QGridLayout, QTextEdit,
                            QCheckBox, QSlider, QComboBox, QFrame, QScrollArea)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QIcon

class ImageViewer(QLabel):
    """Custom QLabel for displaying and annotating images"""
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(600, 400)
        self.setStyleSheet("border: 2px solid #3498db; background-color: #2c3e50;")
        self.setAlignment(Qt.AlignCenter)
        self.setText("Select an image folder to start annotation")
        
        # Annotation variables
        self.image = None
        self.scaled_image = None
        self.annotations = []
        self.current_class = 0
        self.box_thickness = 2
        self.min_box_size = 10
        self.class_colors = {}
        self.class_names = []
        
        # Drawing variables
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.mouse_pos = (0, 0)
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # Enable mouse tracking
        self.setMouseTracking(True)
    
    def set_image(self, image_path):
        """Load and display image"""
        self.image = cv2.imread(image_path)
        if self.image is not None:
            self.update_display()
            return True
        return False
    
    def update_display(self):
        """Update the displayed image with annotations"""
        if self.image is None:
            return
            
        # Create working copy
        display_img = self.image.copy()
        h, w = display_img.shape[:2]
        
        # Draw existing annotations
        for cls_id, x_center, y_center, width, height in self.annotations:
            if cls_id < len(self.class_names):
                x1 = int((x_center - width/2) * w)
                y1 = int((y_center - height/2) * h)
                x2 = int((x_center + width/2) * w)
                y2 = int((y_center + height/2) * h)
                
                color = self.class_colors.get(self.class_names[cls_id], (0, 255, 0))
                cv2.rectangle(display_img, (x1, y1), (x2, y2), color, self.box_thickness)
                
                # Class label
                label = self.class_names[cls_id]
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                cv2.rectangle(display_img, (x1, y1-label_size[1]-5), 
                            (x1+label_size[0], y1), color, -1)
                cv2.putText(display_img, label, (x1, y1-5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw current drawing box
        if self.drawing and self.start_point and self.end_point:
            if self.current_class < len(self.class_names):
                color = self.class_colors.get(self.class_names[self.current_class], (0, 255, 0))
                cv2.rectangle(display_img, self.start_point, self.end_point, color, self.box_thickness)
        
        # Draw crosshair
        cv2.line(display_img, (self.mouse_pos[0], 0), (self.mouse_pos[0], h), (0, 255, 255), 1)
        cv2.line(display_img, (0, self.mouse_pos[1]), (w, self.mouse_pos[1]), (0, 255, 255), 1)
        
        # Convert to QImage and scale to fit widget
        rgb_image = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        qt_image = QImage(rgb_image.data, w, h, w * 3, QImage.Format_RGB888)
        
        # Scale image to fit widget while maintaining aspect ratio
        widget_size = self.size()
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            widget_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Calculate scale factor and offset for coordinate conversion
        self.scale_factor = min(widget_size.width() / w, widget_size.height() / h)
        scaled_w = int(w * self.scale_factor)
        scaled_h = int(h * self.scale_factor)
        self.offset_x = (widget_size.width() - scaled_w) // 2
        self.offset_y = (widget_size.height() - scaled_h) // 2
        
        self.setPixmap(scaled_pixmap)
    
    def widget_to_image_coords(self, x, y):
        """Convert widget coordinates to image coordinates"""
        if self.image is None or self.scale_factor == 0:
            return None, None
            
        # Adjust for offset and scale
        img_x = int((x - self.offset_x) / self.scale_factor)
        img_y = int((y - self.offset_y) / self.scale_factor)
        
        # Clamp to image bounds
        h, w = self.image.shape[:2]
        img_x = max(0, min(img_x, w-1))
        img_y = max(0, min(img_y, h-1))
        
        return img_x, img_y
    
    def mousePressEvent(self, event):
        if self.image is None:
            return
            
        x, y = self.widget_to_image_coords(event.x(), event.y())
        if x is None:
            return
            
        if event.button() == Qt.LeftButton:
            self.start_point = (x, y)
            self.end_point = None
            self.drawing = True
            
        elif event.button() == Qt.RightButton:
            self.delete_annotation_at(x, y)
            
        elif event.button() == Qt.MiddleButton:
            self.cycle_class_at(x, y)
    
    def mouseMoveEvent(self, event):
        if self.image is None:
            return
            
        x, y = self.widget_to_image_coords(event.x(), event.y())
        if x is None:
            return
            
        self.mouse_pos = (x, y)
        
        if self.drawing and self.start_point:
            self.end_point = (x, y)
        
        # Always update display to show crosshair movement
        self.update_display()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            if self.start_point and self.end_point:
                self.add_annotation()
            self.start_point = None
            self.end_point = None
    
    def wheelEvent(self, event):
        """Handle mouse wheel for class cycling"""
        if self.image is None:
            return
            
        x, y = self.widget_to_image_coords(event.x(), event.y())
        if x is None:
            return

        self.cycle_class_at(x, y)
        
    
    def add_annotation(self):
        """Add new annotation from current drawing"""
        if not self.start_point or not self.end_point or self.image is None:
            return
            
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        
        # Ensure minimum box size
        if abs(x2 - x1) < self.min_box_size or abs(y2 - y1) < self.min_box_size:
            return
            
        h, w = self.image.shape[:2]
        
        # Convert to YOLO format (normalized)
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        width = abs(x2 - x1) / w
        height = abs(y2 - y1) / h
        
        self.annotations.append((self.current_class, x_center, y_center, width, height))
        self.update_display()
        
        # Notify parent to update annotations display
        if hasattr(self, 'parent_window'):
            self.parent_window.update_annotations_display()
    
    def delete_annotation_at(self, x, y):
        """Delete annotation at given coordinates"""
        if self.image is None:
            return
            
        h, w = self.image.shape[:2]
        
        for i, (cls_id, x_center, y_center, width, height) in enumerate(self.annotations):
            x1 = int((x_center - width/2) * w)
            y1 = int((y_center - height/2) * h)
            x2 = int((x_center + width/2) * w)
            y2 = int((y_center + height/2) * h)
            
            if x1 <= x <= x2 and y1 <= y <= y2:
                del self.annotations[i]
                self.update_display()
                # Notify parent to update annotations display
                if hasattr(self, 'parent_window'):
                    self.parent_window.update_annotations_display()
                break
    
    def cycle_class_at(self, x, y):
        """Cycle through classes for annotation at given coordinates"""
        if self.image is None or not self.class_names:
            return
            
        h, w = self.image.shape[:2]
        
        for i, (cls_id, x_center, y_center, width, height) in enumerate(self.annotations):
            x1 = int((x_center - width/2) * w)
            y1 = int((y_center - height/2) * h)
            x2 = int((x_center + width/2) * w)
            y2 = int((y_center + height/2) * h)
            
            if x1 <= x <= x2 and y1 <= y <= y2:
                new_class = (cls_id + 1) % len(self.class_names)
                self.annotations[i] = (new_class, x_center, y_center, width, height)
                self.update_display()
                # Notify parent to update annotations display
                if hasattr(self, 'parent_window'):
                    self.parent_window.update_annotations_display()
                break

class YOLOAnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Annotation Tool")
        self.setGeometry(100, 100, 1400, 900)
        self.setWindowIcon(QIcon("logo.ico"))
        
        # Initialize variables
        self.images_folder = ""
        self.output_folder = ""
        self.image_files = []
        self.current_image_index = 0
        self.classes = []
        
        # Setup UI
        self.setup_ui()
        self.setup_styles()
        
        # Load settings
        self.load_settings()
    
    def setup_ui(self):
        """Setup the main user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Center panel - Image viewer
        self.image_viewer = ImageViewer()
        self.image_viewer.parent_window = self  # Set parent reference for callbacks
        splitter.addWidget(self.image_viewer)
        
        # Right panel - Info and navigation
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([300, 800, 300])


    def create_left_panel(self):
        """Create the left control panel"""
        panel = QWidget()
        panel.setMaximumWidth(350)
        layout = QVBoxLayout(panel)
        
        # Project Setup Group
        project_group = QGroupBox("Project Setup")
        project_layout = QGridLayout(project_group)
        
        # Images folder
        project_layout.addWidget(QLabel("Images Folder:"), 0, 0)
        self.images_folder_edit = QLineEdit()
        self.images_folder_edit.setReadOnly(True)
        project_layout.addWidget(self.images_folder_edit, 0, 1)
        
        browse_images_btn = QPushButton("Browse")
        browse_images_btn.clicked.connect(self.browse_images_folder)
        project_layout.addWidget(browse_images_btn, 0, 2)
        
        # Output folder
        project_layout.addWidget(QLabel("Output Folder:"), 1, 0)
        self.output_folder_edit = QLineEdit()
        project_layout.addWidget(self.output_folder_edit, 1, 1)
        
        browse_output_btn = QPushButton("Browse")
        browse_output_btn.clicked.connect(self.browse_output_folder)
        project_layout.addWidget(browse_output_btn, 1, 2)
        
        layout.addWidget(project_group)
        
        # Classes Group
        classes_group = QGroupBox("Classes Management")
        classes_layout = QVBoxLayout(classes_group)
        
        # Classes list
        self.classes_list = QListWidget()
        self.classes_list.currentRowChanged.connect(self.on_class_selected)
        classes_layout.addWidget(self.classes_list)
        
        # Class management buttons
        class_btn_layout = QHBoxLayout()
        add_class_btn = QPushButton("Add Class")
        add_class_btn.clicked.connect(self.add_class)
        class_btn_layout.addWidget(add_class_btn)
        
        remove_class_btn = QPushButton("Remove Class")
        remove_class_btn.clicked.connect(self.remove_class)
        class_btn_layout.addWidget(remove_class_btn)
        
        classes_layout.addLayout(class_btn_layout)
        
        # Load/Save classes
        class_file_layout = QHBoxLayout()
        load_classes_btn = QPushButton("Load Classes")
        load_classes_btn.clicked.connect(self.load_classes)
        class_file_layout.addWidget(load_classes_btn)
        
        save_classes_btn = QPushButton("Save Classes")
        save_classes_btn.clicked.connect(self.save_classes)
        class_file_layout.addWidget(save_classes_btn)
        
        classes_layout.addLayout(class_file_layout)
        
        layout.addWidget(classes_group)
        
        # Annotation Settings Group
        settings_group = QGroupBox("Annotation Settings")
        settings_layout = QGridLayout(settings_group)
        
        # Box thickness
        settings_layout.addWidget(QLabel("Box Thickness:"), 0, 0)
        self.thickness_spin = QSpinBox()
        self.thickness_spin.setRange(1, 10)
        self.thickness_spin.setValue(2)
        self.thickness_spin.valueChanged.connect(self.update_thickness)
        settings_layout.addWidget(self.thickness_spin, 0, 1)
        
        # Minimum box size
        settings_layout.addWidget(QLabel("Min Box Size:"), 1, 0)
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(5, 100)
        self.min_size_spin.setValue(10)
        self.min_size_spin.valueChanged.connect(self.update_min_size)
        settings_layout.addWidget(self.min_size_spin, 1, 1)
        
        layout.addWidget(settings_group)
        
        # Current Class Display
        current_class_group = QGroupBox("Current Class")
        current_class_layout = QVBoxLayout(current_class_group)
        
        self.current_class_label = QLabel("No class selected")
        self.current_class_label.setAlignment(Qt.AlignCenter)
        self.current_class_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        current_class_layout.addWidget(self.current_class_label)
        
        layout.addWidget(current_class_group)
        
        # Controls Help
        help_group = QGroupBox("Controls")
        help_layout = QVBoxLayout(help_group)
        
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setMaximumHeight(120)
        help_text.setHtml("""
        <b>Mouse Controls:</b><br>
        • Left Click + Drag: Draw bounding box<br>
        • Right Click: Delete annotation<br>
        • Middle Click: Cycle class of annotation<br>
        • Mouse Wheel: Cycle class of annotation<br><br>
        <b>Keyboard Shortcuts:</b><br>
        • A/D: Previous/Next image<br>
        • W/S: Previous/Next class<br>
        """)
        help_layout.addWidget(help_text)
        
        layout.addWidget(help_group)
        
        layout.addStretch()
        return panel
    
    def create_right_panel(self):
        """Create the right info panel"""
        panel = QWidget()
        panel.setMaximumWidth(350)
        layout = QVBoxLayout(panel)
        
        # Image Navigation Group
        nav_group = QGroupBox("Image Navigation")
        nav_layout = QGridLayout(nav_group)
        
        # Image counter
        self.image_counter_label = QLabel("0 / 0")
        self.image_counter_label.setAlignment(Qt.AlignCenter)
        self.image_counter_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        nav_layout.addWidget(self.image_counter_label, 0, 0, 1, 3)
        
        # Navigation buttons
        prev_btn = QPushButton("← Previous")
        prev_btn.clicked.connect(self.previous_image)
        nav_layout.addWidget(prev_btn, 1, 0)
        
        save_btn = QPushButton("💾 Save")
        save_btn.clicked.connect(self.save_current_annotations)
        nav_layout.addWidget(save_btn, 1, 1)
        
        next_btn = QPushButton("Next →")
        next_btn.clicked.connect(self.next_image)
        nav_layout.addWidget(next_btn, 1, 2)
        
        layout.addWidget(nav_group)
        
        # Current Image Info Group
        info_group = QGroupBox("Current Image")
        info_layout = QVBoxLayout(info_group)
        
        self.image_name_label = QLabel("No image selected")
        self.image_name_label.setWordWrap(True)
        info_layout.addWidget(self.image_name_label)
        
        self.image_size_label = QLabel("Size: -")
        info_layout.addWidget(self.image_size_label)
        
        self.annotations_count_label = QLabel("Annotations: 0")
        info_layout.addWidget(self.annotations_count_label)
        
        layout.addWidget(info_group)
        
        # Annotations List Group
        annotations_group = QGroupBox("Current Annotations")
        annotations_layout = QVBoxLayout(annotations_group)
        
        self.annotations_list = QListWidget()
        self.annotations_list.setMaximumHeight(200)
        annotations_layout.addWidget(self.annotations_list)
        
        clear_all_btn = QPushButton("Clear All Annotations")
        clear_all_btn.clicked.connect(self.clear_all_annotations)
        clear_all_btn.setStyleSheet("background-color: #e74c3c; color: white;")
        annotations_layout.addWidget(clear_all_btn)
        
        layout.addWidget(annotations_group)
        
        # Progress Group
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_label = QLabel("Progress: 0%")
        progress_layout.addWidget(self.progress_label)
        
        # Statistics
        self.stats_label = QLabel("Total Annotations: 0")
        progress_layout.addWidget(self.stats_label)
        
        layout.addWidget(progress_group)

        # Developer Info & Logo
        dev_info_widget = QWidget()
        dev_info_layout = QHBoxLayout(dev_info_widget)

        dev_label = QLabel("Devp by Jagdish Lamba")
        dev_label.setStyleSheet("color: #bdc3c7; font-size: 20px;")

        logo_label = QLabel()
        logo_pixmap = QPixmap("logo.png").scaled(70, 70, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(logo_pixmap)

        dev_info_layout.addWidget(logo_label)
        dev_info_layout.addWidget(dev_label)
        

        # Overlay the developer info at bottom-right
        overlay_layout = QVBoxLayout()
        overlay_layout.addStretch()
        overlay_layout.addWidget(dev_info_widget)
        layout.addLayout(overlay_layout)
        
        layout.addStretch()
        return panel
    
    def setup_styles(self):
        """Setup application styles"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #34495e;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3498db;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #2c3e50;
                color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QLineEdit, QSpinBox, QListWidget, QTextEdit {
                background-color: #ecf0f1;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 5px;
            }
            QLabel {
                color: white;
            }
            QListWidget::item:selected {
                background-color: #3498db;
            }
        """)
    
    def browse_images_folder(self):
        """Browse for images folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Images Folder")
        if folder:
            self.images_folder = folder
            self.images_folder_edit.setText(folder)
            self.load_image_files()
            
            # Auto-set output folder if not set
            if not self.output_folder:
                self.output_folder = os.path.join(folder, "annotations")
                self.output_folder_edit.setText(self.output_folder)
            
            # Try to load classes.txt if exists
            classes_file = os.path.join(folder, "classes.txt")
            if os.path.exists(classes_file):
                self.load_classes_from_file(classes_file)
    
    def browse_output_folder(self):
        """Browse for output folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = folder
            self.output_folder_edit.setText(folder)
    
    def load_image_files(self):
        """Load all image files from the selected folder"""
        if not self.images_folder:
            return
            
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        self.image_files = []
        
        for file in os.listdir(self.images_folder):
            if any(file.lower().endswith(ext) for ext in extensions):
                self.image_files.append(file)
        
        self.image_files.sort()
        self.current_image_index = 0
        
        if self.image_files:
            self.load_current_image()
        
        self.update_ui()
    
    def load_current_image(self):
        """Load the current image and its annotations"""
        if not self.image_files or self.current_image_index >= len(self.image_files):
            return
            
        image_file = self.image_files[self.current_image_index]
        image_path = os.path.join(self.images_folder, image_file)
        
        if self.image_viewer.set_image(image_path):
            # Load existing annotations
            self.load_annotations_for_current_image()
            self.update_image_viewer()
            self.update_annotations_display()
        
        self.update_ui()
    
    def load_annotations_for_current_image(self):
        """Load annotations for the current image"""
        if not self.image_files or not self.output_folder:
            return
            
        image_file = self.image_files[self.current_image_index]
        txt_file = os.path.splitext(image_file)[0] + ".txt"
        txt_path = os.path.join(self.output_folder, txt_file)
        
        self.image_viewer.annotations.clear()
        
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) == 5:
                                cls_id = int(parts[0])
                                x, y, w, h = map(float, parts[1:])
                                self.image_viewer.annotations.append((cls_id, x, y, w, h))
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error loading annotations: {str(e)}")
    
    def save_current_annotations(self):
        """Save annotations for the current image"""
        if not self.image_files or not self.output_folder:
            return
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        
        image_file = self.image_files[self.current_image_index]
        txt_file = os.path.splitext(image_file)[0] + ".txt"
        txt_path = os.path.join(self.output_folder, txt_file)
        
        try:
            with open(txt_path, 'w') as f:
                for cls_id, x, y, w, h in self.image_viewer.annotations:
                    f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error saving annotations: {str(e)}")
    
    def add_class(self):
        """Add a new class"""
        from PyQt5.QtWidgets import QInputDialog
        
        name, ok = QInputDialog.getText(self, "Add Class", "Enter class name:")
        if ok and name.strip():
            name = name.strip()
            if name not in self.classes:
                self.classes.append(name)
                self.update_classes_display()
                self.generate_class_colors()
            else:
                QMessageBox.warning(self, "Warning", "Class already exists!")
    
    def remove_class(self):
        """Remove selected class"""
        current_row = self.classes_list.currentRow()
        if current_row >= 0:
            del self.classes[current_row]
            self.update_classes_display()
            self.generate_class_colors()
    
    def load_classes(self):
        """Load classes from file"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Classes", "", "Text Files (*.txt)")
        if file_path:
            self.load_classes_from_file(file_path)
    
    def load_classes_from_file(self, file_path):
        """Load classes from a specific file"""
        try:
            with open(file_path, 'r') as f:
                self.classes = [line.strip() for line in f if line.strip()]
            self.update_classes_display()
            self.generate_class_colors()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error loading classes: {str(e)}")
    
    def save_classes(self):
        """Save classes to file"""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Classes", "classes.txt", "Text Files (*.txt)")
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    for cls in self.classes:
                        f.write(cls + '\n')
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error saving classes: {str(e)}")
    
    def update_classes_display(self):
        """Update the classes list widget"""
        self.classes_list.clear()
        for cls in self.classes:
            self.classes_list.addItem(cls)
        
        if self.classes:
            self.classes_list.setCurrentRow(0)
        
        self.update_image_viewer()
    
    def generate_class_colors(self):
        """Generate random colors for each class"""
        self.image_viewer.class_colors.clear()
        for cls in self.classes:
            color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            self.image_viewer.class_colors[cls] = color
        
        self.image_viewer.class_names = self.classes.copy()
    
    def on_class_selected(self, row):
        """Handle class selection"""
        if 0 <= row < len(self.classes):
            self.image_viewer.current_class = row
            self.current_class_label.setText(f"Current: {self.classes[row]}")
            
            # Update label color
            if self.classes[row] in self.image_viewer.class_colors:
                color = self.image_viewer.class_colors[self.classes[row]]
                self.current_class_label.setStyleSheet(
                    f"font-size: 14px; font-weight: bold; padding: 10px; "
                    f"background-color: rgb({color[2]}, {color[1]}, {color[0]}); "
                    f"color: white; border-radius: 5px;"
                )
    
    def cycle_class(self, direction):
        """Cycle through classes (direction: 1 for next, -1 for previous)"""
        if not self.classes:
            return
            
        current_row = self.classes_list.currentRow()
        if current_row < 0:
            current_row = 0
            
        new_row = (current_row + direction) % len(self.classes)
        self.classes_list.setCurrentRow(new_row)
    
    def update_thickness(self):
        """Update bounding box thickness"""
        self.image_viewer.box_thickness = self.thickness_spin.value()
        self.image_viewer.update_display()
    
    def update_min_size(self):
        """Update minimum box size"""
        self.image_viewer.min_box_size = self.min_size_spin.value()
    
    def update_image_viewer(self):
        """Update image viewer with current settings"""
        self.image_viewer.class_names = self.classes.copy()
        self.image_viewer.update_display()
    
    def update_annotations_display(self):
        """Update the annotations list display"""
        self.annotations_list.clear()
        
        for i, (cls_id, x, y, w, h) in enumerate(self.image_viewer.annotations):
            if cls_id < len(self.classes):
                class_name = self.classes[cls_id]
                self.annotations_list.addItem(f"{i+1}. {class_name} ({x:.3f}, {y:.3f}, {w:.3f}, {h:.3f})")
        
        self.annotations_count_label.setText(f"Annotations: {len(self.image_viewer.annotations)}")
    
    def update_ui(self):
        """Update all UI elements"""
        # Image counter
        if self.image_files:
            self.image_counter_label.setText(f"{self.current_image_index + 1} / {len(self.image_files)}")
            
            # Image info
            image_file = self.image_files[self.current_image_index]
            self.image_name_label.setText(f"File: {image_file}")
            
            if self.image_viewer.image is not None:
                h, w = self.image_viewer.image.shape[:2]
                self.image_size_label.setText(f"Size: {w} × {h}")
            
            # Progress
            progress = int((self.current_image_index + 1) / len(self.image_files) * 100)
            self.progress_label.setText(f"Progress: {progress}%")
        else:
            self.image_counter_label.setText("0 / 0")
            self.image_name_label.setText("No image selected")
            self.image_size_label.setText("Size: -")
            self.progress_label.setText("Progress: 0%")
        
        # Total annotations count
        total_annotations = sum(len(self.get_annotations_for_image(i)) for i in range(len(self.image_files)))
        self.stats_label.setText(f"Total Annotations: {total_annotations}")
        
        # Update annotations display
        self.update_annotations_display()
    
    def get_annotations_for_image(self, index):
        """Get annotations for a specific image index"""
        if not self.image_files or not self.output_folder or index >= len(self.image_files):
            return []
            
        image_file = self.image_files[index]
        txt_file = os.path.splitext(image_file)[0] + ".txt"
        txt_path = os.path.join(self.output_folder, txt_file)
        
        annotations = []
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) == 5:
                                annotations.append(tuple(map(float, parts)))
            except:
                pass
        
        return annotations
    
    def previous_image(self):
        """Go to previous image"""
        if self.image_files:
            self.save_current_annotations()  # Auto-save before switching
            self.current_image_index = (self.current_image_index - 1) % len(self.image_files)
            self.load_current_image()
    
    def next_image(self):
        """Go to next image"""
        if self.image_files:
            self.save_current_annotations()  # Auto-save before switching
            self.current_image_index = (self.current_image_index + 1) % len(self.image_files)
            self.load_current_image()
    
    def clear_all_annotations(self):
        """Clear all annotations for current image"""
        reply = QMessageBox.question(self, "Confirm", "Clear all annotations for this image?",
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.image_viewer.annotations.clear()
            self.image_viewer.update_display()
            self.update_annotations_display()

    def show_developer_info(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("About")
        msg.setIcon(QMessageBox.Information)
        msg.setTextFormat(Qt.RichText)
        msg.setStyleSheet("QLabel{min-width:300px;}")
        msg.setText("""
            <h3 style="color:#3498db;">Jagdish Lamba</h3>
            <p style="color:#3498db;"><b>Date:</b> 09 Jun 2025</p>
            <p style="color:#3498db;"><b>Version:</b> 1.0</p>
        """)
        msg.exec_()


    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        key = event.key()
        
        if key == Qt.Key_A:  # Previous image
            self.previous_image()
        elif key == Qt.Key_D:  # Next image
            self.next_image()
        elif key == Qt.Key_W:  # Previous class
            self.cycle_class(-1)
        elif key == Qt.Key_S:  # Next class
            self.cycle_class(1)
        elif key == Qt.Key_Delete:  # Clear all annotations
            self.clear_all_annotations()
        elif key == Qt.Key_J and event.modifiers() == Qt.AltModifier:
            self.show_developer_info()
        
        super().keyPressEvent(event)
    
    def closeEvent(self, event):
        """Handle application close"""
        # Save current annotations before closing
        if self.image_files:
            self.save_current_annotations()
        
        # Save settings
        self.save_settings()
        
        event.accept()
    
    def save_settings(self):
        """Save application settings"""
        settings = {
            'images_folder': self.images_folder,
            'output_folder': self.output_folder,
            'classes': self.classes,
            'box_thickness': self.thickness_spin.value(),
            'min_box_size': self.min_size_spin.value(),
            'current_image_index': self.current_image_index
        }
        
        try:
            settings_file = os.path.join(os.path.expanduser('~'), '.yolo_annotator_settings.json')
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
        except:
            pass  # Ignore save errors
    
    def load_settings(self):
        """Load application settings"""
        try:
            settings_file = os.path.join(os.path.expanduser('~'), '.yolo_annotator_settings.json')
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                
                # Restore settings
                if 'images_folder' in settings and os.path.exists(settings['images_folder']):
                    self.images_folder = settings['images_folder']
                    self.images_folder_edit.setText(self.images_folder)
                    self.load_image_files()
                
                if 'output_folder' in settings:
                    self.output_folder = settings['output_folder']
                    self.output_folder_edit.setText(self.output_folder)
                
                if 'classes' in settings:
                    self.classes = settings['classes']
                    self.update_classes_display()
                    self.generate_class_colors()
                
                if 'box_thickness' in settings:
                    self.thickness_spin.setValue(settings['box_thickness'])
                
                if 'min_box_size' in settings:
                    self.min_size_spin.setValue(settings['min_box_size'])
                
                if 'current_image_index' in settings and self.image_files:
                    index = settings['current_image_index']
                    if 0 <= index < len(self.image_files):
                        self.current_image_index = index
                        self.load_current_image()
        except:
            pass  # Ignore load errors

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("YOLO Annotatotr Tool")
    app.setApplicationVersion("1.0")
    app.setWindowIcon(QIcon("logo.ico")) 
    
    # Create and show main window
    window = YOLOAnnotationTool()
    window.show()
    
    # Run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()