import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QLabel, QFileDialog, QHBoxLayout, 
                           QFrame, QSpacerItem, QSizePolicy, QComboBox, QGridLayout)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QColor, QPalette
from PyQt5.QtCore import Qt
from recognition import FaceRecognizer

class StyledButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setMinimumHeight(45)
        self.setFont(QFont('Arial', 11, QFont.Bold))
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                background: #3498DB;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                margin: 5px;
            }
            QPushButton:hover {
                background: #2980B9;
            }
            QPushButton:pressed {
                background: #2472A4;
            }
        """)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Detection and Recognition")
        self.setGeometry(100, 100, 1200, 800)
        self.current_image = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize recognizers
        self.recognizer_color = FaceRecognizer('Processed/train/color', 'Processed/test/color')
        self.recognizer_color.fit()
        self.recognizer_gray = FaceRecognizer('Processed/train/grayscale', 'Processed/test/grayscale')
        self.recognizer_gray.fit()
        
        self.init_ui()

    def init_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QGridLayout(main_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Set application style with a simpler color scheme
        self.setStyleSheet("""
            QMainWindow {
                background: #2C3E50;
            }
            QLabel {
                color: white;
            }
            QFrame {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
        """)

        # Left panel for controls
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(20)
        left_layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title_label = QLabel("Face Analysis")
        title_label.setFont(QFont('Arial', 28, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                background: #34495E;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        left_layout.addWidget(title_label)

        # Face Detection Section
        detection_frame = QFrame()
        detection_layout = QVBoxLayout(detection_frame)
        
        detection_title = QLabel("Face Detection")
        detection_title.setFont(QFont('Arial', 16, QFont.Bold))
        detection_title.setStyleSheet("""
            QLabel {
                color: white;
                background: #34495E;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        detection_layout.addWidget(detection_title)
        
        self.detect_button = StyledButton("Detect Faces")
        self.detect_button.clicked.connect(self.detect_faces)
        detection_layout.addWidget(self.detect_button)
        
        left_layout.addWidget(detection_frame)
        left_layout.addStretch()

        # Face Recognition Section
        recognition_frame = QFrame()
        recognition_layout = QVBoxLayout(recognition_frame)
        
        recognition_title = QLabel("Face Recognition")
        recognition_title.setFont(QFont('Arial', 16, QFont.Bold))
        recognition_title.setStyleSheet("""
            QLabel {
                color: white;
                background: #34495E;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        recognition_layout.addWidget(recognition_title)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Color", "Grayscale"])
        self.mode_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.1);
                color: white;
                font-size: 14px;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox QAbstractItemView {
                background: #34495E;
                color: white;
                selection-background-color: #2C3E50;
                border-radius: 8px;
            }
        """)
        recognition_layout.addWidget(self.mode_combo)
        
        self.recognize_button = StyledButton("Recognize Face")
        self.recognize_button.clicked.connect(self.recognize_face)
        recognition_layout.addWidget(self.recognize_button)
        
        self.roc_button = StyledButton("Show ROC Curve")
        self.roc_button.clicked.connect(self.show_roc)
        recognition_layout.addWidget(self.roc_button)
        
        left_layout.addWidget(recognition_frame)
        left_layout.addStretch()
        
        # Add left panel to grid layout
        main_layout.addWidget(left_panel, 0, 0, 2, 1)

        # Right panel for image display
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(20)
        right_layout.setContentsMargins(20, 20, 20, 20)

        # Image display area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("""
            QLabel {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 12px;
            }
        """)
        right_layout.addWidget(self.image_label)

        # Result label
        self.result_label = QLabel("")
        self.result_label.setFont(QFont('Arial', 18))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("""
            QLabel {
                color: white;
                background: #34495E;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        right_layout.addWidget(self.result_label)

        # Image control buttons
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setSpacing(20)

        self.upload_button = StyledButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)
        button_layout.addWidget(self.upload_button)

        self.reset_button = StyledButton("Reset")
        self.reset_button.clicked.connect(self.reset_image)
        button_layout.addWidget(self.reset_button)

        right_layout.addWidget(button_container)
        
        # Add right panel to grid layout
        main_layout.addWidget(right_panel, 0, 1, 2, 1)

        # Set column stretch factors
        main_layout.setColumnStretch(0, 1)  # Left panel
        main_layout.setColumnStretch(1, 2)  # Right panel

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", 
                                                 "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            self.current_image = cv2.imread(file_name)
            self.display_image(self.current_image)
            self.result_label.setText("")

    def display_image(self, image):
        if image is not None:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.image_label.size(), 
                                        Qt.KeepAspectRatio, 
                                        Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.setStyleSheet("""
                QLabel {
                    background: transparent;
                    border: none;
                }
            """)

    def detect_faces(self):
        if self.current_image is not None:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4,minSize=(100,100))
            image_with_faces = self.current_image.copy() 
            for (x, y, w, h) in faces:
                cv2.rectangle(image_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 2)
            self.display_image(image_with_faces)
            
    def recognize_face(self):
        if self.current_image is not None:
            temp_path = "temp_recognition.jpg"
            cv2.imwrite(temp_path, self.current_image)
            mode = self.mode_combo.currentText()
            if mode == "Color":
                name, dist = self.recognizer_color.predict(temp_path)
            else:
                name, dist = self.recognizer_gray.predict(temp_path)
            self.result_label.setText(f"Recognized as: {name} (distance: {dist:.2f})")

    def show_roc(self):
        mode = self.mode_combo.currentText()
        if mode == "Color":
            self.recognizer_color.evaluate()
        else:
            self.recognizer_gray.evaluate()

    def reset_image(self):
        self.current_image = None
        self.image_label.clear()
        self.image_label.setStyleSheet("""
            QLabel {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 12px;
            }
        """)
        self.result_label.setText("")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
