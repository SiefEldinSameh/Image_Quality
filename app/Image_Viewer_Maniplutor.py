import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QMessageBox

import matplotlib.pyplot as plt

import traceback
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QPushButton, QLabel, QFileDialog,
                            QComboBox, QSlider, QGroupBox, QDialog, QDialogButtonBox,
                            QRadioButton, QButtonGroup, QSizePolicy)  # Added QSizePolicy
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap,QPainter, QPen, QColor, QPainterPath ,QIcon


class InteractiveHistogram(QWidget):
    def __init__(self, parent=None, viewer=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setMinimumSize(400, 300)

        # Store the image and histogram data
        self.image = None
        self.original_image = None
        self.histogram = None
        self.control_points = [(0, 0), (255, 255)]
        self.dragging_point = None
        self.curve = None

        self.setMouseTracking(True)

    def set_image(self, image):
        # Only set original_image if it hasn't been set before or if it's a new image
        if self.original_image is None or not np.array_equal(image, self.original_image):
            self.original_image = image.copy()
            self.image = image.copy()
            self.control_points = [(0, 0), (255, 255)]
            self.curve = None
        else:
            # If we're updating an existing image, keep the current image
            self.image = image.copy()

        self.calculate_histogram()
        self.update()

    def calculate_histogram(self):
        if self.image is None:
            return

        if len(self.image.shape) == 3:
            self.histogram = []
            for i in range(3):
                hist = cv2.calcHist([self.image], [i], None, [256], [0, 256])
                self.histogram.append(hist / hist.max() * self.height())
        else:
            self.histogram = cv2.calcHist([self.image], [0], None, [256], [0, 256])
            self.histogram = self.histogram / self.histogram.max() * self.height()

    def update_image(self):
        if self.original_image is None:
            return

        # Create lookup table from control points
        x_points = np.array([p[0] for p in self.control_points])
        y_points = np.array([p[1] for p in self.control_points])

        # Interpolate between points
        curve = np.interp(np.arange(256), x_points, y_points)
        self.curve = curve

        # Apply curve to image using the original image as source
        if len(self.original_image.shape) == 3:
            channels = cv2.split(self.original_image)
            adjusted_channels = [cv2.LUT(channel, curve.astype('uint8')) for channel in channels]
            self.image = cv2.merge(adjusted_channels)
        else:
            self.image = cv2.LUT(self.original_image, curve.astype('uint8'))

        # Update the main viewer with the modified image
        if self.viewer:
            self.viewer.update_from_histogram(self.image.copy())

        # Recalculate histogram for the modified image
        self.calculate_histogram()
        self.update()  # Ensure the widget refreshes

    def paintEvent(self, event):
        if self.histogram is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw histogram(s)
        if isinstance(self.histogram, list):  # Color image
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR colors
            for hist, color in zip(self.histogram, colors):
                self.draw_histogram_curve(painter, hist, color)
        else:  # Grayscale image
            self.draw_histogram_curve(painter, self.histogram, (128, 128, 128))

        # Draw curve
        if self.curve is not None:
            painter.setPen(QPen(Qt.GlobalColor.blue, 2))
            for i in range(255):
                x1 = (i / 255.0) * self.width()
                y1 = self.height() - (self.curve[i] / 255.0) * self.height()
                x2 = ((i + 1) / 255.0) * self.width()
                y2 = self.height() - (self.curve[i + 1] / 255.0) * self.height()
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Draw control points
        painter.setPen(QPen(Qt.GlobalColor.red, 4))
        for point in self.control_points:
            x = (point[0] / 255.0) * self.width()
            y = self.height() - (point[1] / 255.0) * self.height()
            painter.drawPoint(int(x), int(y))

    def draw_histogram_curve(self, painter, histogram, color):
        path = QPainterPath()
        path.moveTo(0, self.height())

        for i in range(256):
            x = (i / 255.0) * self.width()
            y = self.height() - histogram[i]
            path.lineTo(x, y)

        path.lineTo(self.width(), self.height())

        # Fill histogram with semi-transparent color
        painter.fillPath(path, QColor(*color, 50))

        # Draw the outline
        painter.setPen(QPen(QColor(*color), 1))
        painter.drawPath(path)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            x = int((event.pos().x() / self.width()) * 255)
            y = int((1 - (event.pos().y() / self.height())) * 255)

            # Check if clicking near existing point
            for i, point in enumerate(self.control_points):
                if abs(point[0] - x) < 5 and abs(point[1] - y) < 5:
                    self.dragging_point = i
                    return

            # Add new point
            self.control_points.append((x, y))
            self.control_points.sort(key=lambda p: p[0])
            self.update_image()  # Ensure the image updates after adding a new point

    def mouseMoveEvent(self, event):
        if self.dragging_point is not None:
            x = int((event.pos().x() / self.width()) * 255)
            y = int((1 - (event.pos().y() / self.height())) * 255)

            if self.dragging_point in [0, len(self.control_points) - 1]:
                x = self.control_points[self.dragging_point][0]

            x = max(0, min(255, x))
            y = max(0, min(255, y))

            self.control_points[self.dragging_point] = (x, y)
            self.control_points.sort(key=lambda p: p[0])

            self.update_image()  # Ensure image updates during dragging

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.dragging_point is not None:
            self.dragging_point = None
            self.update_image()  # Apply final changes


class ZoomDialog(QDialog):
    def __init__(self, parent=None, source_type=None):
        super().__init__(parent)
        self.setWindowTitle("Zoom and FOV Controls")
        self.setModal(True)
        self.source_type = source_type

        layout = QVBoxLayout()


        # Zoom controls group
        zoom_group = QGroupBox("Zoom Controls")
        zoom_layout = QVBoxLayout()

        # Zoom slider
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 300)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setSingleStep(10)
        self.zoom_slider.setTickInterval(10)

        zoom_label = QLabel("Zoom Factor: 100%")
        self.zoom_slider.valueChanged.connect(
            lambda v: zoom_label.setText(f"Zoom Factor: {v}%"))

        zoom_layout.addWidget(self.zoom_slider)
        zoom_layout.addWidget(zoom_label)
        zoom_group.setLayout(zoom_layout)

        # FOV controls group
        fov_group = QGroupBox("Field of View Controls")
        fov_layout = QVBoxLayout()

        # FOV center position controls
        center_layout = QHBoxLayout()
        center_layout.addWidget(QLabel("Center X:"))
        self.center_x = QSlider(Qt.Orientation.Horizontal)
        self.center_x.setRange(0, 100)
        self.center_x.setValue(50)
        center_layout.addWidget(self.center_x)

        center_layout.addWidget(QLabel("Center Y:"))
        self.center_y = QSlider(Qt.Orientation.Horizontal)
        self.center_y.setRange(0, 100)
        self.center_y.setValue(50)
        center_layout.addWidget(self.center_y)

        # FOV size controls
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("FOV Width:"))
        self.fov_width = QSlider(Qt.Orientation.Horizontal)
        self.fov_width.setRange(10, 100)
        self.fov_width.setValue(100)
        size_layout.addWidget(self.fov_width)

        size_layout.addWidget(QLabel("FOV Height:"))
        self.fov_height = QSlider(Qt.Orientation.Horizontal)
        self.fov_height.setRange(10, 100)
        self.fov_height.setValue(100)
        size_layout.addWidget(self.fov_height)

        # Add value labels for FOV controls
        self.center_x_label = QLabel("50%")
        self.center_y_label = QLabel("50%")
        self.width_label = QLabel("100%")
        self.height_label = QLabel("100%")

        # Connect value changes to labels
        self.center_x.valueChanged.connect(lambda v: self.center_x_label.setText(f"{v}%"))
        self.center_y.valueChanged.connect(lambda v: self.center_y_label.setText(f"{v}%"))
        self.fov_width.valueChanged.connect(lambda v: self.width_label.setText(f"{v}%"))
        self.fov_height.valueChanged.connect(lambda v: self.height_label.setText(f"{v}%"))

        # Add all FOV controls to the layout
        fov_layout.addLayout(center_layout)
        fov_layout.addWidget(self.center_x_label)
        fov_layout.addWidget(self.center_y_label)
        fov_layout.addLayout(size_layout)
        fov_layout.addWidget(self.width_label)
        fov_layout.addWidget(self.height_label)
        fov_group.setLayout(fov_layout)

        # Interpolation method selector
        self.interp_combo = QComboBox()
        self.interp_combo.addItems([
            "Nearest Neighbor",
            "Linear",
            "Bilinear",
            "Cubic"
        ])

        # Output location selector
        output_group = QGroupBox("Output Location")
        output_layout = QVBoxLayout()

        self.location_group = QButtonGroup()

        # Always add "Replace Current" option
        replace_radio = QRadioButton("Replace Current Image")
        self.location_group.addButton(replace_radio, 0)
        output_layout.addWidget(replace_radio)

        # Add Output 1 and Output 2 options if not already there
        if source_type != "output1":
            output1_radio = QRadioButton("Output 1")
            self.location_group.addButton(output1_radio, 1)
            output_layout.addWidget(output1_radio)

        if source_type != "output2":
            output2_radio = QRadioButton("Output 2")
            self.location_group.addButton(output2_radio, 2)
            output_layout.addWidget(output2_radio)

        # Select first available option
        first_button = self.location_group.buttons()[0]
        first_button.setChecked(True)

        output_group.setLayout(output_layout)

        # Add all groups and widgets to main layout
        layout.addWidget(zoom_group)
        layout.addWidget(fov_group)
        layout.addWidget(QLabel("Interpolation Method:"))
        layout.addWidget(self.interp_combo)
        layout.addWidget(output_group)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def get_output_location(self):
        button_id = self.location_group.checkedId()
        if button_id == 0:
            return "replace"
        elif button_id == 1:
            return "output1"
        else:
            return "output2"

    def get_fov_settings(self):
        return {
            'center_x': self.center_x.value() / 100.0,
            'center_y': self.center_y.value() / 100.0,
            'width': self.fov_width.value() / 100.0,
            'height': self.fov_height.value() / 100.0
        }

class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Quality Viewer")
        self.setGeometry(100, 100, 1400, 800)
        app_icon = QIcon("../assets/logo.png")
        self.setWindowIcon(app_icon)

        # Image storage
        self.input_image = None
        self.output_image1 = None
        self.output_image2 = None
        self.roi_mode = None  # Mode for ROI drawing: "snr" or "cnr"

        # Store original images
        self.original_input = None
        self.original_output1 = None
        self.original_output2 = None

        # ROI management
        self.roi_start = None
        self.roi_circles = []
        self.selected_roi = None
        self.is_drawing = False
        self.is_moving = False
        self.current_histogram_source = None  # Track which port opened the histogram
        self.image_states = {
            'input': {'current': None, 'original': None},
            'output1': {'current': None, 'original': None},
            'output2': {'current': None, 'original': None}
        }

        self.init_ui()

    # Add the chain_process_to_output2 method here, before init_ui
    def chain_process_to_output2(self):
        if self.output_image1 is None:
            return  # Can't process if Output 1 is empty

        # Get Output 1 controls (use controls from Output 1 instead of Output 2)
        controls = self.output1_controls
        filter_combo = controls.findChild(QComboBox, "filter_1")
        contrast_combo = controls.findChild(QComboBox, "contrast_1")
        noise_combo = controls.findChild(QComboBox, "noise_1")

        # Get processing parameters from Output 1's controls
        filter_method = filter_combo.currentText()
        contrast_method = contrast_combo.currentText()
        noise_method = noise_combo.currentText()

        # Get slider values from Output 1's controls
        sliders = controls.findChild(QWidget, "contrast_sliders_1")
        alpha = sliders.findChild(QSlider, "alpha_1").value() / 100.0 if sliders else 1.0
        beta = sliders.findChild(QSlider, "beta_1").value() if sliders else 0
        gamma = sliders.findChild(QSlider, "gamma_1").value() / 100.0 if sliders else 1.0

        # Process the image using Output 1's settings
        if len(self.output_image1.shape) == 3:
            channels = cv2.split(self.output_image1)
            processed_channels = []

            for channel in channels:
                filtered = self.apply_filter(channel, filter_method)
                contrast_enhanced = self.apply_contrast(filtered, contrast_method, alpha, beta, gamma)
                processed = self.apply_noise(contrast_enhanced, noise_method)
                processed_channels.append(processed)

            result = cv2.merge(processed_channels)
        else:
            filtered = self.apply_filter(self.output_image1, filter_method)
            contrast_enhanced = self.apply_contrast(filtered, contrast_method, alpha, beta, gamma)
            result = self.apply_noise(contrast_enhanced, noise_method)

        # Update Output 2
        self.output_image2 = result
        self.original_output2 = result.copy()
        self.display_image(result, self.output2_label)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # Image display area
        display_widget = QWidget()
        display_layout = QHBoxLayout()  # Initialize display layout
        display_widget.setLayout(display_layout)

        # Create image labels with titles and zoom buttons
        self.input_label = QLabel()
        self.output1_label = QLabel()
        self.output2_label = QLabel()

        # Update the labels to use minimum sizes instead of fixed sizes
        for label in [self.input_label, self.output1_label, self.output2_label]:
            label.setMinimumSize(200, 200)
            label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            label.setStyleSheet("border: 1px solid black")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Connect mouse events for ROI drawing
        self.input_label.mousePressEvent = self.start_roi
        self.input_label.mouseMoveEvent = self.update_roi
        self.input_label.mouseReleaseEvent = self.finish_roi

        # Create image containers
        input_container = self.create_image_container("Input Image", self.input_label, "input")
        output1_container = self.create_image_container("Output 1", self.output1_label, "output1")
        output2_container = self.create_image_container("Output 2", self.output2_label, "output2")

        # Add histogram buttons to image containers
        histogram_btn_input = QPushButton("Show Histogram (Input)")
        histogram_btn_input.clicked.connect(lambda: self.show_histogram(self.input_image, "Input Image"))
        input_container.layout().addWidget(histogram_btn_input)

        histogram_btn_output1 = QPushButton("Show Histogram (Output 1)")
        histogram_btn_output1.clicked.connect(lambda: self.show_histogram(self.output_image1, "Output 1"))
        output1_container.layout().addWidget(histogram_btn_output1)

        histogram_btn_output2 = QPushButton("Show Histogram (Output 2)")
        histogram_btn_output2.clicked.connect(lambda: self.show_histogram(self.output_image2, "Output 2"))
        output2_container.layout().addWidget(histogram_btn_output2)

        # Add containers to display layout
        display_layout.addWidget(input_container)
        display_layout.addWidget(output1_container)
        display_layout.addWidget(output2_container)

        # Controls area
        controls_widget = QWidget()
        controls_layout = QHBoxLayout()
        controls_widget.setLayout(controls_layout)

        # File controls
        controls_layout.addWidget(self.create_file_controls())

        # ROI and Calculation controls
        self.roi_controls = self.create_roi_controls()
        controls_layout.addWidget(self.roi_controls)

        # Processing controls for Output 1
        self.output1_controls = self.create_processing_controls("Output 1 Controls")
        controls_layout.addWidget(self.output1_controls)

        # Processing controls for Output 2
        self.output2_controls = self.create_processing_controls("Output 2 Controls")
        controls_layout.addWidget(self.output2_controls)

        # Add widgets to main layout
        main_layout.addWidget(display_widget, stretch=2)
        main_layout.addWidget(controls_widget, stretch=1)

        # Set the window properties
        self.setWindowTitle("Image Quality Viewer")
        self.setGeometry(100, 100, 1400, 800)

        # Set size policies for the main widgets
        display_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        controls_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

    def create_roi_controls(self):
        group = QGroupBox("ROI and Calculation Controls")
        layout = QVBoxLayout()

        # Add toggle button for ROI section
        roi_toggle_btn = QPushButton("Show/Hide ROI Controls")
        roi_toggle_btn.setCheckable(True)
        roi_toggle_btn.setChecked(True)  # Start expanded
        roi_toggle_btn.toggled.connect(lambda checked: roi_controls_widget.setVisible(checked))
        layout.addWidget(roi_toggle_btn)

        # Collapsible content
        roi_controls_widget = QWidget()
        roi_controls_layout = QVBoxLayout()
        roi_controls_widget.setLayout(roi_controls_layout)

        draw_snr_btn = QPushButton("Draw SNR ROIs")
        draw_snr_btn.clicked.connect(lambda: self.set_roi_mode("snr"))
        roi_controls_layout.addWidget(draw_snr_btn)

        draw_cnr_btn = QPushButton("Draw CNR ROIs")
        draw_cnr_btn.clicked.connect(lambda: self.set_roi_mode("cnr"))
        roi_controls_layout.addWidget(draw_cnr_btn)

        calculate_snr_btn = QPushButton("Calculate SNR")
        calculate_snr_btn.clicked.connect(self.calculate_snr)
        roi_controls_layout.addWidget(calculate_snr_btn)

        calculate_cnr_btn = QPushButton("Calculate CNR")
        calculate_cnr_btn.clicked.connect(self.calculate_cnr)
        roi_controls_layout.addWidget(calculate_cnr_btn)

        self.result_label = QLabel("Results: SNR or CNR will be displayed here")
        roi_controls_layout.addWidget(self.result_label)

        layout.addWidget(roi_controls_widget)
        group.setLayout(layout)
        return group

    def set_roi_mode(self, mode):
        self.roi_mode = mode
        self.roi_circles.clear()
        self.update_image_display()
        self.result_label.setText(f"Mode set to '{mode.upper()}'. Start drawing ROIs.")

    def reset_rois(self):
        self.roi_circles.clear()
        self.update_image_display()
        self.result_label.setText("Results: CNR and SNR will be displayed here")

    def is_point_in_roi(self, x, y):
        """Check if a point is inside any ROI and return its index"""
        for i, (roi_x, roi_y, radius) in enumerate(self.roi_circles):
            distance = np.sqrt((x - roi_x) ** 2 + (y - roi_y) ** 2)
            if distance <= radius:
                return i
        return None

    def map_coordinates(self, event_pos, label, image):
        """Map mouse position from QLabel space to image space."""
        if image is None:
            return None

        label_width, label_height = label.width(), label.height()
        img_height, img_width = image.shape[:2]

        # Calculate scaling factor and offsets
        scale_width = label_width / img_width
        scale_height = label_height / img_height
        scale = min(scale_width, scale_height)

        # Calculate top-left corner offsets for centering the image in QLabel
        x_offset = (label_width - img_width * scale) / 2
        y_offset = (label_height - img_height * scale) / 2

        # Map QLabel position to image position
        x = (event_pos.x() - x_offset) / scale
        y = (event_pos.y() - y_offset) / scale

        # Ensure coordinates are within image bounds
        x = max(0, min(int(x), img_width - 1))
        y = max(0, min(int(y), img_height - 1))

        return x, y

    def start_roi(self, event):
        if self.roi_mode is None:
            self.result_label.setText("Please select 'Draw SNR ROIs' or 'Draw CNR ROIs' before drawing.")
            return

        max_rois = 2 if self.roi_mode == "snr" else 3
        if len(self.roi_circles) >= max_rois:
            self.result_label.setText(f"Maximum of {max_rois} ROIs allowed for '{self.roi_mode.upper()}' mode.")
            return

        mapped_coords = self.map_coordinates(event.pos(), self.input_label, self.input_image)
        if mapped_coords:
            self.roi_start = mapped_coords
            self.is_drawing = True

    def finish_roi(self, event):
        if self.is_drawing and self.roi_start is not None:
            max_rois = 2 if self.roi_mode == "snr" else 3
            if len(self.roi_circles) >= max_rois:
                self.result_label.setText(f"Maximum of {max_rois} ROIs allowed for '{self.roi_mode.upper()}' mode.")
                return

            mapped_coords = self.map_coordinates(event.pos(), self.input_label, self.input_image)
            if mapped_coords:
                start_x, start_y = self.roi_start

                # Enforce the radius of the first ROI for subsequent ROIs
                if len(self.roi_circles) == 0:
                    radius = int(np.sqrt((mapped_coords[0] - start_x) ** 2 + (mapped_coords[1] - start_y) ** 2))
                else:
                    radius = self.roi_circles[0][2]

                # Save the ROI
                self.roi_circles.append((int(start_x), int(start_y), radius))
                self.roi_start = None
                self.is_drawing = False
                self.update_image_display()

    def update_roi(self, event):
        if self.is_drawing and self.roi_start is not None:
            mapped_coords = self.map_coordinates(event.pos(), self.input_label, self.input_image)
            if mapped_coords:
                start_x, start_y = self.roi_start

                # Enforce the radius of the first ROI for temporary display
                if len(self.roi_circles) == 0:
                    radius = int(np.sqrt((mapped_coords[0] - start_x) ** 2 + (mapped_coords[1] - start_y) ** 2))
                else:
                    radius = self.roi_circles[0][2]

                # Show a temporary ROI
                temp_circles = self.roi_circles + [(int(start_x), int(start_y), radius)]
                self.update_image_display(temp_circles)

    def update_image_display(self, temp_circles=None):
        if self.input_image is not None:
            image = self.input_image.copy()
            circles = temp_circles if temp_circles else self.roi_circles

            for (x, y, radius) in circles:
                cv2.circle(image, (x, y), radius, (255, 255, 255), 2)

            self.display_image(image, self.input_label)

    def calculate_snr(self):
        if self.roi_mode != "snr":
            self.result_label.setText("Error: You are not in SNR mode. Switch to SNR mode to calculate SNR.")
            return

        if len(self.roi_circles) != 2:
            self.result_label.setText("Please draw exactly 2 ROIs (first = signal, second = background) in SNR mode.")
            return

        (x1, y1, r1), (x2, y2, r2) = self.roi_circles
        images = [self.input_image, self.output_image1, self.output_image2]
        image_labels = ["Input", "Output 1", "Output 2"]
        results = []

        for image, label in zip(images, image_labels):
            if image is None:
                results.append(f"{label}: N/A (Image not available)")
                continue

            y, x = np.ogrid[:image.shape[0], :image.shape[1]]
            signal_mask = (x - x1) ** 2 + (y - y1) ** 2 <= r1 ** 2
            background_mask = (x - x2) ** 2 + (y - y2) ** 2 <= r2 ** 2

            signal_values = image[signal_mask]
            background_values = image[background_mask]

            if signal_values.size == 0 or background_values.size == 0:
                results.append(f"{label}: Error (Empty ROI)")
                continue

            signal_mean = np.mean(signal_values)
            background_std = np.std(background_values)

            if background_std == 0:
                results.append(f"{label}: Error (Background has no variation)")
                continue

            snr = 10 * np.log10(signal_mean / background_std) if signal_mean > 0 else float('-inf')
            results.append(f"{label}: SNR = {snr:.2f} dB")

        self.result_label.setText("SNR Results:\n" + "\n".join(results))

    def calculate_cnr(self):
        if self.roi_mode != "cnr":
            self.result_label.setText("Error: You are not in CNR mode. Switch to CNR mode to calculate CNR.")
            return

        if len(self.roi_circles) != 3:
            self.result_label.setText("Please draw exactly 3 ROIs (two signals and one background) in CNR mode.")
            return

        (x1, y1, r1), (x2, y2, r2), (x3, y3, r3) = self.roi_circles
        images = [self.input_image, self.output_image1, self.output_image2]
        image_labels = ["Input", "Output 1", "Output 2"]
        results = []

        for image, label in zip(images, image_labels):
            if image is None:
                results.append(f"{label}: N/A (Image not available)")
                continue

            y, x = np.ogrid[:image.shape[0], :image.shape[1]]
            signal1_mask = (x - x1) ** 2 + (y - y1) ** 2 <= r1 ** 2
            signal2_mask = (x - x2) ** 2 + (y - y2) ** 2 <= r2 ** 2
            background_mask = (x - x3) ** 2 + (y - y3) ** 2 <= r3 ** 2

            signal1_values = image[signal1_mask]
            signal2_values = image[signal2_mask]
            background_values = image[background_mask]

            if signal1_values.size == 0 or signal2_values.size == 0 or background_values.size == 0:
                results.append(f"{label}: Error (Empty ROI)")
                continue

            signal1_mean = np.mean(signal1_values)
            signal2_mean = np.mean(signal2_values)
            background_std = np.std(background_values)

            if background_std == 0:
                results.append(f"{label}: Error (Background has no variation)")
                continue

            cnr = abs(signal1_mean - signal2_mean) / background_std
            results.append(f"{label}: CNR = {cnr:.2f}")

        self.result_label.setText("CNR Results:\n" + "\n".join(results))

    def create_image_container(self, title, label, image_type):
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)

        # Title
        title_label = QLabel(title)
        layout.addWidget(title_label)

        # Add the existing label
        layout.addWidget(label)

        # Zoom button
        zoom_btn = QPushButton("Zoom")
        zoom_btn.clicked.connect(lambda: self.show_zoom_dialog(image_type))
        layout.addWidget(zoom_btn)

        return container

    def display_image(self, image, label):
        if image is not None:
            # Calculate the aspect ratio of the image
            if len(image.shape) == 3:
                height, width, channel = image.shape
                bytes_per_line = 3 * width
                q_image = QImage(image.data, width, height, bytes_per_line,
                                 QImage.Format.Format_RGB888).rgbSwapped()
            else:  # Grayscale image
                height, width = image.shape
                q_image = QImage(image.data, width, height, width,
                                 QImage.Format.Format_Grayscale8)

            # Get the size of the label
            label_size = label.size()

            # Calculate scaling factors
            w_scale = label_size.width() / width
            h_scale = label_size.height() / height
            scale = min(w_scale, h_scale)  # Use the smaller scale to fit the image

            # Calculate new dimensions while maintaining aspect ratio
            new_width = int(width * scale)
            new_height = int(height * scale)

            # Create pixmap and scale it
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(new_width, new_height,
                                          Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)

            label.setPixmap(scaled_pixmap)

    def create_zoomed_image(self, image, zoom_factor, interpolation, fov_settings=None):
        if len(image.shape) == 3:
            height, width, _ = image.shape
        else:
            height, width = image.shape

        if fov_settings:
            # Calculate FOV boundaries
            center_x = int(width * fov_settings['center_x'])
            center_y = int(height * fov_settings['center_y'])
            fov_width = int(width * fov_settings['width'])
            fov_height = int(height * fov_settings['height'])

            # Calculate crop boundaries
            x1 = max(0, center_x - fov_width // 2)
            x2 = min(width, center_x + fov_width // 2)
            y1 = max(0, center_y - fov_height // 2)
            y2 = min(height, center_y + fov_height // 2)

            # Crop the image to FOV
            image = image[y1:y2, x1:x2]

        # Apply zoom
        new_width = max(1, int(width * zoom_factor))
        new_height = max(1, int(height * zoom_factor))

        return cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    def show_zoom_dialog(self, image_type):
        if ((image_type == "input" and self.input_image is None) or
                (image_type == "output1" and self.output_image1 is None) or
                (image_type == "output2" and self.output_image2 is None)):
            return

        dialog = ZoomDialog(self, image_type)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Get zoom settings
            zoom_factor = dialog.zoom_slider.value() / 100.0
            interpolation = self.get_interpolation_method(dialog.interp_combo.currentText())
            output_location = dialog.get_output_location()
            fov_settings = dialog.get_fov_settings()

            # Get source image
            if image_type == "input":
                source_image = self.input_image
            elif image_type == "output1":
                source_image = self.output_image1
            else:
                source_image = self.output_image2

            # Apply zoom and FOV
            zoomed_image = self.create_zoomed_image(
                source_image,
                zoom_factor,
                interpolation,
                fov_settings
            )

            # Update appropriate output
            if output_location == "replace":
                if image_type == "input":
                    self.input_image = zoomed_image
                    self.display_image(zoomed_image, self.input_label)
                elif image_type == "output1":
                    self.output_image1 = zoomed_image
                    self.display_image(zoomed_image, self.output1_label)
                else:
                    self.output_image2 = zoomed_image
                    self.display_image(zoomed_image, self.output2_label)
            elif output_location == "output1":
                self.output_image1 = zoomed_image
                self.display_image(zoomed_image, self.output1_label)
            else:  # output2
                self.output_image2 = zoomed_image
                self.display_image(zoomed_image, self.output2_label)

    def zoom_image(self, image, label, zoom_factor, interpolation):
        if len(image.shape) == 3:
            height, width, _ = image.shape
        else:
            height, width = image.shape

        new_width = max(1, int(width * zoom_factor))
        new_height = max(1, int(height * zoom_factor))

        resized = cv2.resize(image, (new_width, new_height),
                             interpolation=interpolation)
        self.display_image(resized, label)


    def create_file_controls(self):
        group = QGroupBox("File Controls")
        layout = QVBoxLayout()

        open_btn = QPushButton("Open Image")
        open_btn.clicked.connect(self.open_image)
        layout.addWidget(open_btn)

        reset_all_btn = QPushButton("Reset All")
        reset_all_btn.clicked.connect(self.reset_all)
        layout.addWidget(reset_all_btn)

        group.setLayout(layout)
        return group

    def create_processing_controls(self, title):
        group = QGroupBox(title)
        layout = QVBoxLayout()

        # Add toggle button for Processing Controls section
        toggle_btn = QPushButton(f"Show/Hide {title}")
        toggle_btn.setCheckable(True)
        toggle_btn.setChecked(True)  # Start expanded
        toggle_btn.toggled.connect(lambda checked: controls_widget.setVisible(checked))
        layout.addWidget(toggle_btn)

        # Collapsible content
        controls_widget = QWidget()
        controls_layout = QVBoxLayout()
        controls_widget.setLayout(controls_layout)

        target = 1 if "1" in title else 2

        # Filter selector
        filter_combo = QComboBox()
        filter_combo.setObjectName(f"filter_{target}")
        filter_combo.addItems([
            "No Filter",
            "Gaussian Blur",
            "Median Filter",
            "Lowpass Filter",
            "Highpass Filter",
            "Bilateral Filter",
            "Sobel Filter",
            "Laplacian Filter",
            "Unsharp Masking",
            "Motion Blur",
            "Emboss Filter",
            "Prewitt Filter",
            "Mean Filter",
            "Box Filter",
            "Edge Enhance"
        ])
        controls_layout.addWidget(QLabel("Filter Type:"))
        controls_layout.addWidget(filter_combo)

        # Contrast enhancement selector
        contrast_combo = QComboBox()
        contrast_combo.setObjectName(f"contrast_{target}")
        contrast_combo.addItems([
            "No Contrast Enhancement",
            "Linear Contrast Stretch",
            "Histogram Equalization",
            "CLAHE",
            "Gamma Correction"
        ])
        controls_layout.addWidget(QLabel("Contrast Enhancement:"))
        controls_layout.addWidget(contrast_combo)

        # Contrast sliders container
        contrast_sliders = QWidget()
        contrast_sliders.setObjectName(f"contrast_sliders_{target}")
        contrast_layout = QVBoxLayout()
        contrast_sliders.setLayout(contrast_layout)

        # Alpha (contrast) slider
        alpha_slider = QSlider(Qt.Orientation.Horizontal)
        alpha_slider.setObjectName(f"alpha_{target}")
        alpha_slider.setRange(10, 300)  # 0.1 to 3.0
        alpha_slider.setValue(100)  # Default 1.0
        contrast_layout.addWidget(QLabel("Contrast:"))
        contrast_layout.addWidget(alpha_slider)

        # Beta (brightness) slider
        beta_slider = QSlider(Qt.Orientation.Horizontal)
        beta_slider.setObjectName(f"beta_{target}")
        beta_slider.setRange(-100, 100)
        beta_slider.setValue(0)
        contrast_layout.addWidget(QLabel("Brightness:"))
        contrast_layout.addWidget(beta_slider)

        # Gamma slider
        gamma_slider = QSlider(Qt.Orientation.Horizontal)
        gamma_slider.setObjectName(f"gamma_{target}")
        gamma_slider.setRange(1, 500)  # 0.01 to 5.0
        gamma_slider.setValue(100)  # Default 1.0
        contrast_layout.addWidget(QLabel("Gamma:"))
        contrast_layout.addWidget(gamma_slider)

        contrast_sliders.hide()
        controls_layout.addWidget(contrast_sliders)

        # Show/hide relevant sliders based on contrast method selection
        contrast_combo.currentTextChanged.connect(
            lambda text: self.update_contrast_controls(text, target))

        # Noise selector
        noise_combo = QComboBox()
        noise_combo.setObjectName(f"noise_{target}")
        noise_combo.addItems([
            "No Noise",
            "Gaussian Noise",
            "Salt & Pepper Noise",
            "Speckle Noise",
            "Poisson Noise",
            "Uniform Noise",
            "Rayleigh Noise",
            "Exponential Noise",
            "Gamma Noise",
            "Periodic Noise",
            "Multiplicative Noise",
            "Localvar Noise",
            "Mixed Noise",
            "Quantization Noise",
            "Random Lines Noise"
        ])
        controls_layout.addWidget(QLabel("Noise Type:"))
        controls_layout.addWidget(noise_combo)

        # Process button
        process_btn = QPushButton(f"Process to Output {target}")
        process_btn.clicked.connect(lambda: self.apply_processing(target))
        controls_layout.addWidget(process_btn)

        # Add chain processing button for Output 1 controls only
        if target == 1:
            chain_process_btn = QPushButton("Process Output 1 to Output 2")
            chain_process_btn.clicked.connect(self.chain_process_to_output2)
            controls_layout.addWidget(chain_process_btn)

        # Reset button for this output
        reset_btn = QPushButton(f"Reset Output {target}")
        reset_btn.clicked.connect(lambda: self.reset_output(target))
        controls_layout.addWidget(reset_btn)

        layout.addWidget(controls_widget)
        group.setLayout(layout)
        return group

    def apply_processing(self, target):
        if target == 1:
            source_image = self.input_image
            controls = self.output1_controls
            output_label = self.output1_label
        else:  # target == 2
            source_image = self.output_image1
            controls = self.output2_controls
            output_label = self.output2_label

        if source_image is None:
            return

        filter_combo = controls.findChild(QComboBox, f"filter_{target}")
        contrast_combo = controls.findChild(QComboBox, f"contrast_{target}")
        noise_combo = controls.findChild(QComboBox, f"noise_{target}")

        filter_method = filter_combo.currentText()
        contrast_method = contrast_combo.currentText()
        noise_method = noise_combo.currentText()

        # Get slider values if needed
        sliders = controls.findChild(QWidget, f"contrast_sliders_{target}")
        alpha = sliders.findChild(QSlider, f"alpha_{target}").value() / 100.0 if sliders else 1.0
        beta = sliders.findChild(QSlider, f"beta_{target}").value() if sliders else 0
        gamma = sliders.findChild(QSlider, f"gamma_{target}").value() / 100.0 if sliders else 1.0

        # Handle color images
        if len(source_image.shape) == 3:
            channels = cv2.split(source_image)
            processed_channels = []

            for channel in channels:
                # Apply processing in sequence: filter -> contrast -> noise
                filtered = self.apply_filter(channel, filter_method)
                contrast_enhanced = self.apply_contrast(filtered, contrast_method, alpha, beta, gamma)
                processed = self.apply_noise(contrast_enhanced, noise_method)
                processed_channels.append(processed)

            result = cv2.merge(processed_channels)
        else:
            filtered = self.apply_filter(source_image, filter_method)
            contrast_enhanced = self.apply_contrast(filtered, contrast_method, alpha, beta, gamma)
            result = self.apply_noise(contrast_enhanced, noise_method)

        if target == 1:
            self.image_states['output1']['current'] = result.copy()
            self.image_states['output1']['original'] = result.copy()
            self.output_image1 = result
            self.original_output1 = result.copy()
        else:
            self.image_states['output2']['current'] = result.copy()
            self.image_states['output2']['original'] = result.copy()
            self.output_image2 = result
            self.original_output2 = result.copy()

        self.display_image(result, self.output1_label if target == 1 else self.output2_label)

    def apply_filter(self, channel, method):
        if method == "No Filter":
            return channel
        elif method == "Gaussian Blur":
            return cv2.GaussianBlur(channel, (5, 5), 0)
        elif method == "Median Filter":
            return cv2.medianBlur(channel, 5)
        elif method == "Lowpass Filter":
            kernel = np.ones((5, 5), np.float32) / 25
            return cv2.filter2D(channel, -1, kernel)
        elif method == "Highpass Filter":
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            return cv2.filter2D(channel, -1, kernel)
        # New Filters
        elif method == "Bilateral Filter":
            return cv2.bilateralFilter(channel, 9, 75, 75)
        elif method == "Sobel Filter":
            sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
            return cv2.convertScaleAbs(cv2.magnitude(sobelx, sobely))
        elif method == "Laplacian Filter":
            return cv2.Laplacian(channel, cv2.CV_64F).astype(np.uint8)
        elif method == "Unsharp Masking":
            gaussian = cv2.GaussianBlur(channel, (5, 5), 2.0)
            return cv2.addWeighted(channel, 1.5, gaussian, -0.5, 0)
        elif method == "Motion Blur":
            kernel = np.zeros((15, 15))
            kernel[7, :] = np.ones(15)
            kernel = kernel / 15
            return cv2.filter2D(channel, -1, kernel)
        elif method == "Emboss Filter":
            kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            return cv2.filter2D(channel, -1, kernel) + 128
        elif method == "Prewitt Filter":
            kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            return cv2.addWeighted(cv2.filter2D(channel, -1, kernelx), 0.5,
                                   cv2.filter2D(channel, -1, kernely), 0.5, 0)
        elif method == "Mean Filter":
            return cv2.blur(channel, (5, 5))
        elif method == "Box Filter":
            return cv2.boxFilter(channel, -1, (10, 10))
        elif method == "Edge Enhance":
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            return cv2.filter2D(channel, -1, kernel)
        return channel

    def apply_noise(self, channel, method):
        if method == "No Noise":
            return channel

        channel_float = channel.astype(float)

        if method == "Gaussian Noise":
            noise = np.random.normal(0, 25, channel.shape)
            noisy = channel_float + noise
        elif method == "Salt & Pepper Noise":
            noise = np.random.random(channel.shape)
            channel_float[noise < 0.02] = 0  # salt
            channel_float[noise > 0.98] = 255  # pepper
            noisy = channel_float
        elif method == "Speckle Noise":
            noise = np.random.normal(0, 0.15, channel.shape)
            noisy = channel_float + channel_float * noise
        elif method == "Poisson Noise":
            noise = np.random.poisson(channel_float)
            noisy = noise
        # New Noise Types
        elif method == "Uniform Noise":
            noise = np.random.uniform(-50, 50, channel.shape)
            noisy = channel_float + noise
        elif method == "Rayleigh Noise":
            noise = np.random.rayleigh(30, channel.shape)
            noisy = channel_float + noise
        elif method == "Exponential Noise":
            noise = np.random.exponential(25, channel.shape)
            noisy = channel_float + noise
        elif method == "Gamma Noise":
            noise = np.random.gamma(2.0, 20.0, channel.shape)
            noisy = channel_float + noise
        elif method == "Periodic Noise":
            x, y = np.meshgrid(np.arange(channel.shape[1]), np.arange(channel.shape[0]))
            noise = 30 * np.sin(0.1 * x + 0.1 * y)
            noisy = channel_float + noise
        elif method == "Multiplicative Noise":
            noise = np.random.normal(1, 0.2, channel.shape)
            noisy = channel_float * noise
        elif method == "Localvar Noise":
            variance = channel_float / 255.0  # Local variance based on pixel intensity
            noise = np.random.normal(0, np.sqrt(variance) * 50, channel.shape)
            noisy = channel_float + noise
        elif method == "Mixed Noise":
            # Combination of Gaussian and Salt & Pepper
            gaussian = np.random.normal(0, 15, channel.shape)
            noisy = channel_float + gaussian
            noise_mask = np.random.random(channel.shape)
            noisy[noise_mask < 0.01] = 0
            noisy[noise_mask > 0.99] = 255
        elif method == "Quantization Noise":
            levels = 32  # Number of quantization levels
            noisy = np.round(channel_float / (255 / levels)) * (255 / levels)
        elif method == "Random Lines Noise":
            noisy = channel_float.copy()
            num_lines = 50
            for _ in range(num_lines):
                y = np.random.randint(0, channel.shape[0])
                noisy[y, :] = 255

        return np.clip(noisy, 0, 255).astype(np.uint8)

    def process_channel(self, channel, method, brightness_slider):
        if method == "No Processing":
            return channel
        elif method == "Histogram Equalization":
            return cv2.equalizeHist(channel)
        elif method == "Gaussian Blur":
            return cv2.GaussianBlur(channel, (5, 5), 0)
        elif method == "Median Filter":
            return cv2.medianBlur(channel, 5)
        elif method == "Brightness Adjustment":
            brightness = brightness_slider.value()
            channel = cv2.add(channel, brightness)
            return np.clip(channel, 0, 255).astype(np.uint8)
        elif method == "Lowpass Filter":
            kernel = np.ones((5, 5), np.float32) / 25
            return cv2.filter2D(channel, -1, kernel)
        elif method == "Highpass Filter":
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            return cv2.filter2D(channel, -1, kernel)
        else:
            return channel

    def open_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            grayscale_choice = QMessageBox.question(
                self, "Choose Image Format",
                "Do you want to open the image in grayscale?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if grayscale_choice == QMessageBox.StandardButton.Yes:
                image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(file_name, cv2.IMREAD_COLOR)

            # Update image states for input
            self.image_states['input']['current'] = image.copy()
            self.image_states['input']['original'] = image.copy()
            self.input_image = image.copy()
            self.original_input = image.copy()

            self.display_image(image, self.input_label)

            # Reset output states
            for port in ['output1', 'output2']:
                self.image_states[port] = {'current': None, 'original': None}
            self.output_image1 = None
            self.output_image2 = None
            self.output1_label.clear()
            self.output2_label.clear()

    def get_interpolation_method(self, method_name):
        if method_name == "Nearest Neighbor":
            return cv2.INTER_NEAREST
        elif method_name == "Linear":
            return cv2.INTER_LINEAR
        elif method_name == "Bilinear":  # OpenCV doesn't have a distinct bilinear flag, use linear
            return cv2.INTER_LINEAR
        elif method_name == "Cubic":
            return cv2.INTER_CUBIC
        else:
            raise ValueError(f"Unknown interpolation method: {method_name}")

    def reset_all(self):
        # Reset images to original state
        if self.original_input is not None:
            self.input_image = self.original_input.copy()
            self.display_image(self.input_image, self.input_label)

        self.output_image1 = None
        self.output_image2 = None
        self.original_output1 = None
        self.original_output2 = None
        self.output1_label.clear()
        self.output2_label.clear()

        # Reset ROIs
        self.roi_circles.clear()
        self.update_image_display()
        self.result_label.setText("Results: CNR and SNR will be displayed here")

        # Reset all comboboxes and sliders
        for target in [1, 2]:
            if hasattr(self, f'output{target}_controls'):
                controls = getattr(self, f'output{target}_controls')
                filter_combo = controls.findChild(QComboBox, f"filter_{target}")
                noise_combo = controls.findChild(QComboBox, f"noise_{target}")
                brightness_slider = controls.findChild(QSlider, f"brightness_{target}")

                if filter_combo:
                    filter_combo.setCurrentIndex(0)
                if noise_combo:
                    noise_combo.setCurrentIndex(0)
                if brightness_slider:
                    brightness_slider.setValue(0)
                    brightness_slider.hide()

    def reset_output(self, target):
        if target == 1:
            self.output_image1 = None
            self.original_output1 = None
            self.output1_label.clear()
            controls = self.output1_controls
        else:
            self.output_image2 = None
            self.original_output2 = None
            self.output2_label.clear()
            controls = self.output2_controls

        # Reset controls
        filter_combo = controls.findChild(QComboBox, f"filter_{target}")
        noise_combo = controls.findChild(QComboBox, f"noise_{target}")
        brightness_slider = controls.findChild(QSlider, f"brightness_{target}")

        if filter_combo:
            filter_combo.setCurrentIndex(0)
        if noise_combo:
            noise_combo.setCurrentIndex(0)
        if brightness_slider:
            brightness_slider.setValue(0)
            brightness_slider.hide()

    def apply_contrast(self, image, method, alpha=1.0, beta=0, gamma=1.0):
        if method == "No Contrast Enhancement":
            return image

        elif method == "Linear Contrast Stretch":
            return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        elif method == "Histogram Equalization":
            return cv2.equalizeHist(image)

        elif method == "CLAHE":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)

        elif method == "Gamma Correction":
            # Normalize the image to 0-1 range
            normalized = image.astype(float) / 255.0
            # Apply gamma correction
            corrected = np.power(normalized, gamma)
            # Scale back to 0-255 range
            return np.uint8(corrected * 255)

        return image

    def update_contrast_controls(self, method, target):
        sliders = self.findChild(QWidget, f"contrast_sliders_{target}")
        if not sliders:
            return

        if method == "Linear Contrast Stretch":
            sliders.show()
            sliders.findChild(QSlider, f"alpha_{target}").show()
            sliders.findChild(QSlider, f"beta_{target}").show()
            sliders.findChild(QSlider, f"gamma_{target}").hide()
        elif method == "Gamma Correction":
            sliders.show()
            sliders.findChild(QSlider, f"alpha_{target}").hide()
            sliders.findChild(QSlider, f"beta_{target}").hide()
            sliders.findChild(QSlider, f"gamma_{target}").show()
        else:
            sliders.hide()

    def show_histogram(self, image, title):
        if image is None:
            print(f"{title} not available.")
            return

        # Determine which port opened the histogram
        if title == "Input Image":
            self.current_histogram_source = "input"
        elif title == "Output 1":
            self.current_histogram_source = "output1"
        elif title == "Output 2":
            self.current_histogram_source = "output2"

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Interactive Histogram - {title}")
        dialog.setMinimumSize(800, 600)

        layout = QVBoxLayout()

        # Create histogram widget with reference to current source
        histogram = InteractiveHistogram(self, viewer=self)
        histogram.set_image(image)
        layout.addWidget(histogram)

        # Reset button now uses the tracked original state
        # Reset button now uses the tracked original state
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(lambda: self.reset_histogram(histogram))
        layout.addWidget(reset_btn)

        dialog.setLayout(layout)
        dialog.exec()

    def reset_histogram(self, histogram):
        if self.current_histogram_source:
            original_image = self.image_states[self.current_histogram_source]['original']
            if original_image is not None:
                histogram.set_image(original_image)

    def update_from_histogram(self, updated_image):
        if not self.current_histogram_source:
            return

        # Update only the specific port's current image
        self.image_states[self.current_histogram_source]['current'] = updated_image.copy()

        # Update the corresponding display
        if self.current_histogram_source == 'input':
            self.input_image = updated_image.copy()
            self.display_image(updated_image, self.input_label)
        elif self.current_histogram_source == 'output1':
            self.output_image1 = updated_image.copy()
            self.display_image(updated_image, self.output1_label)
        elif self.current_histogram_source == 'output2':
            self.output_image2 = updated_image.copy()
            self.display_image(updated_image, self.output2_label)


def exception_hook(exctype, value, tb):
    print(''.join(traceback.format_exception(exctype, value, tb)))
    sys.exit(1)

sys.excepthook = exception_hook
if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec())
