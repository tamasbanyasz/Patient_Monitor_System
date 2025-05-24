import sys
import random
import pandas as pd
import joblib
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QPushButton, QHBoxLayout, QHeaderView, QFrame, QAbstractItemView
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QBrush
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

"""
This Python program is a real-time patiente monitoring system with a graphical interface built using PyQt5. It simulates patient vital 
signs such as EEG alpha activity, pupil size, pulse, body temperature, and oxygen saturation. Every 3 seconds, it generates new data, 
displays it in a table, and uses a pre-trained Random Forest model to predict the appropriate medication. It visually shows body temperature 
with a colored circle and tracks pulse changes on a live-updating line chart. Users can start monitoring and toggle the visibility of the 
visualizations with buttons.

"""

model = joblib.load("Random_Forest_model.pkl")

columns = [
    'EEG_alfa', 'Pupilla_átmérő_mm', 'Pulzus_bpm',
    'Testhőmérséklet_C', 'Oxigénszint_%'
]

last_20_pulse_values = []

def generate_patient_data():
    return {
        'EEG_alfa': round(random.uniform(8.0, 13.0), 2),
        'Pupilla_átmérő_mm': round(random.uniform(2.0, 6.5), 2),
        'Pulzus_bpm': random.randint(50, 140),
        'Testhőmérséklet_C': round(random.uniform(35.5, 40.5), 1),
        'Oxigénszint_%': random.randint(85, 100)
    }

class TemperatureCircle(QWidget):
    def __init__(self):
        super().__init__()
        self.temp = 36.5
        self.setMinimumHeight(100)

    def set_temperature(self, temp):
        self.temp = temp
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        radius = min(self.width(), self.height()) // 2 - 10
        center_x = self.width() // 2
        center_y = self.height() // 2

        if self.temp < 36.5:
            color = QColor("#0096FF")
        elif self.temp < 37.5:
            color = QColor("#FFF700")
        elif self.temp < 38.5:
            color = QColor("#FFA500")
        else:
            color = QColor("#FF0000")

        painter.setBrush(QBrush(color))
        painter.drawEllipse(center_x - radius, center_y - radius, 2 * radius, 2 * radius)

        painter.setPen(Qt.black)
        painter.setFont(QFont("Arial", 12, QFont.Bold))
        painter.drawText(self.rect(), Qt.AlignCenter, f"{self.temp} °C")

class PatientMonitor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Patient Monitoring System")
        self.resize(1000, 800)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Suggested drug display
        self.drug_frame = QFrame()
        self.drug_frame.setFrameShape(QFrame.NoFrame)
        
        drug_layout = QVBoxLayout()
        self.drug_frame.setLayout(drug_layout)

        self.drug_label = QLabel("Suggested Medicine: -")
        self.drug_label.setAlignment(Qt.AlignCenter)
        self.drug_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.drug_label.setStyleSheet("border: 3px solid gray; border-radius: 15px; color: darkblue; font-weight: bold; background: none;")
        drug_layout.addWidget(self.drug_label)

        self.layout.addWidget(self.drug_frame)

        # Data table
        self.table = QTableWidget()
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        header = self.table.horizontalHeader()
        header.setSectionsClickable(False)
        header.setStyleSheet("""
            QHeaderView::section {
                background-color: #d0d0d0;
                color: black;
                font-weight: bold;
                padding: 4px;
            }
        """)
        header.setSectionResizeMode(QHeaderView.Stretch)

        self.layout.addWidget(self.table)

        # Visualization frame
        self.visual_frame = QFrame()
        self.visual_frame.setFrameShape(QFrame.Box)
        self.visual_frame.setLineWidth(2)
        self.visual_frame.setStyleSheet("""
            border: 2px solid gray;
            border-radius: 15px;
        """)
        visual_layout = QHBoxLayout()
        self.visual_frame.setLayout(visual_layout)
        self.layout.addWidget(self.visual_frame)

        # Temperature circle
        temp_container = QVBoxLayout()
        self.temp_label = QLabel("Body Temperature")
        self.temp_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.temp_label.setAlignment(Qt.AlignCenter)
        self.temp_label.setStyleSheet("background: none; border: none;")
        temp_container.addWidget(self.temp_label)

        self.temp_circle = TemperatureCircle()
        temp_container.addWidget(self.temp_circle)
        visual_layout.addLayout(temp_container, 1)

        # Pulse graph
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        visual_layout.addWidget(self.canvas, 3)

        button_layout = QVBoxLayout()

        # Toggle button
        self.toggle_button = QPushButton("Hide Visualizations")
        self.toggle_button.setStyleSheet("""
            background-color: white;
            border: 1px solid gray;
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
        """)
        self.toggle_button.setFixedWidth(300)
        button_layout.addWidget(self.toggle_button, alignment=Qt.AlignCenter)

        # Start Monitoring button
        self.button = QPushButton("Start Monitoring")
        self.button.setStyleSheet("""
            background-color: white;
            border: 1px solid gray;
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
        """)
        self.button.setFixedWidth(300)
        button_layout.addWidget(self.button, alignment=Qt.AlignCenter)

        center_layout = QHBoxLayout()
        center_layout.addStretch()
        center_layout.addLayout(button_layout)
        center_layout.addStretch()

        self.layout.addLayout(center_layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)

        self.toggle_button.clicked.connect(self.toggle_visualization)
        self.button.clicked.connect(self.start_monitoring)
        
    def toggle_visualization(self):
        if self.visual_frame.isVisible():
            self.visual_frame.hide()
            self.toggle_button.setText("Show Visualizations")
        else:
            self.visual_frame.show()
            self.toggle_button.setText("Hide Visualizations")

    def start_monitoring(self):
        self.timer.start(3000)
        self.button.setEnabled(False)

    def update_data(self):
        new_data = generate_patient_data()

        row = self.table.rowCount()
        self.table.insertRow(row)
        for col, key in enumerate(columns):
            self.table.setItem(row, col, QTableWidgetItem(str(new_data[key])))

        input_df = pd.DataFrame([new_data])
        prediction = model.predict(input_df)[0]
        if prediction != '-':
            self.drug_label.setText(f"Suggested Medicine: {prediction}")
            self.drug_label.setStyleSheet("border: 3px solid gray; border-radius: 15px; color: red; font-weight: bold; background: none;")
        else:
            self.drug_label.setText("No medication needed.")
            self.drug_label.setStyleSheet("border: 3px solid gray; border-radius: 15px; color: green; font-weight: bold; background: none;")

        pulse = new_data['Pulzus_bpm']
        last_20_pulse_values.append(pulse)
        if len(last_20_pulse_values) > 20:
            last_20_pulse_values.pop(0)
        self.update_pulse_graph()

        self.temp_circle.set_temperature(new_data['Testhőmérséklet_C'])

    def update_pulse_graph(self):
        self.ax.clear()
        y_values = last_20_pulse_values.copy()
        if len(y_values) < 20:
            y_values = [np.nan] * (20 - len(y_values)) + y_values

        self.ax.plot(range(20), y_values, marker='o', color='darkred')
        self.ax.set_title("Pulse Trend (bpm)")
        self.ax.set_ylabel("Pulse")
        self.ax.set_xlabel("Time")
        self.ax.set_xlim(0, 19)
        self.ax.set_ylim(40, 150)
        self.ax.grid(True)
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PatientMonitor()
    window.show()
    sys.exit(app.exec_())
