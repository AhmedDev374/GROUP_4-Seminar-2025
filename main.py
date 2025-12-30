import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from scipy import signal
import time
import threading
from collections import deque
import random
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings

warnings.filterwarnings('ignore')


class EMGProstheticSimulator:
    def __init__(self):
        # ------------------------
        # Basic GUI & simulation params
        # ------------------------
        self.root = tk.Tk()
        self.root.title("Deep Learning-Enhanced EMG Prosthetic Control System - Clinical Simulation")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')

        # Simulation parameters
        self.sampling_rate = 1000
        self.window_size = 200
        self.overlap = 2
        00
        self.num_channels = 8
        self.is_running = False
        self.current_gesture = "Rest"
        self.scenario = "Normal"

        # Buffers and histories
        self.emg_data = deque(maxlen=2000)          # each sample is an array of length num_channels
        self.processed_data = deque(maxlen=1000)   # stores dicts with features, gesture, confidence, timestamp
        self.prediction_history = deque(maxlen=200)
        self.confidence_history = deque(maxlen=200)
        self.latency_history = deque(maxlen=200)

        # Smoothed attention weights
        self.attention_history = np.ones(self.num_channels) / self.num_channels

        # Gestures mapping
        self.gestures = {
            0: "Rest",
            1: "Hand Open",
            2: "Hand Close",
            3: "Pinch Grip",
            4: "Point",
            5: "Lateral Grip",
            6: "Wrist Flexion",
            7: "Wrist Extension"
        }

        # Initialize AI model and scaler
        self.model = self.create_ai_model()
        self.scaler = StandardScaler()

        # Build GUI
        self.setup_gui()

        # Fill initial data
        self.initialize_data()

        # Thread control
        self.simulation_thread = None

    def create_ai_model(self):
        """Create a hybrid CNN-RNN model for EMG classification (kept from original design)."""
        model = keras.Sequential([
            layers.Input(shape=(self.window_size, self.num_channels)),
            layers.Conv1D(64, 5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Conv1D(128, 5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Conv1D(256, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
            layers.Dropout(0.3),
            layers.Bidirectional(layers.LSTM(64)),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(len(self.gestures), activation='softmax')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def generate_synthetic_emg(self, gesture, noise_level=0.1, fatigue_level=0.0):
        """Generate synthetic EMG (keeps original logic, clarified). Returns 1D array length num_channels."""
        # Use current time to vary signals
        t = time.time()
        base_frequency = 30 + gesture * 5
        amplitude_map = [0.05, 0.8, 1.0, 0.7, 0.6, 0.9, 0.5, 0.5]
        amplitude = amplitude_map[gesture] * (1 - fatigue_level * 0.3)

        channels_data = []
        for channel in range(self.num_channels):
            # slightly different amplitude and frequency per channel
            channel_amp = amplitude * (0.8 + 0.4 * (channel / max(1, self.num_channels - 1)))
            channel_freq = base_frequency * (0.9 + 0.2 * random.random())
            main_signal = channel_amp * np.sin(2 * np.pi * channel_freq * t + channel * 0.5)
            muap_signal = sum([0.1 * channel_amp * np.sin(2 * np.pi * (5 + mu * 10) * t + mu * 1.2) for mu in range(5)])
            noise = noise_level * np.random.randn()
            if self.scenario == "Noisy":
                noise *= 2
            elif self.scenario == "Fatigue":
                noise *= 1.5
                main_signal *= (1 - fatigue_level * 0.5)
            elif self.scenario == "Electrode Shift":
                main_signal *= 0.3 if channel in [0, 1] else 1.7 if channel in [6, 7] else 1.0
            combined_signal = main_signal + muap_signal + noise
            channels_data.append(float(combined_signal))

        return np.array(channels_data, dtype=float)

    def preprocess_emg(self, emg_window):
        """Extract simple features per channel from a window shaped (window_size, num_channels)"""
        features = []
        # emg_window expected shape: (window_size, num_channels)
        arr = np.array(emg_window)
        if arr.ndim != 2 or arr.shape[1] != self.num_channels:
            # If transposed earlier, try to reshape
            try:
                arr = arr.reshape(self.window_size, self.num_channels)
            except Exception:
                # fallback: flatten and pad/truncate
                arr = np.zeros((self.window_size, self.num_channels))
        for ch in range(self.num_channels):
            channel_data = arr[:, ch]
            mean_abs = np.mean(np.abs(channel_data))
            std_dev = np.std(channel_data)
            variance = np.var(channel_data)
            rms = np.sqrt(np.mean(channel_data ** 2))
            zero_cross = np.sum(np.abs(np.diff(np.sign(channel_data)))) / 2.0
            features.extend([mean_abs, std_dev, variance, rms, zero_cross])
        return np.array(features, dtype=float)

    def predict_gesture(self, emg_window):
        """
        NOTE: Original code used a heuristic prediction (randomized under scenarios).
        We keep that behavior (no real model inference), but cleaned and deterministic-ish.
        """
        # default: use current gesture as target
        gesture_id = next(k for k, v in self.gestures.items() if v == self.current_gesture)
        # scenario-induced misclassifications
        if self.scenario == "Noisy" and random.random() < 0.15:
            gesture_id = random.choice([g for g in self.gestures.keys() if g != gesture_id])
        elif self.scenario == "Fatigue" and random.random() < 0.10:
            gesture_id = random.choice([g for g in self.gestures.keys() if g != gesture_id])
        elif self.scenario == "Electrode Shift" and random.random() < 0.20:
            gesture_id = random.choice([g for g in self.gestures.keys() if g != gesture_id])

        confidence = max(0.05, min(0.99, 0.85 + (random.random() - 0.5) * 0.2 - (0.2 if self.scenario != "Normal" else 0)))
        return int(gesture_id), float(confidence)

    # ------------------------
    # GUI setup
    # ------------------------
    def setup_gui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_frame = ttk.LabelFrame(main_frame, text="Clinical Control Panel", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        viz_frame = ttk.Frame(main_frame)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.setup_control_panel(control_frame)
        self.setup_visualization_panel(viz_frame)

    def setup_control_panel(self, parent):
        title_label = ttk.Label(parent, text="AI-Enhanced Prosthetic Control",
                                font=('Arial', 14, 'bold'), foreground='#3498db')
        title_label.pack(pady=(0, 15))

        sim_frame = ttk.LabelFrame(parent, text="Simulation Control", padding=10)
        sim_frame.pack(fill=tk.X, pady=5)
        self.start_btn = ttk.Button(sim_frame, text="Start Simulation", command=self.start_simulation)
        self.start_btn.pack(fill=tk.X, pady=2)
        self.stop_btn = ttk.Button(sim_frame, text="Stop Simulation", command=self.stop_simulation, state='disabled')
        self.stop_btn.pack(fill=tk.X, pady=2)

        gesture_frame = ttk.LabelFrame(parent, text="Gesture Selection", padding=10)
        gesture_frame.pack(fill=tk.X, pady=5)
        self.gesture_var = tk.StringVar(value="Rest")
        for gesture in self.gestures.values():
            ttk.Radiobutton(gesture_frame, text=gesture, variable=self.gesture_var,
                            value=gesture, command=self.gesture_changed).pack(anchor=tk.W, pady=2)

        scenario_frame = ttk.LabelFrame(parent, text="Clinical Scenarios", padding=10)
        scenario_frame.pack(fill=tk.X, pady=5)
        self.scenario_var = tk.StringVar(value="Normal")
        for scenario in ["Normal", "Noisy", "Fatigue", "Electrode Shift", "Long-Term Use"]:
            ttk.Radiobutton(scenario_frame, text=scenario, variable=self.scenario_var,
                            value=scenario, command=self.scenario_changed).pack(anchor=tk.W, pady=2)

        metrics_frame = ttk.LabelFrame(parent, text="Performance Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, pady=5)
        self.accuracy_var = tk.StringVar(value="Accuracy: --%")
        ttk.Label(metrics_frame, textvariable=self.accuracy_var, font=('Arial', 10)).pack(anchor=tk.W)
        self.latency_var = tk.StringVar(value="Latency: -- ms")
        ttk.Label(metrics_frame, textvariable=self.latency_var, font=('Arial', 10)).pack(anchor=tk.W)
        self.confidence_var = tk.StringVar(value="Confidence: --%")
        ttk.Label(metrics_frame, textvariable=self.confidence_var, font=('Arial', 10)).pack(anchor=tk.W)

        model_frame = ttk.LabelFrame(parent, text="AI Model Information", padding=10)
        model_frame.pack(fill=tk.X, pady=5)
        model_info = (
            "Hybrid CNN-RNN Architecture:\n"
            "• 3 Convolutional Layers\n"
            "• 2 Bidirectional LSTM Layers\n"
            "• Attention Mechanism (simulated)\n"
            "• 8 Gesture Classification\n"
            "• Real-time Processing\n"
            "• Adaptive Learning\n"
        )
        ttk.Label(model_frame, text=model_info, justify=tk.LEFT, font=('Arial', 8)).pack(anchor=tk.W)

        clinical_frame = ttk.LabelFrame(parent, text="Clinical Assessment", padding=10)
        clinical_frame.pack(fill=tk.X, pady=5)
        self.assessment_text = tk.Text(clinical_frame, height=8, width=34, font=('Arial', 8))
        self.assessment_text.pack(fill=tk.BOTH)
        self.assessment_text.insert(tk.END, "System ready for clinical evaluation...\n\n")
        self.assessment_text.config(state=tk.DISABLED)

    def setup_visualization_panel(self, parent):
        # Create figure and grid
        self.fig = plt.Figure(figsize=(12, 8), dpi=100)
        self.fig.patch.set_facecolor('#ecf0f1')
        gs = gridspec.GridSpec(3, 2, figure=self.fig, hspace=0.6, wspace=0.3)

        # Top-left: Latency vs Confidence (time-series)
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax1.set_title('Latency vs Confidence', fontsize=12, fontweight='bold', pad=10)
        self.ax1.set_xlabel('Time Windows')
        self.ax1.set_ylabel('Latency (ms)')
        self.ax1.grid(True, alpha=0.3)
        self.latency_line, = self.ax1.plot([], [], label='Latency (ms)')
        # secondary axis for confidence
        self.ax1b = self.ax1.twinx()
        self.ax1b.set_ylabel('Confidence')
        self.confidence_line, = self.ax1b.plot([], [], linestyle='--', label='Confidence')

        # Top-right: REPLACED EMG Feature Space -> Channel RMS over time (clearer)
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax2.set_title('Channel RMS over Time (Recent Windows)', fontsize=12, fontweight='bold', pad=10)
        self.ax2.set_xlabel('Time Windows')
        self.ax2.set_ylabel('RMS Value')
        self.ax2.grid(True, alpha=0.3)
        # create one line per channel
        self.channel_lines = [self.ax2.plot([], [], label=f'Ch {i+1}')[0] for i in range(self.num_channels)]
        self.ax2.legend(loc='upper right', fontsize=7)

        # Middle-left & Middle-right: Combined panel
        # We'll use a single axes (ax3) for gesture distribution (bars) and overlay performance metrics on twin axis.
        self.ax3 = self.fig.add_subplot(gs[1, :])
        self.ax3.set_title('Gesture Prediction Distribution (bars)  —  Performance Metrics (lines)', fontsize=12, fontweight='bold', pad=10)
        self.ax3.set_xlabel('Gesture')
        self.ax3.set_ylabel('Prediction Frequency')
        self.ax3.grid(True, alpha=0.2)
        # twin axis for performance metrics
        self.ax3_perf = self.ax3.twinx()
        self.ax3_perf.set_ylabel('Performance Value (normalized)')
        # placeholders
        self.gesture_bars = None
        self.performance_lines = []
        # initialize 3 perf lines (Accuracy, Latency(normalized), Confidence)
        perf_placeholders = ['Accuracy', 'Latency', 'Confidence']
        for _ in perf_placeholders:
            line, = self.ax3_perf.plot([], [], marker='o', linestyle='-')
            self.performance_lines.append(line)

        # Bottom-left: Temporal pattern analysis (per gesture confidence over recent windows)
        self.ax4 = self.fig.add_subplot(gs[2, 0])
        self.ax4.set_title('Temporal Pattern Analysis', fontsize=12, fontweight='bold', pad=10)
        self.ax4.set_ylabel('Normalized Amplitude/Confidence')
        self.ax4.set_xlabel('Recent Windows')
        self.ax4.grid(True, alpha=0.3)
        self.temporal_lines = [self.ax4.plot([], [], label=gesture, linewidth=1)[0] for gesture in self.gestures.values()]
        self.ax4.legend(loc='upper right', fontsize=6)

        # Bottom-right: AI Attention Weights
        self.ax5 = self.fig.add_subplot(gs[2, 1])
        self.ax5.set_title('AI Model Attention Weights', fontsize=12, fontweight='bold', pad=10)
        self.ax5.set_ylabel('Attention Weight')
        self.ax5.set_xlabel('EMG Channels')
        self.ax5.grid(True, alpha=0.3)
        self.attention_bars = self.ax5.bar(range(1, self.num_channels + 1), self.attention_history)

        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ------------------------
    # Data initialization and simulation control
    # ------------------------
    def initialize_data(self):
        # Fill with rest-like samples initially
        for _ in range(150):
            sample = self.generate_synthetic_emg(0, noise_level=0.1, fatigue_level=0.0)
            self.emg_data.append(sample)
        # also initialize some processed_data to avoid empty plotting
        for _ in range(20):
            # fake window data
            fake_window = [self.generate_synthetic_emg(0) for _ in range(self.window_size)]
            features = self.preprocess_emg(fake_window)
            self.processed_data.append({'features': features, 'gesture': 0, 'confidence': 0.9, 'timestamp': time.time()})

    def start_simulation(self):
        if self.is_running:
            return
        self.is_running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        # start simulation thread
        self.simulation_thread = threading.Thread(target=self.run_simulation, daemon=True)
        self.simulation_thread.start()
        # start GUI update loop
        self.root.after(250, self.update_gui)

    def stop_simulation(self):
        if not self.is_running:
            return
        self.is_running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')

    def gesture_changed(self):
        self.current_gesture = self.gesture_var.get()
        self.add_assessment_note(f"Gesture changed to: {self.current_gesture}")

    def scenario_changed(self):
        self.scenario = self.scenario_var.get()
        self.add_assessment_note(f"Scenario changed to: {self.scenario}")
        notes = {
            "Normal": "Optimal conditions. Expected accuracy >98%",
            "Noisy": "High noise environment. Accuracy may drop to ~90%",
            "Fatigue": "Muscle fatigue detected. Adaptive learning engaged",
            "Electrode Shift": "Electrode position changed. Domain adaptation active",
            "Long-Term Use": "Long-term stability monitoring enabled"
        }
        if self.scenario in notes:
            self.add_assessment_note(f"Clinical Note: {notes[self.scenario]}")

    def run_simulation(self):
        """Background thread that generates synthetic EMG and simulates prediction pipeline."""
        fatigue_level = 0.0
        while self.is_running:
            # compute current gesture id
            gesture_id = next(k for k, v in self.gestures.items() if v == self.current_gesture)
            if self.scenario in ["Fatigue", "Long-Term Use"]:
                fatigue_level = min(0.8, fatigue_level + 0.001)
            else:
                fatigue_level = max(0.0, fatigue_level - 0.002)

            noise_level = 0.3 if self.scenario == "Noisy" else 0.1
            sample = self.generate_synthetic_emg(gesture_id, noise_level=noise_level, fatigue_level=fatigue_level)
            self.emg_data.append(sample)

            # Windowing: every overlap samples (simulates streaming window with overlap)
            if len(self.emg_data) >= self.window_size and (len(self.emg_data) - self.window_size) % self.overlap == 0:
                # get last window_size samples, each sample is array length num_channels
                window_list = list(self.emg_data)[-self.window_size:]
                # shape -> (window_size, num_channels)
                window_array = np.vstack(window_list)
                predicted_gesture, confidence = self.predict_gesture(window_array)
                self.prediction_history.append(predicted_gesture)
                self.confidence_history.append(confidence)
                # latency simulation
                latency_sim = 50 + 20 * random.random()
                self.latency_history.append(latency_sim)
                # features
                features = self.preprocess_emg(window_array)
                self.processed_data.append({
                    'features': features,
                    'gesture': gesture_id,
                    'confidence': confidence,
                    'timestamp': time.time()
                })
            # throttle generation, keep responsive but not too fast
            time.sleep(0.02)

    # ------------------------
    # GUI update
    # ------------------------
    def update_gui(self):
        """Update all plots and UI elements. Should only be called from main thread (via after())."""
        if not self.is_running:
            # still update small parts to reflect state changes if needed
            return

        # ---------- Latency vs Confidence (ax1 & ax1b) ----------
        if self.latency_history:
            x = list(range(len(self.latency_history)))
            y_latency = list(self.latency_history)
            y_conf = list(self.confidence_history) if self.confidence_history else [0] * len(x)
            self.latency_line.set_data(x, y_latency)
            self.confidence_line.set_data(x, y_conf)
            # autoscale
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax1b.relim()
            self.ax1b.autoscale_view()
            # limit y-range for nice viz
            max_lat = max(max(y_latency) * 1.1, 100)
            self.ax1.set_ylim(0, max_lat)
            self.ax1b.set_ylim(0, 1)

        # ---------- Channel RMS over Time (ax2) ----------
        if self.processed_data:
            recent = list(self.processed_data)[-60:]  # keep last 60 windows
            # compute RMS per window per channel: features per channel at indices 3::5
            rms_per_channel = np.array([d['features'][3::5] for d in recent])  # shape (n_windows, n_channels)
            n_windows = rms_per_channel.shape[0]
            x = list(range(n_windows))
            for ch in range(self.num_channels):
                y = rms_per_channel[:, ch] if n_windows > 0 else np.array([])
                # smooth/safe cast
                y_plot = np.array(y, dtype=float)
                downsample = 10  # plot every 2nd point
                self.channel_lines[ch].set_data(x[::downsample], y_plot[::downsample]);            self.ax2.set_xlim(0, max(10, n_windows - 1))
            # autoscale y to data range
            all_rms = rms_per_channel.flatten() if rms_per_channel.size > 0 else np.array([0.1])
            ymin, ymax = 0, max(0.001, all_rms.max() * 1.2)
            self.ax2.set_ylim(ymin, ymax)

        # ---------- Combined Gesture Distribution + Performance Metrics (ax3) ----------
        # Gesture distribution (from prediction_history)
        gesture_counts = [list(self.prediction_history).count(g) for g in self.gestures.keys()] if self.prediction_history else [0] * len(self.gestures)
        total = max(1, sum(gesture_counts))
        percentages = [count / total for count in gesture_counts]
        # Clear and replot bars (safer than trying to update bars in-place)
        self.ax3.cla()
        self.ax3.set_title('Gesture Prediction Distribution (bars)  —  Performance Metrics (lines)', fontsize=12, fontweight='bold', pad=10)
        self.ax3.set_xlabel('Gesture')
        self.ax3.set_ylabel('Prediction Frequency')
        self.ax3.grid(True, alpha=0.2)
        bar_container = self.ax3.bar(list(self.gestures.values()), percentages, alpha=0.75)
        for bar, pct in zip(bar_container, percentages):
            self.ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{pct:.1%}', ha='center', va='bottom', fontsize=8)

        # performance metrics overlay on twin axis
        self.ax3_perf = self.ax3.twinx()
        perf_time = list(range(len(self.confidence_history))) if self.confidence_history else []
        # Example synthetic accuracy (like original): around 0.85-0.95
        perf_accuracy = [0.85 + 0.1 * random.random() for _ in perf_time]
        perf_latency = [min(1.0, (lat / 200.0)) for lat in self.latency_history] if self.latency_history else []
        perf_confidence = list(self.confidence_history) if self.confidence_history else []

        # Plot normalized performance on the same x-scale as the bar indices by mapping them to bar centers.
        # For simplicity, we'll plot averaged performance values as points aligned to gesture bar centers.
        gesture_positions = np.arange(len(self.gestures))
        # compute average performance across recent windows
        avg_acc = np.mean(perf_accuracy) if perf_accuracy else 0
        avg_lat = np.mean(perf_latency) if perf_latency else 0
        avg_conf = np.mean(perf_confidence) if perf_confidence else 0
        # replicate same value per gesture (so lines show across gesture x-axis)
        self.ax3_perf.plot(gesture_positions, [avg_acc] * len(gesture_positions), linestyle='-', marker='o', label='Accuracy (avg)')
        self.ax3_perf.plot(gesture_positions, [avg_lat] * len(gesture_positions), linestyle='--', marker='x', label='Latency (norm avg)')
        self.ax3_perf.plot(gesture_positions, [avg_conf] * len(gesture_positions), linestyle='-.', marker='s', label='Confidence (avg)')
        self.ax3_perf.set_ylim(0, 1)
        # add legend for perf (placed slightly above)
        self.ax3_perf.legend(loc='upper left', fontsize=8)

        # Update textual metrics
        self.accuracy_var.set(f"Accuracy: {avg_acc * 100:.1f}%")
        self.latency_var.set(f"Latency: {np.mean(self.latency_history) if self.latency_history else 0:.1f} ms")
        self.confidence_var.set(f"Confidence: {avg_conf * 100:.1f}%")

        # ---------- Temporal analysis (ax4) ----------
        if self.processed_data:
            recent = list(self.processed_data)[-30:]
            n = len(recent)
            x = list(range(n))
            for gesture_id, line in enumerate(self.temporal_lines):
                y = [d['confidence'] if d['gesture'] == gesture_id else 0.0 for d in recent]
                line.set_data(x, y)
            self.ax4.set_xlim(0, max(1, n - 1))
            self.ax4.set_ylim(0, 1)

        # ---------- Attention weights (ax5) ----------
        # Smoothly update attention_history
        alpha = 0.12
        new_weights = np.random.dirichlet(np.ones(self.num_channels))
        self.attention_history = alpha * new_weights + (1 - alpha) * self.attention_history
        self.ax5.cla()
        self.ax5.set_title('AI Model Attention Weights', fontsize=12, fontweight='bold')
        self.ax5.set_xlabel('EMG Channels')
        self.ax5.set_ylabel('Attention Weight')
        bars = self.ax5.bar(range(1, self.num_channels + 1), self.attention_history, alpha=0.8)
        self.ax5.set_ylim(0, 1)
        for bar, weight in zip(bars, self.attention_history):
            self.ax5.text(bar.get_x() + bar.get_width() / 2., weight, f'{weight:.2f}', ha='center', va='bottom', fontsize=8)

        # Tight layout and redraw
        try:
            self.fig.tight_layout()
        except Exception:
            pass
        self.canvas.draw_idle()

        # schedule next update
        self.root.after(250, self.update_gui)

    def add_assessment_note(self, note):
        self.assessment_text.config(state=tk.NORMAL)
        timestamp = time.strftime("%H:%M:%S")
        self.assessment_text.insert(tk.END, f"[{timestamp}] {note}\n")
        self.assessment_text.see(tk.END)
        self.assessment_text.config(state=tk.DISABLED)

    def run(self):
        # Start Tkinter main loop
        self.root.mainloop()


def main():
    simulator = EMGProstheticSimulator()
    simulator.run()


if __name__ == "__main__":
    main()
