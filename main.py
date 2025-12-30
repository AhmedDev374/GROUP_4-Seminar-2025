"""
================================================================================
PROFESSIONAL ECG SIGNAL CONDITIONING SYSTEM - ADVANCED ANALYTICS DASHBOARD
International Level Project - Biomedical Signal Processing Laboratory
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import lfilter, freqz, butter, cheby1, cheby2, ellip
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import pandas as pd
import time
from datetime import datetime
import warnings
import json
import os
from threading import Thread
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


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

class Config:
    """Project configuration and constants"""
    PROJECT_NAME = "ECG Signal Conditioning System"
    VERSION = "v2.1 Professional"
    AUTHOR = "Biomedical Signal Processing Lab"
    DATE = datetime.now().strftime("%Y-%m-%d")

    # Signal parameters
    SAMPLING_RATE = 1000  # Hz
    DEFAULT_DURATION = 5  # seconds

    # Filter specifications
    HPF_CUTOFF = 0.5  # Hz
    LPF_CUTOFF = 150  # Hz
    NOTCH_FREQ = 50  # Hz
    NOTCH_Q = 30

    # Plot colors
    COLORS = {
        'clean': '#2E7D32',  # Green
        'noisy': '#C62828',  # Red
        'filtered': '#1565C0',  # Blue
        'hpf': '#FF8F00',  # Orange
        'lpf': '#6A1B9A',  # Purple
        'notch': '#00897B',  # Teal
        'background': '#F5F5F5',
        'grid': '#E0E0E0',
        'text': '#212121'
    }


# ============================================================================
# ADVANCED ECG SIGNAL GENERATOR
# ============================================================================

class AdvancedECGGenerator:
    """Professional ECG signal generator with realistic pathologies"""

    def __init__(self, fs=1000):
        self.fs = fs
        self.time = None
        self.signals = {}
        self.metadata = {}

    def generate(self, duration=5, heart_rate=72, pathology=None, noise_level=0.5):
        """Generate ECG signal with configurable parameters"""
        self.time = np.linspace(0, duration, int(self.fs * duration))

        # Generate clean ECG
        clean_ecg = self._generate_clean_ecg(heart_rate)

        # Apply pathology if specified
        if pathology:
            clean_ecg = self._apply_pathology(clean_ecg, pathology)

        # Add noise components
        noisy_ecg = self._add_realistic_noise(clean_ecg, noise_level)

        # Store signals and metadata
        self.signals = {
            'time': self.time,
            'clean': clean_ecg,
            'noisy': noisy_ecg,
            'fs': self.fs
        }

        self.metadata = {
            'duration': duration,
            'heart_rate': heart_rate,
            'pathology': pathology,
            'noise_level': noise_level,
            'sampling_rate': self.fs,
            'timestamp': datetime.now().isoformat()
        }

        return self.signals

    def _generate_clean_ecg(self, heart_rate):
        """Generate clean ECG using piecewise Gaussian model"""
        t = self.time
        rr_interval = 60 / heart_rate  # seconds between beats

        # Initialize signal
        ecg = np.zeros_like(t)

        # Generate each heartbeat
        for i in range(len(t)):
            pos_in_cycle = t[i] % rr_interval
            normalized_pos = pos_in_cycle / rr_interval

            # P wave (Gaussian)
            p_amplitude = 0.25
            p_center = 0.15
            p_width = 0.02
            p_wave = p_amplitude * np.exp(-((normalized_pos - p_center) ** 2) / (2 * p_width ** 2))

            # QRS complex (narrow Gaussian)
            qrs_amplitude = 1.0
            qrs_center = 0.35
            qrs_width = 0.01
            qrs_complex = qrs_amplitude * np.exp(-((normalized_pos - qrs_center) ** 2) / (2 * qrs_width ** 2))

            # T wave (Gaussian)
            t_amplitude = 0.3
            t_center = 0.6
            t_width = 0.03
            t_wave = t_amplitude * np.exp(-((normalized_pos - t_center) ** 2) / (2 * t_width ** 2))

            # Combine
            ecg[i] = p_wave + qrs_complex + t_wave

        # Add respiration modulation
        respiration = 0.1 * np.sin(2 * np.pi * 0.2 * t)
        ecg += respiration

        return ecg

    def _apply_pathology(self, ecg, pathology):
        """Apply various ECG pathologies"""
        if pathology == "tachycardia":
            # Add high heart rate variation
            hr_variation = 0.1 * np.sin(2 * np.pi * 0.5 * self.time)
            modulated = signal.resample(ecg, len(ecg) * 2 // 3)
            ecg = np.interp(self.time, np.linspace(0, self.time[-1], len(modulated)), modulated)

        elif pathology == "bradycardia":
            # Slow heart rate
            modulated = signal.resample(ecg, len(ecg) * 3 // 2)
            ecg = np.interp(self.time, np.linspace(0, self.time[-1], len(modulated)), modulated)

        elif pathology == "arrhythmia":
            # Irregular rhythm
            irregular = np.copy(ecg)
            for i in range(0, len(irregular), 200):
                if np.random.random() > 0.7:
                    irregular[i:i + 50] = 0
            ecg = irregular

        return ecg

    def _add_realistic_noise(self, clean_signal, noise_level):
        """Add realistic biomedical noise components"""
        t = self.time

        # 1. Baseline wander (very low frequency)
        baseline_wander = (0.3 * noise_level * np.sin(2 * np.pi * 0.3 * t) +
                           0.1 * noise_level * np.sin(2 * np.pi * 0.1 * t))

        # 2. Power line interference (50Hz + harmonics)
        power_line = (0.25 * noise_level * np.sin(2 * np.pi * 50 * t + np.random.random()) +
                      0.1 * noise_level * np.sin(2 * np.pi * 100 * t) +
                      0.05 * noise_level * np.sin(2 * np.pi * 150 * t))

        # 3. Muscle artifact (EMG)
        emg_noise = np.zeros_like(t)
        for freq in [20, 40, 60, 80]:
            emg_noise += (0.05 * noise_level * np.sin(2 * np.pi * freq * t + np.random.random()))

        # 4. Electrode motion artifact
        electrode_motion = 0.15 * noise_level * np.sin(2 * np.pi * 1 * t)

        # 5. White measurement noise
        white_noise = 0.1 * noise_level * np.random.randn(len(t))

        # Combine all noise
        total_noise = baseline_wander + power_line + emg_noise + electrode_motion + white_noise

        return clean_signal + total_noise

    def calculate_statistics(self):
        """Calculate comprehensive signal statistics"""
        if not self.signals:
            return {}

        clean = self.signals['clean']
        noisy = self.signals['noisy']

        stats = {
            'mean_clean': np.mean(clean),
            'std_clean': np.std(clean),
            'max_clean': np.max(clean),
            'min_clean': np.min(clean),
            'mean_noisy': np.mean(noisy),
            'std_noisy': np.std(noisy),
            'max_noisy': np.max(noisy),
            'min_noisy': np.min(noisy),
            'snr': 10 * np.log10(np.var(clean) / np.var(noisy - clean)),
            'rms_clean': np.sqrt(np.mean(clean ** 2)),
            'rms_noisy': np.sqrt(np.mean(noisy ** 2)),
            'dynamic_range': np.max(clean) - np.min(clean)
        }

        return stats


# ============================================================================
# PROFESSIONAL ANALOG FILTER DESIGNER
# ============================================================================

class ProfessionalFilterDesigner:
    """Design and simulate analog filters using different topologies"""

    def __init__(self, fs=1000):
        self.fs = fs
        self.designs = {}

    def design_hpf(self, fc=0.5, order=2, topology='butterworth'):
        """Design High-Pass Filter"""
        # Analog design calculations
        if topology == 'butterworth':
            b, a = butter(order, fc / (self.fs / 2), btype='high')
            ripple = None
        elif topology == 'chebyshev1':
            b, a = cheby1(order, 0.5, fc / (self.fs / 2), btype='high')
            ripple = 0.5
        elif topology == 'chebyshev2':
            b, a = cheby2(order, 40, fc / (self.fs / 2), btype='high')
            ripple = 40
        elif topology == 'elliptic':
            b, a = ellip(order, 0.5, 40, fc / (self.fs / 2), btype='high')
            ripple = (0.5, 40)
        else:
            b, a = butter(order, fc / (self.fs / 2), btype='high')
            ripple = None

        # Calculate analog component values (theoretical)
        R = 10e3  # Base resistor value
        C = 1 / (2 * np.pi * fc * R)

        design = {
            'type': 'HPF',
            'topology': topology,
            'order': order,
            'fc_desired': fc,
            'components': {
                'R1': R,
                'R2': R,
                'C1': C,
                'C2': C / 2 if order == 2 else C
            },
            'transfer_function': {'b': b, 'a': a},
            'ripple': ripple
        }

        self.designs['hpf'] = design
        return design

    def design_lpf(self, fc=150, order=2, topology='butterworth'):
        """Design Low-Pass Filter"""
        if topology == 'butterworth':
            b, a = butter(order, fc / (self.fs / 2), btype='low')
            ripple = None
        elif topology == 'chebyshev1':
            b, a = cheby1(order, 0.5, fc / (self.fs / 2), btype='low')
            ripple = 0.5
        elif topology == 'chebyshev2':
            b, a = cheby2(order, 40, fc / (self.fs / 2), btype='low')
            ripple = 40
        elif topology == 'elliptic':
            b, a = ellip(order, 0.5, 40, fc / (self.fs / 2), btype='low')
            ripple = (0.5, 40)
        else:
            b, a = butter(order, fc / (self.fs / 2), btype='low')
            ripple = None

        # Calculate analog component values
        R = 10e3
        C = 1 / (2 * np.pi * fc * R)

        design = {
            'type': 'LPF',
            'topology': topology,
            'order': order,
            'fc_desired': fc,
            'components': {
                'R1': R,
                'R2': R,
                'C1': C,
                'C2': C / 2 if order == 2 else C
            },
            'transfer_function': {'b': b, 'a': a},
            'ripple': ripple
        }

        self.designs['lpf'] = design
        return design

    def design_notch(self, f0=50, Q=30, topology='twin-t'):
        """Design Notch Filter"""
        # Digital design
        w0 = f0 / (self.fs / 2)
        bw = f0 / Q
        b, a = signal.iirnotch(w0, Q)

        # Twin-T analog components
        R = 1 / (2 * np.pi * f0 * 10e-9)  # Using 10nF capacitor
        C = 10e-9

        design = {
            'type': 'NOTCH',
            'topology': topology,
            'f0_desired': f0,
            'Q_desired': Q,
            'components': {
                'R1': R,
                'R2': R,
                'R3': R / 2,
                'C1': C,
                'C2': C,
                'C3': 2 * C
            },
            'transfer_function': {'b': b, 'a': a}
        }

        self.designs['notch'] = design
        return design

    def analyze_filter(self, design):
        """Analyze filter performance"""
        b = design['transfer_function']['b']
        a = design['transfer_function']['a']

        # Frequency response
        w, h = freqz(b, a, worN=2000)
        f = w * self.fs / (2 * np.pi)
        mag = 20 * np.log10(np.abs(h))
        phase = np.angle(h, deg=True)

        # Group delay
        w_gd, gd = signal.group_delay((b, a))
        f_gd = w_gd * self.fs / (2 * np.pi)

        # Find -3dB frequency
        idx_3db = np.where(mag <= -3)[0]
        f_3db = f[idx_3db[0]] if len(idx_3db) > 0 else None

        analysis = {
            'frequency': f,
            'magnitude': mag,
            'phase': phase,
            'group_delay': (f_gd, gd),
            'fc_3db': f_3db,
            'dc_gain': mag[0],
            'peak_gain': np.max(mag),
            'stopband_attenuation': np.min(mag)
        }

        return analysis

    def apply_filter(self, signal_data, design):
        """Apply filter to signal"""
        b = design['transfer_function']['b']
        a = design['transfer_function']['a']
        return lfilter(b, a, signal_data)


# ============================================================================
# ADVANCED ANALYTICS ENGINE
# ============================================================================

class AnalyticsEngine:
    """Comprehensive signal analysis and metrics calculation"""

    @staticmethod
    def calculate_snr(clean, noisy):
        """Calculate Signal-to-Noise Ratio"""
        signal_power = np.mean(clean ** 2)
        noise_power = np.mean((noisy - clean) ** 2)
        return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf

    @staticmethod
    def calculate_thd(signal_data, fs, fundamental=1):
        """Calculate Total Harmonic Distortion"""
        # FFT analysis
        n = len(signal_data)
        fft_result = np.fft.fft(signal_data)
        frequencies = np.fft.fftfreq(n, 1 / fs)

        # Find harmonics
        fundamental_idx = np.argmin(np.abs(frequencies - fundamental))
        harmonic_power = 0
        fundamental_power = np.abs(fft_result[fundamental_idx]) ** 2

        for i in range(2, 10):  # Up to 9th harmonic
            harmonic_idx = np.argmin(np.abs(frequencies - i * fundamental))
            harmonic_power += np.abs(fft_result[harmonic_idx]) ** 2

        return np.sqrt(harmonic_power / fundamental_power) * 100 if fundamental_power > 0 else 0

    @staticmethod
    def calculate_rms(signal_data):
        """Calculate Root Mean Square"""
        return np.sqrt(np.mean(signal_data ** 2))

    @staticmethod
    def calculate_crest_factor(signal_data):
        """Calculate Crest Factor"""
        rms = AnalyticsEngine.calculate_rms(signal_data)
        peak = np.max(np.abs(signal_data))
        return peak / rms if rms > 0 else 0

    @staticmethod
    def calculate_spectral_analysis(signal_data, fs):
        """Perform comprehensive spectral analysis"""
        n = len(signal_data)
        fft_result = np.fft.fft(signal_data)
        frequencies = np.fft.fftfreq(n, 1 / fs)

        # Power spectral density
        psd = np.abs(fft_result) ** 2 / (fs * n)

        # Find dominant frequencies
        idx_pos = np.where(frequencies >= 0)[0]
        dominant_idx = np.argsort(psd[idx_pos])[-5:]  # Top 5 frequencies
        dominant_freqs = frequencies[idx_pos][dominant_idx]
        dominant_powers = psd[idx_pos][dominant_idx]

        return {
            'frequencies': frequencies[idx_pos],
            'psd': psd[idx_pos],
            'dominant_frequencies': dominant_freqs,
            'dominant_powers': dominant_powers,
            'bandwidth_90': np.percentile(frequencies[idx_pos], 90)
        }


# ============================================================================
# PROFESSIONAL DASHBOARD GUI
# ============================================================================

class ProfessionalDashboard:
    """Main application dashboard with professional interface"""

    def __init__(self, root):
        self.root = root
        self.root.title(f"{Config.PROJECT_NAME} - {Config.VERSION}")
        self.root.geometry("1600x900")
        self.root.configure(bg='white')

        # Initialize components
        self.generator = AdvancedECGGenerator(fs=Config.SAMPLING_RATE)
        self.designer = ProfessionalFilterDesigner(fs=Config.SAMPLING_RATE)
        self.analytics = AnalyticsEngine()

        # Data storage
        self.signals = {}
        self.filters = {}
        self.results = {}
        self.metrics = {}

        # Performance tracking
        self.processing_times = []

        # Setup GUI
        self.setup_gui()
        self.load_default_config()

        # Initial generation
        self.generate_signals()
        self.design_filters()
        self.run_simulation()

    def setup_gui(self):
        """Setup professional GUI layout"""

        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Controls
        left_panel = ttk.LabelFrame(main_container, text="CONTROL PANEL", padding="15")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Right panel - Visualization
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ===== SIGNAL GENERATION CONTROLS =====
        signal_frame = ttk.LabelFrame(left_panel, text="Signal Generation", padding="10")
        signal_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(signal_frame, text="Duration (s):", font=('Arial', 10)).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.duration_var = tk.DoubleVar(value=Config.DEFAULT_DURATION)
        ttk.Scale(signal_frame, from_=1, to=10, orient=tk.HORIZONTAL,
                  variable=self.duration_var, length=200).grid(row=0, column=1, pady=5, padx=10)
        ttk.Label(signal_frame, textvariable=self.duration_var, width=5).grid(row=0, column=2)

        ttk.Label(signal_frame, text="Heart Rate (BPM):", font=('Arial', 10)).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.hr_var = tk.IntVar(value=72)
        ttk.Scale(signal_frame, from_=40, to=120, orient=tk.HORIZONTAL,
                  variable=self.hr_var, length=200).grid(row=1, column=1, pady=5, padx=10)
        ttk.Label(signal_frame, textvariable=self.hr_var, width=5).grid(row=1, column=2)

        ttk.Label(signal_frame, text="Noise Level:", font=('Arial', 10)).grid(row=2, column=0, sticky=tk.W, pady=5)
        self.noise_var = tk.DoubleVar(value=0.5)
        ttk.Scale(signal_frame, from_=0, to=1, orient=tk.HORIZONTAL,
                  variable=self.noise_var, length=200).grid(row=2, column=1, pady=5, padx=10)
        ttk.Label(signal_frame, textvariable=self.noise_var, width=5).grid(row=2, column=2)

        ttk.Label(signal_frame, text="Pathology:", font=('Arial', 10)).grid(row=3, column=0, sticky=tk.W, pady=5)
        self.pathology_var = tk.StringVar(value="none")
        pathology_combo = ttk.Combobox(signal_frame, textvariable=self.pathology_var,
                                       values=["none", "tachycardia", "bradycardia", "arrhythmia"],
                                       state="readonly", width=15)
        pathology_combo.grid(row=3, column=1, columnspan=2, pady=5, sticky=tk.W)

        # ===== FILTER DESIGN CONTROLS =====
        filter_frame = ttk.LabelFrame(left_panel, text="Filter Design", padding="10")
        filter_frame.pack(fill=tk.X, pady=(0, 15))

        # HPF Controls
        ttk.Label(filter_frame, text="HPF Cutoff (Hz):", font=('Arial', 9)).grid(row=0, column=0, sticky=tk.W, pady=3)
        self.hpf_var = tk.DoubleVar(value=Config.HPF_CUTOFF)
        ttk.Scale(filter_frame, from_=0.1, to=2, orient=tk.HORIZONTAL,
                  variable=self.hpf_var, length=180).grid(row=0, column=1, pady=3)
        ttk.Label(filter_frame, textvariable=self.hpf_var, width=5).grid(row=0, column=2)

        ttk.Label(filter_frame, text="HPF Topology:", font=('Arial', 9)).grid(row=1, column=0, sticky=tk.W, pady=3)
        self.hpf_topology_var = tk.StringVar(value="butterworth")
        ttk.Combobox(filter_frame, textvariable=self.hpf_topology_var,
                     values=["butterworth", "chebyshev1", "chebyshev2", "elliptic"],
                     state="readonly", width=12).grid(row=1, column=1, columnspan=2, pady=3, sticky=tk.W)

        # LPF Controls
        ttk.Label(filter_frame, text="LPF Cutoff (Hz):", font=('Arial', 9)).grid(row=2, column=0, sticky=tk.W, pady=3)
        self.lpf_var = tk.DoubleVar(value=Config.LPF_CUTOFF)
        ttk.Scale(filter_frame, from_=50, to=300, orient=tk.HORIZONTAL,
                  variable=self.lpf_var, length=180).grid(row=2, column=1, pady=3)
        ttk.Label(filter_frame, textvariable=self.lpf_var, width=5).grid(row=2, column=2)

        ttk.Label(filter_frame, text="LPF Topology:", font=('Arial', 9)).grid(row=3, column=0, sticky=tk.W, pady=3)
        self.lpf_topology_var = tk.StringVar(value="butterworth")
        ttk.Combobox(filter_frame, textvariable=self.lpf_topology_var,
                     values=["butterworth", "chebyshev1", "chebyshev2", "elliptic"],
                     state="readonly", width=12).grid(row=3, column=1, columnspan=2, pady=3, sticky=tk.W)

        # Notch Controls
        ttk.Label(filter_frame, text="Notch Freq (Hz):", font=('Arial', 9)).grid(row=4, column=0, sticky=tk.W, pady=3)
        self.notch_var = tk.DoubleVar(value=Config.NOTCH_FREQ)
        ttk.Scale(filter_frame, from_=45, to=55, orient=tk.HORIZONTAL,
                  variable=self.notch_var, length=180).grid(row=4, column=1, pady=3)
        ttk.Label(filter_frame, textvariable=self.notch_var, width=5).grid(row=4, column=2)

        ttk.Label(filter_frame, text="Notch Q:", font=('Arial', 9)).grid(row=5, column=0, sticky=tk.W, pady=3)
        self.notch_q_var = tk.IntVar(value=Config.NOTCH_Q)
        ttk.Scale(filter_frame, from_=10, to=50, orient=tk.HORIZONTAL,
                  variable=self.notch_q_var, length=180).grid(row=5, column=1, pady=3)
        ttk.Label(filter_frame, textvariable=self.notch_q_var, width=5).grid(row=5, column=2)

        # ===== CONTROL BUTTONS =====
        button_frame = ttk.Frame(left_panel)
        button_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(button_frame, text="GENERATE", command=self.generate_signals,
                   style='Accent.TButton').pack(fill=tk.X, pady=3)
        ttk.Button(button_frame, text="DESIGN FILTERS", command=self.design_filters,
                   style='Accent.TButton').pack(fill=tk.X, pady=3)
        ttk.Button(button_frame, text="RUN SIMULATION", command=self.run_simulation,
                   style='Accent.TButton').pack(fill=tk.X, pady=3)
        ttk.Button(button_frame, text="EXPORT DATA", command=self.export_data,
                   style='Accent.TButton').pack(fill=tk.X, pady=3)
        ttk.Button(button_frame, text="GENERATE REPORT", command=self.generate_report,
                   style='Accent.TButton').pack(fill=tk.X, pady=3)

        # ===== METRICS DISPLAY =====
        metrics_frame = ttk.LabelFrame(left_panel, text="Performance Metrics", padding="10")
        metrics_frame.pack(fill=tk.X, pady=(0, 10))

        self.metrics_text = scrolledtext.ScrolledText(metrics_frame, height=15, width=30,
                                                      font=('Courier', 9))
        self.metrics_text.pack(fill=tk.BOTH, expand=True)

        # ===== STATUS BAR =====
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(left_panel, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # ===== VISUALIZATION AREA =====
        # Create notebook for tabs
        notebook = ttk.Notebook(right_panel)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Time Domain Analysis
        time_tab = ttk.Frame(notebook)
        notebook.add(time_tab, text="Time Domain")
        self.setup_time_domain_tab(time_tab)

        # Tab 2: Frequency Domain Analysis
        freq_tab = ttk.Frame(notebook)
        notebook.add(freq_tab, text="Frequency Domain")
        self.setup_frequency_domain_tab(freq_tab)

        # Tab 3: Filter Analysis
        filter_tab = ttk.Frame(notebook)
        notebook.add(filter_tab, text="Filter Analysis")
        self.setup_filter_analysis_tab(filter_tab)

        # Tab 4: Statistical Analysis
        stats_tab = ttk.Frame(notebook)
        notebook.add(stats_tab, text="Statistics")
        self.setup_statistics_tab(stats_tab)

        # Configure styles
        self.configure_styles()

    def configure_styles(self):
        """Configure ttk styles for professional appearance"""
        style = ttk.Style()
        style.theme_use('clam')

        # Configure colors
        style.configure('Accent.TButton', background='#2196F3', foreground='white',
                        font=('Arial', 10, 'bold'))
        style.map('Accent.TButton',
                  background=[('active', '#1976D2'), ('pressed', '#0D47A1')])

    def setup_time_domain_tab(self, parent):
        """Setup time domain visualization tab"""
        fig = Figure(figsize=(12, 8), dpi=100)
        self.time_ax1 = fig.add_subplot(311)
        self.time_ax2 = fig.add_subplot(312)
        self.time_ax3 = fig.add_subplot(313)

        fig.subplots_adjust(hspace=0.4)

        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.time_fig = fig
        self.time_canvas = canvas

    def setup_frequency_domain_tab(self, parent):
        """Setup frequency domain visualization tab"""
        fig = Figure(figsize=(12, 8), dpi=100)
        self.freq_ax1 = fig.add_subplot(221)
        self.freq_ax2 = fig.add_subplot(222)
        self.freq_ax3 = fig.add_subplot(223)
        self.freq_ax4 = fig.add_subplot(224)

        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.freq_fig = fig
        self.freq_canvas = canvas

    def setup_filter_analysis_tab(self, parent):
        """Setup filter analysis visualization tab"""
        fig = Figure(figsize=(12, 8), dpi=100)
        self.filter_ax1 = fig.add_subplot(221)
        self.filter_ax2 = fig.add_subplot(222)
        self.filter_ax3 = fig.add_subplot(223)
        self.filter_ax4 = fig.add_subplot(224)

        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.filter_fig = fig
        self.filter_canvas = canvas

    def setup_statistics_tab(self, parent):
        """Setup statistics display tab"""
        # Create frame for statistics display
        stats_frame = ttk.Frame(parent)
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Create text widget for statistics
        self.stats_text = scrolledtext.ScrolledText(stats_frame, wrap=tk.WORD,
                                                    font=('Courier', 10))
        self.stats_text.pack(fill=tk.BOTH, expand=True)

        # Create button to update statistics
        ttk.Button(stats_frame, text="Update Statistics",
                   command=self.update_statistics_display).pack(pady=10)

    def load_default_config(self):
        """Load default configuration"""
        self.status_var.set("Loading default configuration...")

    def generate_signals(self):
        """Generate ECG signals"""
        try:
            start_time = time.time()

            self.signals = self.generator.generate(
                duration=self.duration_var.get(),
                heart_rate=self.hr_var.get(),
                pathology=self.pathology_var.get(),
                noise_level=self.noise_var.get()
            )

            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            self.status_var.set(f"Signals generated in {processing_time:.3f}s")
            self.update_metrics_display()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate signals: {str(e)}")

    def design_filters(self):
        """Design all filters"""
        try:
            start_time = time.time()

            self.filters = {
                'hpf': self.designer.design_hpf(
                    fc=self.hpf_var.get(),
                    topology=self.hpf_topology_var.get()
                ),
                'lpf': self.designer.design_lpf(
                    fc=self.lpf_var.get(),
                    topology=self.lpf_topology_var.get()
                ),
                'notch': self.designer.design_notch(
                    f0=self.notch_var.get(),
                    Q=self.notch_q_var.get()
                )
            }

            # Analyze each filter
            self.filter_analyses = {}
            for name, design in self.filters.items():
                self.filter_analyses[name] = self.designer.analyze_filter(design)

            processing_time = time.time() - start_time
            self.status_var.set(f"Filters designed in {processing_time:.3f}s")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to design filters: {str(e)}")

    def run_simulation(self):
        """Run complete simulation"""
        if not self.signals or not self.filters:
            messagebox.showwarning("Warning", "Please generate signals and design filters first")
            return

        try:
            start_time = time.time()

            # Apply filters in cascade
            signal = self.signals['noisy'].copy()
            filtered_signals = []

            for name in ['hpf', 'lpf', 'notch']:
                signal = self.designer.apply_filter(signal, self.filters[name])
                filtered_signals.append(signal.copy())

            self.results = {
                'stage1': filtered_signals[0],
                'stage2': filtered_signals[1],
                'stage3': filtered_signals[2],
                'final': filtered_signals[2]
            }

            # Calculate metrics
            self.calculate_metrics()

            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            self.status_var.set(f"Simulation completed in {processing_time:.3f}s")

            # Update visualizations
            self.update_time_domain_plots()
            self.update_frequency_domain_plots()
            self.update_filter_analysis_plots()
            self.update_statistics_display()

        except Exception as e:
            messagebox.showerror("Error", f"Simulation failed: {str(e)}")

    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        if not self.signals or not self.results:
            return

        clean = self.signals['clean']
        noisy = self.signals['noisy']
        final = self.results['final']

        self.metrics = {
            'input_snr': self.analytics.calculate_snr(clean, noisy),
            'output_snr': self.analytics.calculate_snr(clean, final),
            'snr_improvement': 0,
            'input_thd': self.analytics.calculate_thd(noisy, Config.SAMPLING_RATE),
            'output_thd': self.analytics.calculate_thd(final, Config.SAMPLING_RATE),
            'input_rms': self.analytics.calculate_rms(noisy),
            'output_rms': self.analytics.calculate_rms(final),
            'crest_factor': self.analytics.calculate_crest_factor(final),
            'processing_time': self.processing_times[-1] if self.processing_times else 0
        }

        self.metrics['snr_improvement'] = self.metrics['output_snr'] - self.metrics['input_snr']

    def update_time_domain_plots(self):
        """Update time domain visualizations"""
        if not self.signals or not self.results:
            return

        time = self.signals['time']
        clean = self.signals['clean']
        noisy = self.signals['noisy']
        final = self.results['final']
        stage1 = self.results['stage1']
        stage2 = self.results['stage2']

        # Clear axes
        self.time_ax1.clear()
        self.time_ax2.clear()
        self.time_ax3.clear()

        # Plot 1: Original vs Filtered
        self.time_ax1.plot(time, clean, color=Config.COLORS['clean'],
                           alpha=0.7, label='Clean ECG', linewidth=1)
        self.time_ax1.plot(time, noisy, color=Config.COLORS['noisy'],
                           alpha=0.5, label='Noisy ECG', linewidth=0.8)
        self.time_ax1.plot(time, final, color=Config.COLORS['filtered'],
                           label='Filtered ECG', linewidth=1.5)
        self.time_ax1.set_xlabel('Time (s)', fontsize=10)
        self.time_ax1.set_ylabel('Amplitude (mV)', fontsize=10)
        self.time_ax1.set_title('ECG Signal: Clean, Noisy, and Filtered', fontsize=12, fontweight='bold')
        self.time_ax1.legend(loc='upper right')
        self.time_ax1.grid(True, alpha=0.3)

        # Plot 2: Filter Stages
        self.time_ax2.plot(time, stage1, color=Config.COLORS['hpf'],
                           label='After HPF', linewidth=1.5)
        self.time_ax2.plot(time, stage2, color=Config.COLORS['lpf'],
                           label='After LPF', linewidth=1.5)
        self.time_ax2.plot(time, final, color=Config.COLORS['notch'],
                           label='After Notch', linewidth=1.5)
        self.time_ax2.set_xlabel('Time (s)', fontsize=10)
        self.time_ax2.set_ylabel('Amplitude (mV)', fontsize=10)
        self.time_ax2.set_title('Signal at Each Filter Stage', fontsize=12, fontweight='bold')
        self.time_ax2.legend(loc='upper right')
        self.time_ax2.grid(True, alpha=0.3)

        # Plot 3: Residual Noise
        residual = noisy - final
        self.time_ax3.plot(time, residual, color='#757575', linewidth=1)
        self.time_ax3.fill_between(time, 0, residual, alpha=0.3, color='#757575')
        self.time_ax3.set_xlabel('Time (s)', fontsize=10)
        self.time_ax3.set_ylabel('Amplitude (mV)', fontsize=10)
        self.time_ax3.set_title('Residual Noise (Noisy - Filtered)', fontsize=12, fontweight='bold')
        self.time_ax3.grid(True, alpha=0.3)

        # Update canvas
        self.time_fig.tight_layout()
        self.time_canvas.draw()

    def update_frequency_domain_plots(self):
        """Update frequency domain visualizations"""
        if not self.signals or not self.results:
            return

        fs = Config.SAMPLING_RATE
        noisy = self.signals['noisy']
        final = self.results['final']

        # Calculate FFT
        n = len(noisy)
        freqs = np.fft.rfftfreq(n, 1 / fs)
        fft_noisy = np.abs(np.fft.rfft(noisy))
        fft_filtered = np.abs(np.fft.rfft(final))

        # Clear axes
        self.freq_ax1.clear()
        self.freq_ax2.clear()
        self.freq_ax3.clear()
        self.freq_ax4.clear()

        # Plot 1: Frequency Spectrum
        self.freq_ax1.semilogy(freqs, fft_noisy, color=Config.COLORS['noisy'],
                               alpha=0.7, label='Noisy')
        self.freq_ax1.semilogy(freqs, fft_filtered, color=Config.COLORS['filtered'],
                               linewidth=1.5, label='Filtered')
        self.freq_ax1.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50 Hz')
        self.freq_ax1.set_xlabel('Frequency (Hz)', fontsize=10)
        self.freq_ax1.set_ylabel('Magnitude', fontsize=10)
        self.freq_ax1.set_title('Frequency Spectrum Comparison', fontsize=12, fontweight='bold')
        self.freq_ax1.legend(loc='upper right')
        self.freq_ax1.grid(True, alpha=0.3)
        self.freq_ax1.set_xlim(0, 200)

        # Plot 2: Power Spectral Density
        f, Pxx_noisy = signal.welch(noisy, fs, nperseg=1024)
        f, Pxx_filtered = signal.welch(final, fs, nperseg=1024)

        self.freq_ax2.semilogy(f, Pxx_noisy, color=Config.COLORS['noisy'],
                               alpha=0.7, label='Noisy')
        self.freq_ax2.semilogy(f, Pxx_filtered, color=Config.COLORS['filtered'],
                               linewidth=1.5, label='Filtered')
        self.freq_ax2.set_xlabel('Frequency (Hz)', fontsize=10)
        self.freq_ax2.set_ylabel('PSD', fontsize=10)
        self.freq_ax2.set_title('Power Spectral Density', fontsize=12, fontweight='bold')
        self.freq_ax2.legend(loc='upper right')
        self.freq_ax2.grid(True, alpha=0.3)
        self.freq_ax2.set_xlim(0, 200)

        # Plot 3: Spectral Subtraction
        spectral_diff = fft_noisy - fft_filtered
        self.freq_ax3.fill_between(freqs, 0, spectral_diff[:len(freqs)],
                                   color='#FF5252', alpha=0.5)
        self.freq_ax3.plot(freqs, spectral_diff[:len(freqs)], color='#D32F2F', linewidth=1)
        self.freq_ax3.set_xlabel('Frequency (Hz)', fontsize=10)
        self.freq_ax3.set_ylabel('Removed Power', fontsize=10)
        self.freq_ax3.set_title('Spectral Subtraction (Removed Noise)', fontsize=12, fontweight='bold')
        self.freq_ax3.grid(True, alpha=0.3)
        self.freq_ax3.set_xlim(0, 200)

        # Plot 4: Frequency Response
        if hasattr(self, 'filter_analyses'):
            for name, analysis in self.filter_analyses.items():
                color = Config.COLORS.get(name, 'black')
                self.freq_ax4.semilogx(analysis['frequency'], analysis['magnitude'],
                                       color=color, linewidth=2, label=name.upper())

        self.freq_ax4.set_xlabel('Frequency (Hz)', fontsize=10)
        self.freq_ax4.set_ylabel('Magnitude (dB)', fontsize=10)
        self.freq_ax4.set_title('Filter Frequency Responses', fontsize=12, fontweight='bold')
        self.freq_ax4.legend(loc='upper right')
        self.freq_ax4.grid(True, which='both', alpha=0.3)
        self.freq_ax4.set_xlim(0.1, 500)
        self.freq_ax4.set_ylim(-80, 10)

        # Update canvas
        self.freq_fig.tight_layout()
        self.freq_canvas.draw()

    def update_filter_analysis_plots(self):
        """Update filter analysis visualizations"""
        if not hasattr(self, 'filter_analyses'):
            return

        # Clear axes
        self.filter_ax1.clear()
        self.filter_ax2.clear()
        self.filter_ax3.clear()
        self.filter_ax4.clear()

        colors = [Config.COLORS['hpf'], Config.COLORS['lpf'], Config.COLORS['notch']]

        # Plot 1: Magnitude Response
        for (name, analysis), color in zip(self.filter_analyses.items(), colors):
            self.filter_ax1.semilogx(analysis['frequency'], analysis['magnitude'],
                                     color=color, linewidth=2, label=name.upper())
        self.filter_ax1.set_xlabel('Frequency (Hz)', fontsize=10)
        self.filter_ax1.set_ylabel('Magnitude (dB)', fontsize=10)
        self.filter_ax1.set_title('Filter Magnitude Responses', fontsize=12, fontweight='bold')
        self.filter_ax1.legend(loc='upper right')
        self.filter_ax1.grid(True, which='both', alpha=0.3)
        self.filter_ax1.set_xlim(0.1, 500)
        self.filter_ax1.set_ylim(-80, 5)

        # Plot 2: Phase Response
        for (name, analysis), color in zip(self.filter_analyses.items(), colors):
            self.filter_ax2.semilogx(analysis['frequency'], analysis['phase'],
                                     color=color, linewidth=2, label=name.upper())
        self.filter_ax2.set_xlabel('Frequency (Hz)', fontsize=10)
        self.filter_ax2.set_ylabel('Phase (degrees)', fontsize=10)
        self.filter_ax2.set_title('Filter Phase Responses', fontsize=12, fontweight='bold')
        self.filter_ax2.legend(loc='upper right')
        self.filter_ax2.grid(True, which='both', alpha=0.3)
        self.filter_ax2.set_xlim(0.1, 500)

        # Plot 3: Group Delay
        for (name, analysis), color in zip(self.filter_analyses.items(), colors):
            f_gd, gd = analysis['group_delay']
            self.filter_ax3.semilogx(f_gd, gd, color=color, linewidth=2, label=name.upper())
        self.filter_ax3.set_xlabel('Frequency (Hz)', fontsize=10)
        self.filter_ax3.set_ylabel('Group Delay (samples)', fontsize=10)
        self.filter_ax3.set_title('Filter Group Delay', fontsize=12, fontweight='bold')
        self.filter_ax3.legend(loc='upper right')
        self.filter_ax3.grid(True, which='both', alpha=0.3)
        self.filter_ax3.set_xlim(0.1, 500)

        # Plot 4: Pole-Zero Plot
        self.filter_ax4.text(0.5, 0.5, 'Pole-Zero Analysis\n(Click to expand)',
                             horizontalalignment='center', verticalalignment='center',
                             transform=self.filter_ax4.transAxes, fontsize=12)
        self.filter_ax4.set_xlabel('Real Part', fontsize=10)
        self.filter_ax4.set_ylabel('Imaginary Part', fontsize=10)
        self.filter_ax4.set_title('Filter Pole-Zero Map', fontsize=12, fontweight='bold')
        self.filter_ax4.grid(True, alpha=0.3)

        # Update canvas
        self.filter_fig.tight_layout()
        self.filter_canvas.draw()

    def update_metrics_display(self):
        """Update metrics display in control panel"""
        if not hasattr(self, 'metrics_text'):
            return

        stats = self.generator.calculate_statistics()

        metrics_text = "=" * 40 + "\n"
        metrics_text += "SIGNAL STATISTICS\n"
        metrics_text += "=" * 40 + "\n\n"

        if stats:
            metrics_text += f"Clean Signal:\n"
            metrics_text += f"  Mean: {stats['mean_clean']:.4f} mV\n"
            metrics_text += f"  Std Dev: {stats['std_clean']:.4f} mV\n"
            metrics_text += f"  Max: {stats['max_clean']:.4f} mV\n"
            metrics_text += f"  Min: {stats['min_clean']:.4f} mV\n"
            metrics_text += f"  RMS: {stats['rms_clean']:.4f} mV\n"
            metrics_text += f"  Dynamic Range: {stats['dynamic_range']:.4f} mV\n\n"

            metrics_text += f"Noisy Signal:\n"
            metrics_text += f"  Mean: {stats['mean_noisy']:.4f} mV\n"
            metrics_text += f"  Std Dev: {stats['std_noisy']:.4f} mV\n"
            metrics_text += f"  SNR: {stats['snr']:.2f} dB\n\n"

            metrics_text += f"Signal Parameters:\n"
            metrics_text += f"  Duration: {self.duration_var.get():.1f} s\n"
            metrics_text += f"  Heart Rate: {self.hr_var.get()} BPM\n"
            metrics_text += f"  Noise Level: {self.noise_var.get():.2f}\n"
            metrics_text += f"  Pathology: {self.pathology_var.get()}\n"

        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(1.0, metrics_text)
        self.metrics_text.configure(state='disabled')

    def update_statistics_display(self):
        """Update statistics display tab"""
        if not hasattr(self, 'stats_text'):
            return

        stats_text = "=" * 60 + "\n"
        stats_text += "COMPREHENSIVE PERFORMANCE ANALYSIS\n"
        stats_text += "=" * 60 + "\n\n"

        if self.metrics:
            stats_text += "PERFORMANCE METRICS:\n"
            stats_text += "-" * 40 + "\n"
            stats_text += f"Input SNR: {self.metrics['input_snr']:.2f} dB\n"
            stats_text += f"Output SNR: {self.metrics['output_snr']:.2f} dB\n"
            stats_text += f"SNR Improvement: {self.metrics['snr_improvement']:.2f} dB\n"
            stats_text += f"Input THD: {self.metrics['input_thd']:.2f}%\n"
            stats_text += f"Output THD: {self.metrics['output_thd']:.2f}%\n"
            stats_text += f"Input RMS: {self.metrics['input_rms']:.4f} mV\n"
            stats_text += f"Output RMS: {self.metrics['output_rms']:.4f} mV\n"
            stats_text += f"Crest Factor: {self.metrics['crest_factor']:.2f}\n"
            stats_text += f"Processing Time: {self.metrics['processing_time']:.3f} s\n\n"

        if hasattr(self, 'filters'):
            stats_text += "FILTER SPECIFICATIONS:\n"
            stats_text += "-" * 40 + "\n"
            for name, design in self.filters.items():
                stats_text += f"\n{name.upper()}:\n"
                stats_text += f"  Type: {design['type']}\n"
                stats_text += f"  Topology: {design['topology']}\n"
                if 'fc_desired' in design:
                    stats_text += f"  Cutoff: {design['fc_desired']:.1f} Hz\n"
                if 'f0_desired' in design:
                    stats_text += f"  Center: {design['f0_desired']:.1f} Hz\n"
                    stats_text += f"  Q Factor: {design['Q_desired']}\n"

        stats_text += "\n" + "=" * 60 + "\n"
        stats_text += f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        stats_text += "=" * 60

        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)

    def export_data(self):
        """Export all data to files"""
        if not self.signals or not self.results:
            messagebox.showwarning("Warning", "No data to export. Please run simulation first.")
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"ECG_Analysis_{timestamp}"

            # Save signals to CSV
            data_dict = {
                'time': self.signals['time'],
                'clean': self.signals['clean'],
                'noisy': self.signals['noisy'],
                'stage1_hpf': self.results['stage1'],
                'stage2_lpf': self.results['stage2'],
                'stage3_notch': self.results['final']
            }

            df = pd.DataFrame(data_dict)
            csv_filename = f"{base_filename}.csv"
            df.to_csv(csv_filename, index=False)

            # Save parameters to JSON
            params = {
                'project': Config.PROJECT_NAME,
                'version': Config.VERSION,
                'timestamp': timestamp,
                'signal_parameters': {
                    'duration': self.duration_var.get(),
                    'heart_rate': self.hr_var.get(),
                    'noise_level': self.noise_var.get(),
                    'pathology': self.pathology_var.get()
                },
                'filter_parameters': {
                    'hpf': {
                        'cutoff': self.hpf_var.get(),
                        'topology': self.hpf_topology_var.get()
                    },
                    'lpf': {
                        'cutoff': self.lpf_var.get(),
                        'topology': self.lpf_topology_var.get()
                    },
                    'notch': {
                        'frequency': self.notch_var.get(),
                        'Q': self.notch_q_var.get()
                    }
                },
                'performance_metrics': self.metrics
            }

            json_filename = f"{base_filename}_params.json"
            with open(json_filename, 'w') as f:
                json.dump(params, f, indent=4)

            # Save plots
            for fig_name, fig in [('time', self.time_fig), ('freq', self.freq_fig), ('filter', self.filter_fig)]:
                if fig:
                    fig.savefig(f"{base_filename}_{fig_name}.png", dpi=150, bbox_inches='tight')

            self.status_var.set(f"Data exported to {base_filename}* files")
            messagebox.showinfo("Export Successful",
                                f"All data exported successfully!\n\nFiles created:\n"
                                f"• {csv_filename}\n"
                                f"• {json_filename}\n"
                                f"• {base_filename}_time.png\n"
                                f"• {base_filename}_freq.png\n"
                                f"• {base_filename}_filter.png")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")

    def generate_report(self):
        """Generate comprehensive project report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"ECG_Project_Report_{timestamp}.html"

            report_content = self._create_report_content()

            with open(report_filename, 'w') as f:
                f.write(report_content)

            self.status_var.set(f"Report generated: {report_filename}")
            messagebox.showinfo("Report Generated",
                                f"Professional report generated successfully!\n\n"
                                f"File: {report_filename}")

        except Exception as e:
            messagebox.showerror("Report Error", f"Failed to generate report: {str(e)}")

    def _create_report_content(self):
        """Create HTML report content"""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{Config.PROJECT_NAME} - Professional Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{Config.PROJECT_NAME}</h1>
        <h2>Professional Analysis Report</h2>
        <p>Version: {Config.VERSION} | Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Author: {Config.AUTHOR}</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <p>This report presents a comprehensive analysis of the ECG Signal Conditioning System,
           demonstrating the effectiveness of multi-stage analog filtering in biomedical signal processing.
           The system successfully removes baseline wander, high-frequency noise, and power line interference
           while preserving essential cardiac information.</p>
    </div>

    <div class="section">
        <h2>Performance Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>SNR Improvement</h3>
                <div class="metric-value">{self.metrics.get('snr_improvement', 0):.2f} dB</div>
                <p>Signal Quality Enhancement</p>
            </div>
            <div class="metric-card">
                <h3>Processing Time</h3>
                <div class="metric-value">{self.metrics.get('processing_time', 0):.3f} s</div>
                <p>Real-time Performance</p>
            </div>
            <div class="metric-card">
                <h3>THD Reduction</h3>
                <div class="metric-value">{abs(self.metrics.get('input_thd', 0) - self.metrics.get('output_thd', 0)):.2f}%</div>
                <p>Distortion Improvement</p>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Filter Specifications</h2>
        <table>
            <tr>
                <th>Filter</th>
                <th>Type</th>
                <th>Cutoff/Center</th>
                <th>Topology</th>
                <th>Purpose</th>
            </tr>
            <tr>
                <td>Stage 1</td>
                <td>High-Pass</td>
                <td>{self.hpf_var.get():.1f} Hz</td>
                <td>{self.hpf_topology_var.get().title()}</td>
                <td>Remove baseline wander</td>
            </tr>
            <tr>
                <td>Stage 2</td>
                <td>Low-Pass</td>
                <td>{self.lpf_var.get():.1f} Hz</td>
                <td>{self.lpf_topology_var.get().title()}</td>
                <td>Attenuate high-frequency noise</td>
            </tr>
            <tr>
                <td>Stage 3</td>
                <td>Notch</td>
                <td>{self.notch_var.get():.1f} Hz</td>
                <td>Twin-T</td>
                <td>Remove power line interference</td>
            </tr>
        </table>
    </div>

    <div class="section">
        <h2>Signal Parameters</h2>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
                <th>Description</th>
            </tr>
            <tr>
                <td>Duration</td>
                <td>{self.duration_var.get():.1f} seconds</td>
                <td>Analysis time window</td>
            </tr>
            <tr>
                <td>Heart Rate</td>
                <td>{self.hr_var.get()} BPM</td>
                <td>Cardiac rhythm</td>
            </tr>
            <tr>
                <td>Noise Level</td>
                <td>{self.noise_var.get():.2f}</td>
                <td>Signal contamination factor</td>
            </tr>
            <tr>
                <td>Pathology</td>
                <td>{self.pathology_var.get().title()}</td>
                <td>Simulated cardiac condition</td>
            </tr>
        </table>
    </div>

    <div class="footer">
        <p>© {datetime.now().year} Biomedical Signal Processing Laboratory</p>
        <p>This report was automatically generated by the ECG Signal Conditioning System.</p>
        <p>For educational and research purposes only.</p>
    </div>
</body>
</html>
"""


# ============================================================================
# MAIN APPLICATION LAUNCHER
# ============================================================================

def main():
    """Main application entry point"""
    print("\n" + "=" * 70)
    print(" " * 20 + "ECG SIGNAL CONDITIONING SYSTEM")
    print(" " * 15 + "Professional Biomedical Analytics Platform")
    print("=" * 70)
    print(f"\nVersion: {Config.VERSION}")
    print(f"Date: {Config.DATE}")
    print(f"Author: {Config.AUTHOR}")
    print("\n" + "=" * 70)

    # Create and run application
    root = tk.Tk()
    app = ProfessionalDashboard(root)

    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')

    # Start main loop
    root.mainloop()
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
        self.overlap = 200
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
