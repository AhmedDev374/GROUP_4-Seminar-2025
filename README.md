# 🦾 Deep Learning–Enhanced EMG Prosthetic Control Simulator

This project is a **clinical-style simulation system** for EMG-based prosthetic hand control.
It demonstrates how **AI, signal processing, and real-time visualization** can be used to classify hand gestures from EMG signals.

The system includes:

* Synthetic EMG signal generation
* Real-time gesture prediction
* Performance metrics (accuracy, latency, confidence)
* Interactive graphical user interface (GUI)
* Multiple clinical scenarios (noise, fatigue, electrode shift)

---

## 📁 Project Structure

```
GROUP_4-Seminar-2025/
│
├── main.py              # Main application (run this file)
├── requirements.txt     # Required Python libraries
├── README.md            # Project documentation
└── .gitignore           # Ignored files (venv, cache, etc.)
```

---

## 🛠️ Requirements

* **Python 3.9 – 3.11** (recommended)
* Operating System: Windows / Linux / macOS

All required Python libraries are listed in `requirements.txt`.

---

## 🚀 How to Run the Project (Step by Step)

### 1️⃣ Clone the repository

```bash
git clone https://github.com/AhmedDev374/GROUP_4-Seminar-2025.git
cd GROUP_4-Seminar-2025
```

---

### 2️⃣ Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

---

### 3️⃣ Install required libraries

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the application

```bash
python main.py
```

🎉 The GUI window will open and the simulation will start.

---

## 🎮 How to Use the Simulator

* **Start / Stop Simulation** using the control panel
* Select **hand gestures** (Rest, Open, Close, etc.)
* Choose **clinical scenarios**:

  * Normal
  * Noisy
  * Fatigue
  * Electrode Shift
  * Long-Term Use
* Observe:

  * EMG signal behavior
  * Gesture prediction distribution
  * Latency and confidence
  * Simulated AI attention weights

---

## 🧠 AI Model Overview

The system uses a **hybrid deep learning architecture**:

* Convolutional Neural Networks (CNN)
* Bidirectional LSTM layers
* Feature extraction from EMG windows
* Softmax classification for gestures

> ⚠️ Note: This is a **simulation for educational and research purposes**, not a medical device.

---

## 👥 Team & Course

* **Course / Seminar:** GROUP 4 – Seminar 2025
* **Author:** AhmedDev374
* **Field:** Biomedical Engineering / AI / Signal Processing

---

## 📜 License

This project is for **educational and academic use**.
* Add screenshots section
* Add **installation errors FAQ**

Just tell me 👍
