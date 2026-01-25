# 🌸 Flower Image Classification using CNN

This project implements a **Convolutional Neural Network (CNN)** to classify images of flowers into five distinct categories. Built with **TensorFlow** and **Keras**, it is optimized for **Google Colab** using **GPU (T4)** acceleration.

---

## 📁 Dataset Information
The project utilizes the official **TensorFlow Flower Photos dataset**.
* **Total Images:** ~3,670
* **Categories:** * **Daisy**
  * **Dandelion**
  * **Roses**
  * **Sunflowers**
  * **Tulips**

The dataset is programmatically downloaded and extracted from Google's public storage during execution.

---

## 🛠️ Technology Stack
* **Framework:** **TensorFlow / Keras**
* **Image Processing:** **OpenCV** (`cv2`), **Pillow** (`PIL`)
* **Data Handling:** **NumPy**, **Pathlib**
* **Visualization:** **Matplotlib**

---

## 🚀 Key Features
* **Automated Pipeline:** Automatic downloading and extraction of `.tgz` datasets.
* **Data EDA:** Visual verification of classes using directory mapping and sample plotting.
* **Hardware Optimized:** Pre-configured for **NVIDIA T4 GPU** acceleration in Colab.
* **Efficient Pre-processing:** High-speed image-to-array conversion via OpenCV for training readiness.

---

## 📊 Project Workflow
1.  **Library Imports:** Loading essential tools like `tensorflow`, `numpy`, and `cv2`.
2.  **Data Acquisition:** Programmatically fetching the `.tgz` flower dataset.
3.  **Data Exploration:** Using `Path.iterdir()` to map categories and verify image counts.
4.  **Pre-processing:** Scaling, resizing, and reshaping image arrays for the model.
5.  **CNN Architecture:** Building a deep learning model with convolutional, pooling, and dense layers.

---

## 💻 Installation & Usage

### 1. Prerequisites
Open this project in **Google Colab** and ensure the **Runtime type** is set to **GPU**.

### 2. Install Dependencies
pip install tensorflow pillow matplotlib opencv-python numpy

### **3. Running the Project**

* **Clone the Repository:**
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
  
* **Launch the Notebook:** Open `Flowers_dataset_CNN.ipynb`.
* **Execute Cells:** Run the cells in sequence. The notebook will:
    1. Initialize the environment and install **Pillow**.
    2. Download the **flower_photos.tgz** dataset.
    3. Map and display flower categories.
    4. Build, train, and evaluate the **CNN model**.



---

### **📁 Project Structure**

```text
├── Flowers_dataset_CNN.ipynb  # Main Notebook containing code & analysis

└── README.md                  # Project documentation
