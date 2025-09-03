
# 🧑‍⚕️ Skin Lesion Classification

A deep learning project for **automatic skin lesion classification** using the ISIC dataset.
It provides:

* 📦 **Training pipeline** with TensorFlow/Keras (VGG19 backbone, fine-tuning, class balancing).
* 🌐 **FastAPI REST API** for serving predictions.
* 🖥 **Desktop GUI (PyQt5)** for interactive image prediction.

---

## 📂 Project Structure

```
thanhdinhbao-skin-classification/
│── api.py         # FastAPI server for prediction
│── main.py        # PyQt5 desktop GUI
│── train.py       # Training pipeline (VGG19, fine-tuning)
│── labels.txt     # Default labels (9 ISIC classes)
```

---

## 🧑‍🏫 Dataset

* Source: [ISIC - Skin Cancer ISIC: The International Skin Imaging Collaboration](https://www.isic-archive.com/)
* Structure expected:

```
Skin cancer ISIC The International Skin Imaging Collaboration/
│── Train/
│    ├── actinic keratosis/
│    ├── basal cell carcinoma/
│    └── ...
│── Test/
│── Val/   # Auto-created if missing
```

* **Classes (9):**

  1. actinic keratosis
  2. basal cell carcinoma
  3. dermatofibroma
  4. melanoma
  5. nevus
  6. pigmented benign keratosis
  7. seborrheic keratosis
  8. squamous cell carcinoma
  9. vascular lesion

---

## ⚙️ Installation

```bash
git clone https://github.com/thanhdinhbao/skin-classification.git
cd skin-classification/thanhdinhbao-skin-classification

# Create environment
python -m venv .venv
source .venv/bin/activate   # or: .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

Example minimal `requirements.txt`:

```
tensorflow
numpy
pillow
matplotlib
fastapi
uvicorn
python-multipart
PyQt5
```

---

## 🚀 Training

Run the training script:

```bash
python train.py
```

* Uses **VGG19** pretrained on ImageNet.
* Training strategy:

  1. **Phase 1**: Warm-up (freeze backbone, train dense head).
  2. **Phase 2**: Fine-tune top convolutional blocks.
* Augmentation: Random flip, rotation, zoom.
* Class weights to handle imbalance.
* Early stopping, LR scheduling.

Output:

* Best checkpoints:

  * `vgg19_isic_best_warmup.h5`
  * `vgg19_isic_best_finetune.h5`
* Final model:

  * `vgg19_skin_lesion_model_finetuned.h5`

---

## 🌐 FastAPI REST API

Start server:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Endpoints

* `GET /health` → server status
* `GET /meta` → model & labels info
* `POST /predict` (multipart, file upload)
* `POST /predict_base64` (JSON, base64 image)

Example request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@example.jpg" \
  -F "preprocess=rescale" \
  -F "top_k=3"
```

---

## 🖥 PyQt5 Desktop GUI

Run:

```bash
python main.py
```

Features:

* Load `.h5` / `.keras` / SavedModel models.
* Load custom labels (`labels.txt`).
* Drag & drop or browse images.
* Choose preprocessing (`rescale` / `ResNet50`).
* Show **Top-1 prediction** with confidence bar.

---

## 📊 Results & Evaluation

After training, the script evaluates on **TEST set**:

```bash
Evaluating on TEST set...
Test metrics: {'loss': ..., 'accuracy': ...}
```

Training & validation curves are plotted (Accuracy, Loss).

---

## 📜 License

MIT License.
Feel free to use, modify, and share.

---

## ✨ Acknowledgements

* [ISIC Archive](https://www.isic-archive.com/)
* TensorFlow / Keras
* FastAPI
* PyQt5

---
