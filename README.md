# 🔍 SearchAlike — Triplet Learning–Based Image Similarity Search Engine

SearchAlike is an **end-to-end image similarity search system** built around **Triplet Loss–based metric learning**, enabling semantic image comparison rather than naive pixel matching.

Users can upload an image and instantly retrieve **visually similar images** from a dataset using **deep embeddings + FAISS vector search**.
The system is fully functional using **pretrained triplet embeddings** and does **not require training**.

---

## 🚀 Project Status

* ✔ Triplet Network–based embedding model
* ✔ Pretrained model included
* ✔ Precomputed embeddings & FAISS index
* ✔ Fully runnable Flask web application
* ✔ No training required

---

## 🎥 Project Demo

▶ **YouTube Execution & Walkthrough:**
👉 *[https://youtu.be/fWxtLV_KUgo]*

---

## 🧠 Key Features (Core Strengths)

* **Triplet Loss–based metric learning (key feature)**
* Learns **relative similarity** instead of class labels
* Deep image embeddings for semantic similarity
* **FAISS-powered fast nearest-neighbor search**
* Pretrained embeddings for instant inference
* Flask-based image upload & result visualization
* Clean separation of ML, search, and web layers

> This project focuses on **metric learning**, not classification.

---

## 🏗️ System Architecture

```
            ┌──────────────┐
            │  User Upload │
            └───────┬──────┘
                    │
                    ▼
          ┌───────────────────┐
          │   Flask Web App   │
          │ (HTML / CSS / JS) │
          └─────────┬─────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │ Triplet Embedding Model  │
        │   (Metric Learning)      │
        └─────────┬───────────────┘
                  │
                  ▼
        ┌─────────────────────────┐
        │    FAISS Vector Index    │
        │ (Similarity Search)      │
        └─────────┬───────────────┘
                  │
                  ▼
        ┌─────────────────────────┐
        │   Image Dataset Store   │
        └─────────────────────────┘
```

---

## 📁 Project Architecture

| Folder / File      | Description                      |
| ------------------ | -------------------------------- |
| `app.py`           | Flask application entry point    |
| `src/`             | Core ML & similarity logic       |
| `checkpoints/`     | Pretrained triplet model weights |
| `data/images/`     | Image dataset                    |
| `data/embeddings/` | Triplet embeddings & FAISS index |
| `static/`          | CSS, JS, icons, uploads          |
| `templates/`       | HTML templates                   |
| `requirements.txt` | Project dependencies             |

---

## 🧠 Model Layer (Triplet Learning)

| Path                       | Purpose                           |
| -------------------------- | --------------------------------- |
| `src/triplet_network.py`   | Triplet embedding network         |
| `src/triplet_dataset.py`   | Anchor–Positive–Negative sampling |
| `src/feature_extractor.py` | Embedding extraction              |
| `checkpoints/*.pth`        | Trained triplet model             |

**Why Triplet Loss?**
Because similarity is learned via **relative distance comparisons**, making the system robust to unseen classes.

---

## 🔎 Inference & Similarity Search

| Path                       | Purpose                       |
| -------------------------- | ----------------------------- |
| `src/similarity_search.py` | FAISS-based similarity engine |
| `data/embeddings/*.npy`    | Image feature vectors         |
| `data/embeddings/*.bin`    | FAISS index                   |
| `data/embeddings/*.pkl`    | Image path mappings           |

---

## 🚀 System Overview

| Layer         | Function                      |
| ------------- | ----------------------------- |
| Triplet Model | Learns semantic similarity    |
| FAISS         | Fast vector similarity search |
| Backend       | Handles uploads & inference   |
| Frontend      | Displays similar images       |

---

## 📦 Installation

```bash
git clone https://github.com/Vedant2004X/SearchAlike.git
cd SearchAlike
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

Upload an image to retrieve visually similar results.

---

## ⚠️ Limitations (Transparent)

* Dataset size is limited
* CPU-based FAISS
* UI is minimal (ML-focused system)

These do **not** affect the core demonstration of metric learning.

---

## 🔮 Future Improvements

* GPU-accelerated FAISS
* Larger datasets
* Stronger pretrained backbones
* Cloud deployment
* Similarity score visualization
