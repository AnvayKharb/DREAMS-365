
#  Background Scene Classifier (DREAMS Prototype)

A **local prototype implementation** of background scene classification using the Places365 model, built to validate environment understanding for the DREAMS pipeline.

>  This is an experimental setup created to evaluate accuracy and feasibility.
> Integration into the main DREAMS system will be done during the GSoC coding period.

---

##  Purpose

This module was developed to answer a critical question for DREAMS:

> *Can we reliably extract environmental context (place/scene) from user-uploaded images without relying on GPS?*

### Result

* The model produced **highly accurate and stable predictions** across diverse test images
* Demonstrated strong potential for integration into the DREAMS ingestion pipeline

---

##  Local Setup (Prototype)

### 1. Install Dependencies

```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Run the Server

```bash
python app.py
```

Server runs on:
👉 **[http://localhost:3000](http://localhost:3000)**

---

### 3. Use the Interface

1. Open **[http://localhost:3000](http://localhost:3000)**
2. Upload an image (drag & drop supported)
3. Click **Analyze Image**
4. View predicted scene categories with confidence scores

---

##  What It Does

* Classifies images into **365 scene categories** (Places365)
* Returns top predictions with confidence scores
* Provides environmental context such as:

  * indoor vs outdoor
  * social vs isolated spaces
  * structured vs natural environments

---

##  Technical Stack

* **Backend:** Flask
* **Model:** Places365 (ResNet18)
* **Framework:** PyTorch
* **Frontend:** HTML/CSS/JS

---

##  Project Structure

```
├── app.py
├── background_place_detector.py
├── templates/
│   └── index.html
├── uploads/
├── requirements.txt
└── README.md
```

---

##  API

### POST `/api/detect`

Upload image and get scene predictions

```bash
curl -X POST -F "file=@image.jpg" http://localhost:3000/api/detect
```

---

##  Observations (Local Validation)

* Strong performance on:

  * outdoor environments (parks, streets, landscapes)
  * structured indoor scenes (rooms, buildings)
* Stable confidence scores across varied inputs
* Works reliably without additional fine-tuning

---

## Relation to DREAMS

This prototype directly maps to **Phase 1: Scene Classification Backbone** in the DREAMS proposal.

### Planned Integration (GSoC)

* Embed into DREAMS ingestion pipeline
* Map Places365 outputs → DREAMS-specific categories
* Add CLIP fallback for low-confidence predictions
* Store `scene_type` alongside emotion and caption data
* Use scene context in similarity and recovery analysis

---

##  Current Scope

* Local prototype only
* Not yet integrated into DREAMS backend
* No database storage or pipeline connection
* No privacy-layer integration yet

---

##  Note

This implementation was created purely for **validation purposes**.
It confirms that scene classification is **feasible, reliable, and ready for integration**.

Full integration, optimization, and pipeline alignment will be completed during the GSoC coding period.

---

##  Stop Server

```bash
Ctrl + C
```
