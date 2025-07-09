
# 📦 How to Prepare Datasets for ControlMLLM

This document provides detailed instructions for preparing datasets required by **ControlMLLM++**, including for the **ROC**, **RTC**, and **Reference Description (RD)** tasks.

We recommend placing all datasets under a root folder, e.g., `$DATA/`, for consistency and ease of path management. You may create symbolic links to reuse existing dataset files.

---

## 🧠 Task Overview

- **ROC (Referring Object Classification)**  
  Given an image and a region, the model classifies the type of the object referred by the region.

- **RTC (Referring Text Classification)**  
  Given an image and a text region, the model classifies or interprets the text content shown in the image.

- **RD (Reference Description)**  
  The model is asked to **generate** a natural language description of a referred region, aiming at free-form expression and understanding.

In all tasks, we focus on **single-region prompts** to keep input precise and interpretable.

---

## 📁 Directory Structure Overview

```

\$DATA/
├── ROC/
│   ├── question_roc.json
│   └── LVIS/
│       ├── image/
│       └── mask/
├── RTC/
│   ├── question_rtc.json
│   └── COCO-Text/
│       ├── image/
│       └── mask/
├── RD/
│   ├── RefCOCOg/
│   │   ├── refcocog.json
│   │   └── COCO2014/
│   │       ├── train2014/
│   │       └── annotations/
│   │           └── instances_train2014.json
│   └── ScreenSpot/
│       ├── question_screenspot.json
│       └── image/

```

---

## 🔽 Dataset Download

### ROC + RTC  
📎 [Download ROC & RTC (Google Drive)](https://drive.google.com/drive/folders/1k45OVgWmt3Y04hPJe7rSWXGVZq98YLS4?usp=sharing)

Unzip the contents and place them in:
```

\$DATA/
├── ROC/
└── RTC/

```

### RefCOCOg

- 📄 Question file: [refcocog.json (Google Drive)]()
- 🖼 Image download: [COCO2014 Train Images (train2014.zip)](http://images.cocodataset.org/zips/train2014.zip)
- 📋 Annotations: [COCO2014 Annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)

Unpack files and organize as:
```

RD/RefCOCOg/
├── refcocog.json
└── COCO2014/
    ├── train2014/
    └── annotations/
        └── instances_train2014.json

```

### ScreenSpot

- 📄 Question file: [question_screenspot.json (Google Drive)](https://drive.google.com/file/d/11T_ONq05C77GNdFYo2XAt6gSwaam9ux3/view?usp=sharing)
- 🖼 Images: [Download Screenshots (NJU Box)](https://box.nju.edu.cn/d/5b8892c1901c4dbeb715/)

Organize as:
```

RD/ScreenSpot/
├── question_screenspot.json
└── image/

```

---

## 🗣 Prompt Format

### **ScreenSpot**
ScreenSpot is an evaluation benchmark for GUI grounding, comprising over 1,200 instructions from diverse environments including iOS, Android, macOS, Windows, and Web. Each data point is annotated with element type (`Text` or `Icon`).

- For `Icon` elements:  
  **"What is this icon used for?"**

- For `Text` elements:  
  **"What does this text say?"**

### **RefCOCOg**
The RefCOCOg dataset is a referring expression generation (REG) benchmark used to evaluate understanding of language that refers to specific objects in natural images.

- Generic prompt:  
  **"Can you provide a description of the region in a sentence?"**

### Prompt Differences by Model

- **LLaVA-based models (no localization pretraining)**:
  Use direct natural language queries as above.

- **Qwen2.5-VL (trained with grounding)**:
  Include box location to enhance region awareness:  
  **"Can you provide me with a detailed description of the region in the picture marked by box @ [x1, y1, x2, y2]."**

---

## 📝 Final Notes

- Make sure all `.json` files and images/masks follow the specified structure.
- Task scripts will expect the default root directory to be `data/` (relative to project root).
- You may modify `--data_path` arguments to specify custom locations during execution.

