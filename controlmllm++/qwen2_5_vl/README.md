# ControlMLLM++ for Qwen2.5-VL

This folder contains the setup and execution scripts for **ControlMLLM++** on the Qwen2.5-VL model, supporting **ROC**, **RTC**, and **Reference Description (RD)** tasks.


## 🔧 Environment Setup

We recommend creating the environment using the provided `.yml` file:

```bash
conda env create -f controlmllm_plus_qwen.yml
conda activate controlmllm_plus_qwen
````

### Install Visualizer

```bash
git clone https://github.com/mrwu-mac/Visualizer
cd Visualizer
pip install -e .
```

### Install Qwen2.5-VL-compatible Transformers

```bash
git clone https://github.com/C3236455482/transformers-qwen2_5_vl
cd transformers-qwen2_5_vl
pip install -e .
```

> 📌 Note: This customized `transformers-qwen2_5_vl` repo includes hooks for integrating gradient-based attention with ControlMLLM++.



## 🧩 Utility Script

Make sure `qwen_utils.py` is placed in this directory (`controlmllm++/qwen2_5_vl/`), as all task scripts rely on it for visualization and data loading.


## ▶️ Running Tasks

### 1. ROC Task

```bash
sh roc/qwen2_5_vl_7b_roc.sh
```

### 2. RTC Task

```bash
sh rtc/qwen2_5_vl_7b_rtc.sh
```

### 3. Reference Description (RD)

#### RefCOCOg

```bash
sh refcocog/qwen2_5_vl_7b_refcocog.sh
```

#### ScreenSpot

```bash
sh screenspot/qwen2_5_vl_7b_screenspot.sh
```


## 📁 Directory Structure

```
controlmllm++
└── qwen2_5_vl
    ├── roc/
    │   ├── qwen2_5_vl_7b_roc.py
    │   └── qwen2_5_vl_7b_roc.sh
    ├── rtc/
    │   ├── qwen2_5_vl_7b_rtc.py
    │   └── qwen2_5_vl_7b_rtc.sh
    ├── refcocog/
    │   ├── qwen2_5_vl_7b_refcocog.py
    │   └── qwen2_5_vl_7b_refcocog.sh
    ├── screenspot/
    │   ├── qwen2_5_vl_7b_screenspot.py
    │   └── qwen2_5_vl_7b_screenspot.sh
    ├── controlmllm_plus_qwen.yml
    └── qwen_utils.py
```

Each shell script includes customizable arguments such as:

* `--model_path`
* `--data_path`
* `--question_file`
* `--answers_file`

Modify as needed to match your file structure or prompt style.


## 📌 Notes

* ROC and RTC evaluate visual grounding precision.
* RD evaluates free-form language generation grounded to a region.
* All outputs will be saved to `outputs/` by default.

