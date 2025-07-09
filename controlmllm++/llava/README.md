# ControlMLLM++ for LLaVA

This folder contains the setup and execution scripts for **ControlMLLM++** on the LLaVA model, supporting **ROC**, **RTC**, and **Reference Description (RD)** tasks.

## 🔧 Environment Setup

We recommend creating the environment using the provided `.yml` file:

```bash
conda env create -f controlmllm_plus_llava.yml
conda activate controlmllm_plus_llava
````

### Install Visualizer

```bash
git clone https://github.com/mrwu-mac/Visualizer
cd Visualizer
pip install -e .
```

### Install LLaVA-compatible Transformers

```bash
git clone https://github.com/C3236455482/transformers-llava
cd transformers-llava
pip install -e .
```

> 📌 Note: This customized `transformers-llava` repo includes necessary hooks to integrate with gradient-based attention visualization using ControlMLLM++.


## 🧩 Utility Script
Ensure the shared utility file utils.py is placed in this directory (controlmllm++/llava/), so that all task scripts can access the necessary visualization and IO functions.

## ▶️ Tasks

### 1. ROC Task

```bash
sh roc/run_llava_roc.sh
```

### 2. RTC Task

```bash
sh rtc/run_llava_rtc.sh
```

### 3. Reference Description (RD)

#### RefCOCOg

```bash
sh refcocog/run_llava_refcocog.sh
```

#### ScreenSpot

```bash
sh screenspot/run_llava_screenspot.sh
```


## 📁 Directory Structure

```
controlmllm++
└── llava
    ├── roc/
    │   └── run_llava_roc.sh
    ├── rtc/
    │   └── run_llava_rtc.sh
    ├── refcocog/
    │   └── run_llava_refcocog.sh
    ├── screenspot/
    │   └── run_llava_screenspot.sh
    └── controlmllm_plus_llava.yml
```

Each shell script includes customizable arguments such as:

* `--model_path`
* `--data_path`
* `--question_file`
* `--answers_file`

You may modify them to suit your file paths and desired prompt types.


## 📌 Notes

* The ROC and RTC tasks evaluate visual grounding precision.
* The RD task evaluates generation ability using natural language prompts.
* All results are saved to the `outputs/` directory by default.

