# ControlMLLM

<div align="center">
    <img src="assets/method.png" alt="method" width="550"/>
</div>

The repo is for the paper [ControlMLLM: Training-Free Visual Prompt Learning for Multimodal Large Language Models (NeurIPS2024)](https://arxiv.org/abs/2407.21534).

```
@inproceedings{NEURIPS2024_4fd96b99,
 author = {Wu, Mingrui and Cai, Xinyue and Ji, Jiayi and Li, Jiale and Huang, Oucheng and Gen Luo and Fei, Hao and Jiang, Guannan and Sun, Xiaoshuai and Ji, Rongrong},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {45206--45234},
 publisher = {Curran Associates, Inc.},
 title = {ControlMLLM: Training-Free Visual Prompt Learning for Multimodal Large Language Models},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/4fd96b997454b5b02698595df70fccaf-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}
```
## Features
 - Training-free method, supports running on a single RTX 3090 24GB GPU.
 - Provides visualization tools in ```utils.py``` for interpretability.

## News
 - ```2025/5/26:``` We release the code of ControlMLLM++, an extension of ControlMLLM, which introduces a new optimization strategy for better test-time stability and convergence. The technical report is coming soon.
 - ```2024/9/26:``` ControlMLLM is accepted by NeurIPS 2024.
 - ```2024/8/21:``` We release eval pipeline on ROC and RTC task. 
 - ```2024/8/8:``` We release demo on InstructBLIP.
 - ```2024/8/2:``` We release demo on LLaVA v1.5.

## Project Structure

| Folder / File         | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `controlmllm/`         | Original ControlMLLM implementation. Includes demo scripts, ROC & RTC tasks. |
| `controlmllm++/`       | Enhanced ControlMLLM++. Supports multi-model pipelines & RD task.            |
| `datasets.md`          | Unified dataset preparation guide (ROC, RTC, RefCOCOg, ScreenSpot).         |

### Setup and Usage Instructions

- For **ControlMLLM** (ROC, RTC, demo), see [`controlmllm/RUN.md`](controlmllm/RUN.md)
- For **ControlMLLM++**, each model has its own setup and run instructions:
  - [`controlmllm++/llava/RUN.md`](controlmllm++/llava/RUN.md)
  - [`controlmllm++/qwen2_5_vl/RUN.md`](controlmllm++/qwen2_5_vl/RUN.md)

## Data preparation
Please follow the instructions at [DATASETS.md](DATASETS.md) to prepare all datasets.


## Support Models

 - [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
 - [LLaVA v1.5](https://huggingface.co/llava-hf/llava-1.5-7b-hf)(version<='05ae243')
 - [InstructBLIP](https://huggingface.co/Salesforce/instructblip-vicuna-7b)
 - [LLaVA-HR](https://github.com/luogen1996/LLaVA-HR)
 - More coming soon
   
## Demo
```
python controlmllm/llava/llava_demo.py
```
![demo](assets/demo.png)

Tips: Due to the image cropping during preprocessing in LLaVA1.5, referring to region at the edges of the image may become unreliable. If your referring does not work, you can also try slightly adjusting the visual prompt or text prompt, which might produce surprising results.

## Results
The results of combining with different MLLMs on ROC and RTC tasks.

| MODELS                     | ROC    | RTC    |
|----------------------------|--------|--------|
| LLAVA-1.5                  | 54.72  | 57.42  |
| LLAVA-1.5 + CONTROLMLLM    | 60.59  | 63.06  |
| LLAVA-1.5 + CONTROLMLLM++  | 71.19  | 74.66  |
| LLAVA-HR                   | 53.81  | 57.00  |
| LLAVA-HR + CONTROLMLLM     | 58.92  | 66.89  |
| LLAVA-HR + CONTROLMLLM++   | 69.06  | 82.68  |
| QWEN2.5-VL                 | 78.81  | 81.91  |
| QWEN2.5-VL + CONTROLMLLM   | 79.20  | 86.43  |
| QWEN2.5-VL + CONTROLMLLM++ | 79.20  | 88.23  |

## Acknowledgement

[Layout-Guidance](https://github.com/silent-chen/layout-guidance), [ml-ferret](https://github.com/apple/ml-ferret), [Transformers](https://github.com/huggingface/transformers), [SeeClick](https://github.com/njucckevin/SeeClick) and [Visualizer](https://github.com/luo3300612/Visualizer).
