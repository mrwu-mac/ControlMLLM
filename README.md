# ControlMLLM
![teaser](assets/demo.png)

The repo is for the paper [ControlMLLM: Training-Free Visual Prompt Learning for Multimodal Large Language Models](https://arxiv.org/abs/2407.21534).

```
@misc{wu2024controlmllmtrainingfreevisualprompt,
      title={ControlMLLM: Training-Free Visual Prompt Learning for Multimodal Large Language Models}, 
      author={Mingrui Wu and Xinyue Cai and Jiayi Ji and Jiale Li and Oucheng Huang and Gen Luo and Hao Fei and Xiaoshuai Sun and Rongrong Ji},
      year={2024},
      eprint={2407.21534},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.21534}, 
}
```
## Setup
```
pip install -r requirements.txt
```

Install the Visulizer,
```
git clone https://github.com/luo3300612/Visualizer
cd Visualizer
pip install -e .
```
Install the Transformers we preprocessed,
```
git clone https://github.com/mrwu-mac/transformers
cd transformers
pip install -e .
```

## Demo
We will upload the code in a few days once it's ready.

## Acknowledge
[Layout-Guidance](https://github.com/silent-chen/layout-guidance), [Transformers](https://github.com/huggingface/transformers) and [Visualizer](https://github.com/luo3300612/Visualizer).
