# ControlMLLM

This folder contains code and instructions for the original **ControlMLLM**.


## 🔧 Environment Setup

We recommend using the following environment setup to run ControlMLLM:

```bash
conda create -n controlmllm python=3.9
conda activate controlmllm
pip install -r requirements.txt
````

### Install Visualizer

```bash
git clone https://github.com/mrwu-mac/Visualizer
cd Visualizer
pip install -e .
```

### Install Transformers (with attention hook support)

```bash
git clone https://github.com/mrwu-mac/transformers
cd transformers
pip install -e .
```

> 💡 **Note**: The customized `transformers` and `visualizer` packages are used to capture attention maps via gradients.

Tips: Our key modification in the Transformers involves using the Visualizer toolkit to obtain gradient-based attention maps. If you need to modify your own model, you can easily do so by using the Visualizer to decorate the attention function in your LLM decoder,
```python
from visualizer import get_local
class LLMAttention():
    @get_local('attn_weights')
    def forward():
         ...
        attn_weights, value = scaled_dot_product_attention(
            query,
            key,
            value
        )
        attn_output = attn_weights @ value
         ...
        return attn_output
```
And obtain gradients by removing the @torch.no_grad() decorator during generation,
```python
class LLMGeneration():
    # @torch.no_grad()
    def generate():
```

## 🚀 Running ROC and RTC Tasks

Please make sure datasets are prepared as described in [datasets.md](../datasets.md).

### Step 1. Run Inference

For ROC task:

```bash
sh llava/roc/llava_roc.sh
```

For RTC task:

```bash
sh llava/rtc/llava_rtc.sh
```

Each script supports optional arguments:

* `--model_path`: Path to model (default: `pretrained_models/llava-1.5-7b-hf`)
* `--data_path`: Path to dataset
* `--question_file`: Input prompt file
* `--answers_file`: Output result file
* `--visual_prompt`: Type of prompt (e.g., `box`, `scribble`, etc.)

### Step 2. Evaluate

ROC:

```bash
python llava/roc/eval.py --pred_file=outputs/llava_7b_roc_box.json --set=test
```

RTC:

```bash
python llava/rtc/eval.py --pred_file=outputs/llava_7b_rtc_box.json
```


For enhanced visual grounding and generation, refer to the upgraded version in [`controlmllm++`](../controlmllm++/llava/RUN.md).
