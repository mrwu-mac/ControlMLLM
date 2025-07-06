python qwen2_5_vl_7b_roc.py \
    --model_path='pretrained_models/Qwen2.5-VL-7B-Instruct' \
    --data_path='dataset/LVIS' \
    --question_file='dataset/LVIS/question_roc.json' \
    --answers_file='outputs/qwen2_5_vl_7b_roc.json' \
    --visual_prompt='Box' \
    --use_cd