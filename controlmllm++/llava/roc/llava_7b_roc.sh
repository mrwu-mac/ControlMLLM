python llava_7b_roc.py \
    --model_path='pretrained_models/llava-1.5-7b-hf' \
    --data_path='dataset/LVIS' \
    --question_file='dataset/LVIS/question_roc.json' \
    --answers_file='outputs/llava_7b_roc.json' \
    --visual_prompt='Box' \
    --use_cd