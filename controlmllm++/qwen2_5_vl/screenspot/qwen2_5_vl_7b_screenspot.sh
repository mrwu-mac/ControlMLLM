python qwen2_5_vl_7b_screenspot.py \
    --model_path='pretrained_models/Qwen2.5-VL-7B-Instruct' \
    --data_path='dataset/ScreenSpot' \
    --question_file='dataset/ScreenSpot/question_screenspot.json' \
    --answers_file='outputs/qwen2_5_vl_7b_screenspot.json' \
    --visual_prompt='Box' \
    --use_cd