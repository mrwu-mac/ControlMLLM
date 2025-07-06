python llava_7b_screenspot.py \
    --model_path='pretrained_models/llava-1.5-7b-hf' \
    --data_path='dataset/ScreenSpot' \
    --question_file='dataset/ScreenSpot/question_screenspot.json' \
    --answers_file='outputs/llava_7b_screenspot.json' \
    --visual_prompt='Box' \
    --use_cd