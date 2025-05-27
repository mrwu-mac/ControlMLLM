python llava_7b_rtc.py \
    --model_path='pretrained_models/llava-1.5-7b-hf' \
    --data_path='dataset/COCO-Text' \
    --question_file='dataset/COCO-Text/question_rtc.json' \
    --answers_file='outputs/llava_7b_rtc.json' \
    --visual_prompt='Box' \
    --use_cd