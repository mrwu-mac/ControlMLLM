python qwen2_5_vl_7b_rtc.py \
    --model_path='pretrained_models/Qwen2.5-VL-7B-Instruct' \
    --data_path='dataset/COCO-Text' \
    --question_file='dataset/COCO-Text/question_rtc.json' \
    --answers_file='outputs/qwen2_5_vl_7b_rtc.json' \
    --visual_prompt='Box' \
    --use_cd