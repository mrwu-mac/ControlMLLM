# llava with box
CUDA_VISIBLE_DEVICES=6,7 python task/RTC/llava_rtc.py --visual_prompt='box' --answers_file='outputs/llava_7b_rtc_box.json' --early_stop

# llava with mask
# CUDA_VISIBLE_DEVICES=6,7 python task/RTC/llava_rtc.py --visual_prompt='mask' --answers_file='outputs/llava_7b_rtc_mask.json' --early_stop
