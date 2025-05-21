# llava with box
CUDA_VISIBLE_DEVICES=6,7 python task/ROC/llava_roc.py --visual_prompt='Box' --answers_file='outputs/llava_7b_roc_box.json' --early_stop

# llava with mask
# CUDA_VISIBLE_DEVICES=6,7 python task/ROC/llava_roc.py --visual_prompt='Mask' --answers_file='outputs/llava_7b_roc_mask.json' --early_stop

# llava with scribble
# CUDA_VISIBLE_DEVICES=6,7 python task/ROC/llava_roc.py --visual_prompt='Scribble' --answers_file='outputs/llava_7b_roc_scribble.json' --early_stop

# llava with point
# CUDA_VISIBLE_DEVICES=6,7 python task/ROC/llava_roc.py --visual_prompt='Point' --answers_file='outputs/llava_7b_roc_point.json' --early_stop
