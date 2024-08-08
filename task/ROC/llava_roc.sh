# llava with box
CUDA_VISIBLE_DEVICES=6,7 python task/ROC/llava_roc.py --visual_prompt='box' --answers_file='outputs/llava_7b_roc_box.json' --early_stop

# llava with mask
# CUDA_VISIBLE_DEVICES=6,7 python task/ROC/llava_roc.py --visual_prompt='mask' --answers_file='outputs/llava_7b_roc_mask.json' --early_stop

# llava with scribble
# CUDA_VISIBLE_DEVICES=6,7 python task/ROC/llava_roc.py --visual_prompt='scribble' --answers_file='outputs/llava_7b_roc_scribble.json' --early_stop

# llava with point
# CUDA_VISIBLE_DEVICES=6,7 python task/ROC/llava_roc.py --visual_prompt='point' --answers_file='outputs/llava_7b_roc_point.json' --early_stop