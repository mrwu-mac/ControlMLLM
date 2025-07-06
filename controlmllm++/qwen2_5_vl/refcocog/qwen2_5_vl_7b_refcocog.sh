python qwen2_5_vl_7b_refcocog.py \
    --model_path='pretrained_models/Qwen2.5-VL-7B-Instruct' \
    --data_path='dataset/COCO2014/train2014' \
    --coco_annotation_path='dataset/COCO2014/annotations/instances_train2014.json' \
    --question_file='dataset/COCO2014/refcocog.json' \
    --answers_file='outputs/qwen2_5_vl_7b_refcocog.json' \
    --use_cd