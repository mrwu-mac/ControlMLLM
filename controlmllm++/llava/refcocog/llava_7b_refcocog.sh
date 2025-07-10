python llava_7b_refcocog.py \
    --model_path='pretrained_models/llava-1.5-7b-hf' \
    --data_path='dataset/COCO2014/train2014' \
    --coco_annotation_path='dataset/COCO2014/annotations/instances_train2014.json' \
    --question_file='dataset/COCO2014/refcocog.json' \
    --answers_file='outputs/llava_7b_refcocog.json' \
    --use_cd