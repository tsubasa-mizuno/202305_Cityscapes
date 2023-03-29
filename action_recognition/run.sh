CUDA_VISIBLE_DEVICES=0 python train.py -s /mnt/SSD4TB/detectron/panopticUCFframe -t /mnt/SSD4TB/detectron/UCFframe  --multiview -p 0.4 --paste
CUDA_VISIBLE_DEVICES=0 python train.py -s /mnt/SSD4TB/panopticUCFframe -t /mnt/SSD4TB/UCFframe  --multiview -p 0.2 --paste --category_sampling Random
CUDA_VISIBLE_DEVICES=0 python train.py -s /mnt/SSD4TB/panopticUCFframe -t /mnt/SSD4TB/UCFframe  --multiview -p 0.8
CUDA_VISIBLE_DEVICES=0 python train.py -s /mnt/SSD4TB/panopticUCFframe -t /mnt/SSD4TB/UCFframe  --multiview -p 0.8 --paste
CUDA_VISIBLE_DEVICES=0 python train.py -s /mnt/SSD4TB/panopticUCFframe -t /mnt/SSD4TB/UCFframe  --multiview -p 1
CUDA_VISIBLE_DEVICES=1 python train.py -s /mnt/SSD4TB/panopticUCFframe -t /mnt/SSD4TB/UCFframe  --multiview --paste
CUDA_VISIBLE_DEVICES=1 python train.py -s /mnt/SSD4TB/panopticUCFframe -t /mnt/SSD4TB/UCFframe  --multiview -p 0.8
CUDA_VISIBLE_DEVICES=1 python train.py -s /mnt/SSD4TB/panopticUCFframe -t /mnt/SSD4TB/UCFframe  --multiview -p 1
