CUDA_VISIBLE_DEVICES=0 python pretrain.py \
--data_dir ./data/sports/ \
--cuda \
--batch_size 64 \
--ratio  1:3:1:1 \
--checkpoint ./checkpoint/sports-ratio/ 

CUDA_VISIBLE_DEVICES=0 python seq.py \
--data_dir ./data/sports/ \
--cuda \
--batch_size 32 \
--checkpoint ./checkpoint/sports-ratio/

# CUDA_VISIBLE_DEVICES=0 python topn.py \
# --data_dir ./data/sports/ \
# --cuda \
# --batch_size 32 \
# --checkpoint ./checkpoint/sports/

# CUDA_VISIBLE_DEVICES=0 python exp.py \
# --data_dir ./data/sports/ \
# --cuda \
# --batch_size 32 \
# --checkpoint ./checkpoint/sports/