CUDA_VISIBLE_DEVICES=0 python rl_ppa.py \
--data_dir ./data/sports/ \
--cuda \
--batch_size 64 \
--lr  5e-6 \
--beta 0.1 \
--num_negative 8 \
--reverse 0 \
--checkpoint ./checkpoint/sports-ratio/ \
--gcheckpoint ./ppa/sports-ratio/

CUDA_VISIBLE_DEVICES=0 python seq.py \
--data_dir ./data/sports/ \
--cuda \
--batch_size 32 \
--checkpoint ./ppa/sports-ratio/