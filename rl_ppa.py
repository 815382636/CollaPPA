import wandb
import argparse
import torch
import os
from transformers import T5Tokenizer
from utils.utils import now_time
from utils.utils import ExpDataLoader, SeqDataLoader, RLSeqBatchify, SeqBatchify, DPOSeqBatchify
import copy
import torch.nn as nn
import random

os.environ["WANDB_API_KEY"] = '' # 将引号内的+替换成自己在wandb上的一串值
os.environ["WANDB_MODE"] = "offline"   # 离线  （此行代码不用修改）
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser(description='PPA')
parser.add_argument('--data_dir', type=str, default=None,
                    help='directory for loading the data')
parser.add_argument('--model_version', type=int, default=0,
                    help='1: t5-base; 2: t5-large; 3: t5-3b; 4: t5-11b; otherwise: t5-small')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--checkpoint', type=str, default='./sft/',
                    help='directory to load the sft model')
parser.add_argument('--gcheckpoint', type=str, default='./dpo/',
                    help='directory to store the grpo model')
parser.add_argument('--epochs', type=int, default=1,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--log_interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--endure_times', type=int, default=2,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--num_beam', type=int, default=20,
                    help='number of beam')
parser.add_argument('--num_return_sequences', type=int, default=15,
                    help='number of return sequence')
parser.add_argument('--num_negative', type=int, default=4)
parser.add_argument('--reverse', type=int, default=0, help='beam_search reasult applied reverse or not')
parser.add_argument('--lr', type=float, default=5e-6,
                    help='learning rate')
parser.add_argument('--beta', type=float, default=0.1)
args = parser.parse_args()

seed = 42
torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
wandb.init(project="CollaPPA", config=vars(args))

if args.model_version == 1:
    model_version = 't5-base'
elif args.model_version == 2:
    model_version = 't5-large'
elif args.model_version == 3:
    model_version = 't5-3b'
elif args.model_version == 4:
    model_version = 't5-11b'
else:
    model_version = 't5-small'

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

torch.manual_seed(42)
if torch.cuda.is_available():
    if not args.cuda:
        print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
device = torch.device('cuda' if args.cuda else 'cpu')

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)
model_path = os.path.join(args.checkpoint, 'model.pt')

if not os.path.exists(args.gcheckpoint):
    os.makedirs(args.gcheckpoint)
new_model_path = os.path.join(args.gcheckpoint, 'model.pt')

tokenizer = T5Tokenizer.from_pretrained(model_version)

print(now_time() + 'Loading model')
with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

wandb.watch(model)

print(now_time() + 'Creat reference model')
ref_model = copy.deepcopy(model)
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False

print(now_time() + 'Loading datasets')
seq_corpus = SeqDataLoader(args.data_dir)
nitem = len(seq_corpus.id2item)
all_iterator = DPOSeqBatchify(seq_corpus.user2items_positive, tokenizer, args.batch_size)
seq_iterator = SeqBatchify(seq_corpus.user2items_positive, tokenizer, args.batch_size)

def selective_log_softmax(logits, input_ids):
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

def compute_log_probs(model=None, task=None, source=None, whole_word=None, source_mask=None, generate=None, encoder_tensor=None):

    # print(generate.shape)

    result  = model(task, source, whole_word, source_mask, labels=generate, encoder_outputs=encoder_tensor)
    # print(result.loss)
    logits = result.logits

    return selective_log_softmax(logits, generate)

def train():
    model.train()
    while True:
        remain_seg, seq_output, (task, source, source_mask, whole_word, target, target_mask) = all_iterator.next_batch()
        task = task.to(device)  # (batch_size,)
        source = source.to(device)  # (batch_size, seq_len)
        source_mask = source_mask.to(device)
        whole_word = whole_word.to(device)
        target = target.to(device)
        target_mask = target_mask.to(device)

        # ref_model = copy.deepcopy(model)
        # ref_model.eval()
        # for param in ref_model.parameters():
        #     param.requires_grad = False

        with torch.no_grad():    
            beam_outputs = model.beam_search(task, source, whole_word, source_mask,
                                        num_beams=args.num_beam,
                                        # num_beam_groups=2, 
                                        num_return_sequences=args.num_return_sequences,
                                        )  # (batch_size*num_generations, seq_len)

            # print(beam_outputs.shape)

            formatted_completions = []
            output_tensor = beam_outputs.view(task.size(0), args.num_return_sequences, -1)
            for i in range(task.size(0)):
                results = tokenizer.batch_decode(output_tensor[i], skip_special_tokens=True)
                if args.reverse == 0:
                    # 逆序
                    for j in reversed(results):
                        if len(formatted_completions) >= (i + 1) * args.num_negative:
                            break 
                        if j != seq_output[i]  and j not in remain_seg[i]:
                            formatted_completions.append(j)
                    while len(formatted_completions) < (i + 1) * args.num_negative:
                            formatted_completions.append("-1")                            
                elif args.reverse == 1:  
                    # 顺序         
                    for index,j in enumerate(results):
                        if len(formatted_completions) >= (i + 1) * args.num_negative:
                            break 

                        if j != seq_output[i]  and j not in remain_seg[i]:
                            formatted_completions.append(j)

                        while index == len(results) - 1 and len(formatted_completions) < (i + 1) * args.num_negative:
                            formatted_completions.append("-1")

                else:
                    # random
                    random.shuffle(results)
                    for index,j in enumerate(results):
                        if len(formatted_completions) >= (i + 1) * args.num_negative:
                            break 

                        if j != seq_output[i]  and j not in remain_seg[i]:
                            formatted_completions.append(j)

                        while index == len(results) - 1 and len(formatted_completions) < (i + 1) * args.num_negative:
                            formatted_completions.append("-1")                    


        encoded_target = tokenizer(formatted_completions, padding=True, return_tensors='pt')
        target_seq = encoded_target['input_ids'][:, :target.size(1)]
        target_seq = target_seq.to(device)

        target_reshape = target_seq.reshape(args.batch_size, args.num_negative, -1)

        target_sub_tensors = torch.unbind(target_reshape, dim=1)

        # task = task.repeat_interleave(args.num_negative, dim=0)
        # source = source.repeat_interleave(args.num_negative, dim=0)
        # source_mask = source_mask.repeat_interleave(args.num_negative, dim=0)
        # whole_word = whole_word.repeat_interleave(args.num_negative, dim=0)  
        # target = target.repeat_interleave(args.num_negative, dim=0)
        # target_mask = target_mask.repeat_interleave(args.num_negative, dim=0)

        encoder_tensor = model.fixed_encoder(task, source, whole_word, source_mask)
        with torch.no_grad():  
            ref_log_probs_lose_list = []
            for i in target_sub_tensors:
                sub_tensors = i.contiguous()
                # print(i.shape, target.shape)
                ref_log_probs_lose = compute_log_probs(model=ref_model, generate=sub_tensors, encoder_tensor=encoder_tensor)
                ref_log_probs_lose_list.append(ref_log_probs_lose)
            ref_log_probs_win = compute_log_probs(model=ref_model, generate=target, encoder_tensor=encoder_tensor)      

        log_probs_lose_list = []
        for i in target_sub_tensors:
            sub_tensors = i.contiguous()
            log_probs_lose = compute_log_probs(model=model, generate=sub_tensors, encoder_tensor=encoder_tensor)
            log_probs_lose_list.append(log_probs_lose)
        log_probs_win = compute_log_probs(model=model, generate=target, encoder_tensor=encoder_tensor)

        chosen_logratios = log_probs_win - ref_log_probs_win
        rejected_logratios = []
        for i in range(len(log_probs_lose_list)):
            rejected_logratios.append(log_probs_lose_list[i] - ref_log_probs_lose_list[i])

        temp = sum(torch.exp(args.beta * (key - chosen_logratios)) for key in rejected_logratios)
        temp1 = -torch.log(temp)

        loss = - torch.log(torch.sigmoid(temp1)).mean()
        
        # constant = 1.0 / (args.beta * 2.0)
        # loss = torch.square( logit_diff - constant ).mean()

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # Log to wandb
        wandb.log({
            "loss": loss.item(),
            # "average_reward": avg_reward,
            # "iteration": iteration + 1,
            # "step": step + 1,
            # "grpo_iter": grpo_iter + 1
        })
        # if all_iterator.batch_index % args.log_interval == 0 or all_iterator.batch_index % all_iterator.batch_num == 0:
        print(now_time() + 'loss {:4.4f}  | {:5d}/{:5d} batches'.format(loss, all_iterator.batch_index, all_iterator.batch_num))

        if all_iterator.batch_index % all_iterator.batch_num == 0:
            break

def evaluate(iterator):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    text_loss = 0.
    total_sample = 0
    with torch.no_grad():
        while True:
            task, source, source_mask, whole_word, target = iterator.next_batch_valid()
            task = task.to(device)  # (batch_size,)
            source = source.to(device)  # (batch_size, seq_len)
            source_mask = source_mask.to(device)
            whole_word = whole_word.to(device)
            target = target.to(device)
            outputs = model(task, source, whole_word, source_mask, labels=target)
            loss = outputs.loss

            batch_size = task.size(0)
            text_loss += batch_size * loss.item()
            total_sample += batch_size

            if iterator.step == iterator.total_step:
                break
    return text_loss / total_sample

print(now_time() + 'Start training')
# Loop over epochs.
best_val_loss = float('inf')
endure_count = 0

# seq_loss = evaluate(seq_iterator)
# print(now_time() + 'sequential loss {:4.4f}'.format(seq_loss))

for epoch in range(1, args.epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    train()
    print(now_time() + 'validation')
    seq_loss = evaluate(seq_iterator)
    print(now_time() + 'sequential loss {:4.4f}'.format(seq_loss))

    if seq_loss < best_val_loss:
        best_val_loss = seq_loss
        with open(new_model_path, 'wb') as f:
            torch.save(model, f)
            print("save model success")
    else:
        endure_count += 1
        print(now_time() + 'Endured {} time(s)'.format(endure_count))
        if endure_count == args.endure_times:
            print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
            break