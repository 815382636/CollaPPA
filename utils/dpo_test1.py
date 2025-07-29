import wandb
import argparse
import torch
import os
from transformers import T5Tokenizer
from utils.utils import now_time
from utils.utils import ExpDataLoader, SeqDataLoader, AllBatchify, SeqBatchify
import copy
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser(description='DPO4Rec')
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
parser.add_argument('--ratio', type=str, default='0:1:0:0',  # ratio: explanation:sequential:rationale:topn
                    help='ratio of various tasks')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--log_interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--endure_times', type=int, default=5,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--exp_len', type=int, default=20,
                    help='the maximum length of an explanation')
parser.add_argument('--negative_num', type=int, default=99,
                    help='number of negative items for top-n recommendation')
parser.add_argument('--num_beam', type=int, default=20,
                    help='number of beam')
parser.add_argument('--num_return_sequences', type=int, default=4,
                    help='number of return sequence')
parser.add_argument('--lr', type=float, default=5e-6,
                    help='learning rate')
parser.add_argument('--beta', type=float, default=0.1)
args = parser.parse_args()

# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# wandb.init(project="GRPO4rec", name="grpo-500-test")

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

print(now_time() + 'Creat reference model')
ref_model = copy.deepcopy(model)
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False

print(now_time() + 'Loading datasets')
exp_corpus = ExpDataLoader(args.data_dir)
seq_corpus = SeqDataLoader(args.data_dir)
nitem = len(seq_corpus.id2item)
all_iterator = AllBatchify(exp_corpus.train, seq_corpus.user2items_positive, args.negative_num, nitem, tokenizer, args.exp_len, args.batch_size, args.ratio)
seq_iterator = SeqBatchify(seq_corpus.user2items_positive, tokenizer, args.batch_size)

def selective_log_softmax(logits, input_ids):
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

def compute_log_probs(model, task, source, whole_word, source_mask, generate, encoder_tensor=None):

    # print(generate.shape)

    logits = model(task, source, whole_word, source_mask, labels=generate, encoder_outputs=encoder_tensor).logits

    return selective_log_softmax(logits, generate)

def train():
    model.train()
    while True:
        seq_input, seq_output, (task, source, source_mask, whole_word, target, target_mask) = all_iterator.rl_next_batch()
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
                for index,j in enumerate(results):
                    if j != seq_output[i]:
                        formatted_completions.append(j)

                    if index == len(results) - 1:
                        mid_len = len(formatted_completions)
                        for k in range((i + 1) * args.num_return_sequences - mid_len):
                            formatted_completions.append("-1")


            encoded_target = tokenizer(formatted_completions, padding=True, return_tensors='pt')
            target_seq = encoded_target['input_ids'][:, :target.size(1)]
            target_seq = target_seq.to(device)

            task = task.repeat_interleave(args.num_return_sequences, dim=0)
            source = source.repeat_interleave(args.num_return_sequences, dim=0)
            source_mask = source_mask.repeat_interleave(args.num_return_sequences, dim=0)
            whole_word = whole_word.repeat_interleave(args.num_return_sequences, dim=0)  
            target = target.repeat_interleave(args.num_return_sequences, dim=0)
            target_mask = target_mask.repeat_interleave(args.num_return_sequences, dim=0)

            # encoder_tensor_lose = model.fixed_encoder(task, source, whole_word, source_mask, labels=target_seq)
            # encoder_tensor_win = model.fixed_encoder(task, source, whole_word, source_mask, labels=target)


            ref_log_probs_lose = compute_log_probs(ref_model, task, source, whole_word, source_mask, target_seq)
            ref_log_probs_win = compute_log_probs(ref_model, task, source, whole_word, source_mask, target)      

        log_probs_lose = compute_log_probs(model, task, source, whole_word, source_mask, target_seq)
        log_probs_win = compute_log_probs(model, task, source, whole_word, source_mask, target)

        logit_diff = log_probs_win - log_probs_lose - (ref_log_probs_win - ref_log_probs_lose)

        loss = - torch.log(torch.sigmoid(args.beta * logit_diff)).mean()
        

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # Log to wandb
        # wandb.log({
        #     "loss": loss.item(),
        #     "average_reward": avg_reward,
        #     # "iteration": iteration + 1,
        #     # "step": step + 1,
        #     # "grpo_iter": grpo_iter + 1
        # })
        if all_iterator.batch_index % args.log_interval == 0 or all_iterator.batch_index % all_iterator.batch_num == 0:
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