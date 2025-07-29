import os
import torch
import random
import argparse
from transformers import T5Tokenizer
from utils.utils import ExpDataLoader, SeqDataLoader, AllBatchify, SeqBatchify, now_time, evaluate_ndcg, evaluate_hr
import copy
from utils.utils import compute_whole_word_id
import torch.nn as nn
import wandb
import math

# wandb.init(project="grpo-rec", name="grpo-500-test")
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = argparse.ArgumentParser(description='GRPO')
parser.add_argument('--data_dir', type=str, default=None,
                    help='directory for loading the data')
parser.add_argument('--model_version', type=int, default=0,
                    help='1: t5-base; 2: t5-large; 3: t5-3b; 4: t5-11b; otherwise: t5-small')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--checkpoint', type=str, default='./sft/',
                    help='directory to load the sft model')
parser.add_argument('--gcheckpoint', type=str, default='./grpo/',
                    help='directory to store the grpo model')
parser.add_argument('--num_grpo', type=int, default=4,
                    help='number of generations')
parser.add_argument('--num_beams', type=int, default=20,
                    help='number of beams')
parser.add_argument('--top_n', type=int, default=15,
                    help='number of items to predict')
args = parser.parse_args()

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

print(now_time() + 'Loading model')
tokenizer = T5Tokenizer.from_pretrained(model_version)

# Load the best saved model.
with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)


print(now_time() + 'Loading datasets')
seq_corpus = SeqDataLoader(args.data_dir)
nitem = len(seq_corpus.id2item)
seq_iterator = SeqBatchify(seq_corpus.user2items_positive, tokenizer, args.batch_size)

def eval_seq(model):
    def generate(model):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        idss_predict = []
        with torch.no_grad():
            while True:
                task, source, source_mask, whole_word, _ = seq_iterator.next_batch_test()
                task = task.to(device)  # (batch_size,)
                source = source.to(device)  # (batch_size, seq_len)
                source_mask = source_mask.to(device)
                whole_word = whole_word.to(device)

                beam_outputs = model.beam_search(task, source, whole_word, source_mask,
                                                num_beams=args.num_beams,
                                                num_return_sequences=args.top_n,
                                                )

                output_tensor = beam_outputs.view(task.size(0), args.top_n, -1)
                for i in range(task.size(0)):
                    results = tokenizer.batch_decode(output_tensor[i], skip_special_tokens=True)
                    idss_predict.append(results)

                if seq_iterator.step == seq_iterator.total_step:
                    break
        return idss_predict
    idss_predicted = generate(model)
    print(now_time() + 'Evaluation')
    user2item_test = {}
    interacted_items = {}
    for user, item_list in seq_corpus.user2items_positive.items():
        user2item_test[user] = [int(item_list[-1])]
        interacted_items[user] = [int(item_id) for item_id in item_list[:-1]]
    user2rank_list = {}
    for predictions, user in zip(idss_predicted, seq_iterator.user_list):
        prediction_list = []
        for p in predictions:
            # For exclude the unexpected results, including Non-ID, e.g., 'user_' and '_', and IDs of interacted items.
            try:
                predicted_item_id = int(p.split(' ')[0])  # use the id before white space
                if predicted_item_id not in interacted_items[user]:
                    prediction_list.append(predicted_item_id)
            except:
                prediction_list.append(random.randint(1, nitem))  # randomly generate a recommendation
        user2rank_list[user] = prediction_list

    for top_n in [1, 5, 10]:
        hr = evaluate_hr(user2item_test, user2rank_list, top_n)
        print(now_time() + 'HR@{} {:7.4f}'.format(top_n, hr))
        ndcg = evaluate_ndcg(user2item_test, user2rank_list, top_n)
        print(now_time() + 'NDCG@{} {:7.4f}'.format(top_n, ndcg))

print("\nSFT model evaluation before GRPO:")
# eval_seq(model)

# 先省略optimize_model_memory
# model = optimize_model_memory(model)

def combined_reward(prompts, completions, answer):
    rewards = [] 
    
    for i,v in enumerate(completions):
        rw = 1
        if not v.isdigit():
            rw -= 1
        elif v in prompts[i]:
            rw -= 0.7
        elif answer[i] == v:
            rw += 1
        else:
            rw -= 0.6

        rewards.append(round(rw, 1))
    return rewards

def selective_log_softmax(logits, input_ids):
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

def compute_log_probs(model, task_id, input_ids, whole_word_ids, attention_mask, decoder_input_ids, decoder_attention_mask):

    logits = model(
        task_id=task_id, 
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        whole_word_ids=whole_word_ids, 
        decoder_input_ids=decoder_input_ids, 
        decoder_attention_mask=decoder_attention_mask
        ).logits

    # print(decoder_input_ids.shape)
    # print(logits.shape)

    return selective_log_softmax(logits, decoder_input_ids)

def generate_rollout_data(model, ref_model, tokenizer, num_generations):

    item_list, answers, (task, source, source_mask, whole_word, _) = seq_iterator.gpo_next_batch()
    with torch.no_grad():
        task = task.to(device)  # (batch_size,)
        source = source.to(device)  # (batch_size, seq_len)
        source_mask = source_mask.to(device)
        whole_word = whole_word.to(device)

        beam_outputs = model.beam_search(task, source, whole_word, source_mask,
                                    num_beams=num_generations,
                                    num_beam_groups=num_generations, 
                                    num_return_sequences=num_generations,
                                    )  # (batch_size*num_generations, seq_len)

        print(beam_outputs.shape)
        # output mask
        merge = None
        formatted_completions = []
        output_tensor = beam_outputs.view(task.size(0), num_generations, -1)
        for i in range(task.size(0)):
            results = tokenizer.batch_decode(output_tensor[i], skip_special_tokens=True)
            for j in results:
                formatted_completions.append(j)
            
            mid = tokenizer(results, padding=True, return_tensors='pt')
            mid_mask = mid['attention_mask'].contiguous()

            if mid_mask.size(1) < beam_outputs.size(1):
                zero_matrix = torch.zeros(num_generations, 1, dtype=torch.int)
                new_mask = torch.cat([zero_matrix, mid_mask], dim=1)
            else:
                new_mask = mid_mask
            while new_mask.size(1) < beam_outputs.size(1):
                zero_matrix = torch.zeros(num_generations, 1, dtype=torch.int)
                new_mask = torch.cat([new_mask, zero_matrix], dim=1) 

            if merge is None:
                merge = new_mask
            else:
                merge = torch.vstack((merge, new_mask))

        task = task.repeat_interleave(num_generations, dim=0)
        source = source.repeat_interleave(num_generations, dim=0)
        source_mask = source_mask.repeat_interleave(num_generations, dim=0)
        whole_word = whole_word.repeat_interleave(num_generations, dim=0)

        target_mask = merge.to(device)

        # print("task shape:", task.shape)
        # print("source shape:",source.shape)
        # print("merge.shape:", merge.shape)
        # print("beam_outputs shape:", beam_outputs.shape)
        # print("input_ids shape:", input_ids.shape)
        # print("attention_mask shape:", attention_mask.shape)
        # print("whole_word shape:", whole_word.shape)

        old_log_probs = compute_log_probs(model, task, source, whole_word, source_mask, beam_outputs, target_mask)
        ref_log_probs = compute_log_probs(ref_model, task, source, whole_word, source_mask, beam_outputs, target_mask)

    repeated_prompts = [p for p in item_list for _ in range(num_generations)]
    repeated_answers = [a for a in answers for _ in range(num_generations)]        
        
    return {
        "source": source,
        "source_mask": source_mask,
        "beam_outputs": beam_outputs,
        "target_mask": target_mask,
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
        "formatted_completions": formatted_completions,
        "repeated_prompts": repeated_prompts,
        "repeated_answers": repeated_answers,
        "batch_size": len(item_list),
        "num_generations": num_generations,
        "task": task,
        "whole_word": whole_word
    }

def grpo_loss(model, ref_model, rollout_data, tokenizer, reward_function, beta=0.04, epsilon=0.1):

    source = rollout_data["source"]
    source_mask = rollout_data["source_mask"]
    beam_outputs = rollout_data["beam_outputs"]
    target_mask = rollout_data["target_mask"]
    task = rollout_data["task"]
    whole_word = rollout_data["whole_word"]
    old_log_probs = rollout_data["old_log_probs"]
    ref_log_probs = rollout_data["ref_log_probs"]
    token_log_probs = compute_log_probs(model, task, source, whole_word, source_mask, beam_outputs, target_mask)   
    
    mid = token_log_probs - old_log_probs
    ratio = torch.exp(mid / (old_log_probs + token_log_probs + 1e-8))
    rewards = torch.tensor(
        reward_function(prompts=rollout_data["repeated_prompts"], completions=rollout_data["formatted_completions"], answer=rollout_data["repeated_answers"]),
        dtype=torch.float32,
        device=device
    )
    #print(f"Rewards: {rewards}")  # Debug rewards
    batch_size = rollout_data["batch_size"]
    num_generations = rollout_data["num_generations"]
    rewards = rewards.view(batch_size, num_generations)
    avg_reward = rewards.mean().item()
    print("Average Reward:", avg_reward)
    mean_rewards = rewards.mean(dim=1).repeat_interleave(num_generations)
    std_rewards = rewards.std(dim=1).repeat_interleave(num_generations)
    advantages = ((rewards.view(-1) - mean_rewards) / (std_rewards + 1e-4)).unsqueeze(1)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    surrogate_loss = torch.min(surr1, surr2)
    kl = torch.exp(ref_log_probs - token_log_probs) - (ref_log_probs - token_log_probs) - 1
    per_token_loss = surrogate_loss - beta * kl

    loss = -((per_token_loss * target_mask).sum(dim=1) / target_mask.sum(dim=1)).mean()
    return loss, avg_reward

def train_with_grpo(model, tokenizer, num_iterations=1, 
                           steps_per_iteration=500, num_generations=8, 
                           beta=0.04, learning_rate=5e-6, 
                           mu=1, epsilon=0.1, reward_function=combined_reward):
    
    for iteration in range(num_iterations):
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        print("Reference model created.")    

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        model.train()

        for step in range(10):
            with torch.no_grad():
                rollout_data = generate_rollout_data(model, ref_model, tokenizer, num_generations)
            loss, avg_reward = grpo_loss(
                model,
                ref_model,
                rollout_data,
                tokenizer,
                reward_function,
                beta=beta,
                epsilon=epsilon
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            wandb.log({
                "loss": loss.item(),
                "average_reward": avg_reward,
                "iteration": iteration + 1,
                "step": step + 1,
            })
            print(f"Iteration {iteration+1}/{num_iterations}, Step {step+1}/{steps_per_iteration}, "
                    f"loss: {loss.item():.4f}")
    return model            

model = train_with_grpo(
    model=model,
    tokenizer=tokenizer,
)

print("\nFinal model evaluation after GRPO RL finetuning:")
eval_seq(model)
print("\nSaving GRPO finetuned model...")
if not os.path.exists(args.gcheckpoint):
    os.makedirs(args.gcheckpoint)
save_path = os.path.join(args.gcheckpoint, 'model.pt')
with open(save_path, 'wb') as f:
    torch.save(model, f)