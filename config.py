import argparse

import torch

RAW_DATASET_ROOT_FOLDER = 'data'
EXPERIMENT_ROOT = 'experiments'
STATE_DICT_KEY = 'model_state_dict'
OPTIMIZER_STATE_DICT_KEY = 'optimizer_state_dict'
PROJECT_NAME = 'llmrec'


def set_template(args):
    if args.dataset_code is None:
        print('******************** Dataset Selection ********************')
        dataset_code = {'1': 'ml-100k', 'b': 'beauty', 'g': 'games', 't': 'toys'}
        args.dataset_code = dataset_code[input('Input 1 for ml-100k, b for beauty, g for games and t for toys: ')]

    if args.dataset_code == 'ml-100k':
        args.bert_max_len = 200
    else:
        args.bert_max_len = 50

    if 'llm' in args.model_code:
        batch = 2 if args.dataset_code in ['ml-100k'] else 3
        if args.dataset_code in ["toys"]:
            batch = 2
        elif args.dataset_code in ["games"]:
            batch = 2
        elif args.dataset_code in ["beauty"]:
            batch = 6
        args.lora_micro_batch_size = batch
    else:
        batch = 16 if args.dataset_code == 'ml-100k' else 64

    args.train_batch_size = batch
    args.val_batch_size = batch
    args.test_batch_size = batch

    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'
    args.optimizer = 'AdamW'
    args.lr = 0.001
    args.weight_decay = 0.01
    args.enable_lr_schedule = False
    args.decay_step = 10000
    args.gamma = 1.
    args.enable_lr_warmup = False
    args.warmup_steps = 100

    args.metric_ks = [1, 5, 10, 20, 50]
    args.rerank_metric_ks = [1, 5, 10]
    args.best_metric = 'Recall@10'
    args.rerank_best_metric = 'NDCG@10'

    args.bert_num_blocks = 2
    args.bert_num_heads = 2
    args.bert_head_size = None

    if args.model_code == 'llm':
        if args.llm_base_tokenizer is None:
            args.llm_base_tokenizer = args.llm_base_model

        if args.llm_retrieved_path is None:
            args.llm_retrieved_path = f"./experiments/lru/{args.dataset_code}/"

        if args.llm_system_template is None:
            if args.dataset_code == 'ml-100k':
                args.llm_system_template = "You are going to play a movie enthusiast who is selecting movies they are interested in. I will provide you with the movies you have previously browsed in chronological order,candidates pool, and vision descriptions for each item in thepreviously browsed item and candidates pool. The provided data is in the format {item : vision descriptions}. please correlate the item with the corresponding vision descriptions. Recommend an item from the candidate pool  with its index letter."
            elif args.dataset_code == 'games':
                args.llm_system_template = "You will play as a gamer, selecting the games you love. I will provide you with the games you have previously browsed in chronological order,candidates pool, and vision descriptions for each item in thepreviously browsed item and candidates pool. The provided data is in the format {item : vision descriptions}. please correlate the item with the corresponding vision descriptions. Recommend an item from the candidate pool  with its index letter."
            elif args.dataset_code == 'beauty':
                args.llm_system_template = "You will play the role of a female, purchasing female items for yourself. I will provide you with the items you have previously browsed in chronological order,candidates pool, and vision descriptions for each item in thepreviously browsed item and candidates pool. The provided data is in the format {item : vision descriptions}. please correlate the item with the corresponding vision descriptions. Recommend an item from the candidate pool  with its index letter."
            elif args.dataset_code == 'toys':
                args.llm_system_template = "You will play the role of a guardian, purchasing children's toys for your own child. I will provide you with the items you have previously browsed in chronological order,candidates pool, and vision descriptions for each item in thepreviously browsed item and candidates pool. The provided data is in the format {item : vision descriptions}. please correlate the item with the corresponding vision descriptions. Recommend an item from the candidate pool  with its index letter."


parser = argparse.ArgumentParser()

################
# Dataset
################
parser.add_argument('--dataset_code', type=str, default=None)
parser.add_argument('--min_rating', type=int, default=0)
parser.add_argument('--min_uc', type=int, default=5)
parser.add_argument('--min_sc', type=int, default=5)
parser.add_argument('--seed', type=int, default=42)

################
# Dataloader
################
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--sliding_window_size', type=float, default=1.0)
parser.add_argument('--negative_sample_size', type=int, default=10)

################
# Trainer
################
# optimization #
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'Adam'])
parser.add_argument('--weight_decay', type=float, default=None)
parser.add_argument('--adam_epsilon', type=float, default=1e-9)
parser.add_argument('--momentum', type=float, default=None)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--max_grad_norm', type=float, default=5.0)
parser.add_argument('--enable_lr_schedule', type=bool, default=True)
parser.add_argument('--decay_step', type=int, default=10000)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--enable_lr_warmup', type=bool, default=True)
parser.add_argument('--warmup_steps', type=int, default=100)

# evaluation #
parser.add_argument('--val_strategy', type=str, default='iteration', choices=['epoch', 'iteration'])
parser.add_argument('--val_iterations', type=int, default=500)  # only for iteration val_strategy
parser.add_argument('--early_stopping', type=bool, default=True)
parser.add_argument('--early_stopping_patience', type=int, default=20)
parser.add_argument('--metric_ks', nargs='+', type=int, default=[1, 5, 10, 20, 50])
parser.add_argument('--rerank_metric_ks', nargs='+', type=int, default=[1, 5, 10])
parser.add_argument('--best_metric', type=str, default='Recall@10')
parser.add_argument('--rerank_best_metric', type=str, default='NDCG@10')
parser.add_argument('--use_wandb', type=bool, default=False)

################
# Retriever Model
################
parser.add_argument('--model_code', type=str, default=None)
parser.add_argument('--bert_max_len', type=int, default=50)
parser.add_argument('--bert_hidden_units', type=int, default=64)
parser.add_argument('--bert_num_blocks', type=int, default=2)
parser.add_argument('--bert_num_heads', type=int, default=2)
parser.add_argument('--bert_head_size', type=int, default=32)
parser.add_argument('--bert_dropout', type=float, default=0.2)
parser.add_argument('--bert_attn_dropout', type=float, default=0.2)
parser.add_argument('--bert_mask_prob', type=float, default=0.25)

################
# LLM Model
################
parser.add_argument('--llm_base_model', type=str, default=None)
parser.add_argument('--llm_base_tokenizer', type=str, default=None)
parser.add_argument('--llm_max_title_len', type=int, default=32)
parser.add_argument('--llm_max_text_len', type=int, default=1536)
parser.add_argument('--llm_max_history', type=int, default=25)
parser.add_argument('--llm_train_on_inputs', type=bool, default=False)
parser.add_argument('--llm_negative_sample_size', type=int, default=19)  # 19 negative & 1 positive

# instruction
parser.add_argument('--llm_system_template', type=str, default=None)
parser.add_argument('--llm_input_template', type=str, default='Previously browsed: {}; \n Candidate pool: {}')
parser.add_argument('--llm_load_in_4bit', type=bool, default=True)
parser.add_argument('--llm_retrieved_path', type=str, default=None)
parser.add_argument('--llm_cache_dir', type=str, default=None)

################
# Lora
################
parser.add_argument('--lora_r', type=int, default=8)
parser.add_argument('--lora_alpha', type=int, default=32)
parser.add_argument('--lora_dropout', type=float, default=0.05)
parser.add_argument('--lora_target_modules', type=list, default=['q_proj', 'v_proj'])
parser.add_argument('--lora_num_epochs', type=int, default=100)
parser.add_argument('--lora_val_iterations', type=int, default=1)
parser.add_argument('--lora_early_stopping_patience', type=int, default=10)
parser.add_argument('--lora_lr', type=float, default=1e-4)
parser.add_argument('--lora_micro_batch_size', type=int, default=16)

################
# blip2
parser.add_argument('--model_path', type=str, default=None)
################

args = parser.parse_args()
