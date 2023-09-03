import time

out_dir = "out-python-peps"
eval_interval = 5
eval_iters = 40
wandb_log = False  # feel free to turn on
wandb_project = "python-peps"
wandb_run_name = None

dataset = "python_peps"
init_from = "gpt2"  # use the smallest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# python-peps has 2,906,859 tokens, so 1 epoch ~= 88.7 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 20

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
