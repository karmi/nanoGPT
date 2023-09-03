# train a miniature character-level model based on Python PEPs
# https://peps.python.org/pep-0001/

out_dir = "out-python-peps-char"
eval_interval = 100
eval_iters = 20
log_interval = 1

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False  # override via command line if you like
wandb_project = "python-peps-char"
wandb_run_name = None

dataset = "python_peps_char"
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = max_iters  # make equal to max_iters usually
min_lr = 1e-4  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 0  # not super necessary potentially

# on macbook also add
# device = 'mps'  # run on MPS (https://github.com/karpathy/nanoGPT/issues/28)
# compile = False # do not torch compile the model
