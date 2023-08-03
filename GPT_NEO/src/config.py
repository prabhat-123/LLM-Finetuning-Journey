DATASET_NAME = "tatsu-lab/alpaca"
MODEL_NAME="EleutherAI/gpt-neo-125M"
OUTPUT_DIR = "./results"
LOGGING_DIR = "./logs"
LEARNING_RATE = 1e-4
NUM_TRAIN_EPOCHS = 5
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4
EVAL_STEPS = 12000
SAVE_STEPS = 25000
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 100
MAX_LENGTH=1024
TRAIN_RATIO=0.8
CHECKPOINT_PATH="./results/checkpoint-42000"
