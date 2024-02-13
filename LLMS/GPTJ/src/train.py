import math
import torch
import random
import pandas as pd
import config as cfg
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

def load_tokenize_and_split_datasets(tokenizer, max_length, train_ratio):
    """
    Tokenizes the input dataset and splits it into training and validation sets.

    Args:
        dataset (datasets.Dataset): The input dataset to be tokenized and split.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.
        max_length (int): The maximum sequence length for tokenization.
        train_ratio (float): The ratio of samples to use for training (between 0 and 1).

    Returns:
        tuple: A tuple containing the tokenized training and validation datasets.

    """
    df = pd.read_csv(cfg.DATASET_PATH)
    dataset = Dataset.from_pandas(df)
    tokenized_datasets = dataset.map(lambda examples: tokenizer(examples["text"],
                                                                padding="max_length",
                                                                truncation=True,
                                                                max_length=max_length),
                                                                batched=True)

    # Split tokenized dataset into training and validation sets
    num_samples = len(tokenized_datasets)
    num_train_samples = int(train_ratio * num_samples)
    # Randomly select samples for the training set
    train_indices = random.sample(range(num_samples), num_train_samples)
    train_dataset = tokenized_datasets.select(train_indices)
    # Select the remaining samples for the testing set
    eval_indices = list(set(range(num_samples)) - set(train_indices))
    eval_dataset = tokenized_datasets.select(eval_indices)
    return train_dataset, eval_dataset

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def train():
    """
    Trains a language model for causal language modeling.

    This function loads the dataset, tokenizer, and model from the configuration,
    defines training configurations and hyperparameters, creates a data collator,
    tokenizes and splits the dataset, and then trains the model.

    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.MODEL_NAME, load_in_8bit=True,
                                             torch_dtype = torch.float16)
    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    
    # Define training configurations and hyperparameters
    training_args = TrainingArguments(
    output_dir=cfg.OUTPUT_DIR,
    learning_rate=cfg.LEARNING_RATE,
    num_train_epochs=cfg.NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=cfg.PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=cfg.PER_DEVICE_EVAL_BATCH_SIZE,
    optim=cfg.OPTIM,
    fp16=True,
    gradient_accumulation_steps = cfg.GRADIENT_ACCUMULATION_STEPS,
    warmup_ratio=cfg.WARMUP_RATIO,
    weight_decay=cfg.WEIGHT_DECAY,
    eval_steps=cfg.EVAL_STEPS,
    save_strategy=cfg.SAVE_STRATEGY,
    logging_dir=cfg.LOGGING_DIR,
    logging_steps=cfg.LOGGING_STEPS,
    save_total_limit=cfg.SAVE_TOTAL_LIMIT,
    load_best_model_at_end=True,
    deepspeed=cfg.DEEPSPEED_CONFIG_PATH
    )


    # Create data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_datasets, eval_datasets = load_tokenize_and_split_datasets(tokenizer=tokenizer,
                                                                    max_length=cfg.MAX_LENGTH,
                                                                    train_ratio=cfg.TRAIN_RATIO)

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_datasets,
        eval_dataset=eval_datasets,
        data_collator=data_collator,
    )
    # Start training
    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    # Save the trained model
    trainer.save_model(cfg.OUTPUT_DIR)



if __name__ == "__main__":
    train()
