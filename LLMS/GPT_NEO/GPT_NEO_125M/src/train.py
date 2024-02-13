import random
import config as cfg

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)


def tokenize_and_split_datasets(dataset, tokenizer, max_length, train_ratio):
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


def train():
    """
    Trains a language model for causal language modeling.

    This function loads the dataset, tokenizer, and model from the configuration,
    defines training configurations and hyperparameters, creates a data collator,
    tokenizes and splits the dataset, and then trains the model.

    """
    dataset = load_dataset(cfg.DATASET_NAME, split="train")
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.MODEL_NAME)

    # Define training configurations and hyperparameters
    training_args = TrainingArguments(
        output_dir=cfg.OUTPUT_DIR,
        learning_rate=cfg.LEARNING_RATE,
        num_train_epochs=cfg.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=cfg.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=cfg.PER_DEVICE_EVAL_BATCH_SIZE,
        eval_steps=cfg.EVAL_STEPS,
        save_steps=cfg.SAVE_STEPS,
        warmup_steps=cfg.WARMUP_STEPS,
        weight_decay=cfg.WEIGHT_DECAY,
        logging_dir=cfg.LOGGING_DIR,
        logging_steps=cfg.LOGGING_STEPS,
    )

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_datasets, eval_datasets = tokenize_and_split_datasets(dataset=dataset,
                                                                tokenizer=tokenizer,
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


if __name__ == "__main__":
    train()
