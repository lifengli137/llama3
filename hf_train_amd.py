import gc
import sys
import json
import torch
import logging
import warnings
import datasets
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, HfArgumentParser, TrainingArguments
from transformers.data.data_collator import DataCollatorForTokenClassification

def prepare_chat_data(tokenizer, max_length, fn):
    dataset = datasets.load_dataset("parquet", data_files=[fn])
    logging.info(f"Initial dataset size: {dataset['train'].num_rows}")
    
    def filter_labels(example):  
        return not all(value == -100 for value in example['labels'])  
    
    def process(example):
        prompt = example['instruction']
        response = example['response']
        prompt_ids = tokenizer(prompt, add_special_tokens=False, padding=False).input_ids
        response_ids = tokenizer(response, add_special_tokens=False, padding=False).input_ids
        
        input_ids = prompt_ids + response_ids        
        labels = [-100] * len(prompt_ids)  + response_ids
        
        assert len(input_ids) == len(labels), 'Input id and label mismatch!'
        
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
         
        return {"input_ids": input_ids, "labels": labels}
 
    dataset = dataset.map(process, batched=False, remove_columns=list(dataset["train"].column_names), num_proc=8)
    dataset = dataset.filter(filter_labels)
    print(dataset)
    
    return dataset
  

def cleanup_memory():
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    parser = HfArgumentParser(TrainingArguments)
    parser.add_argument("--model_name_or_dir")
    parser.add_argument("--data_path", default='')
    parser.add_argument('--max_seq_len', default=1024)
    
    training_args, args = parser.parse_args_into_dataclasses()

    max_length = int(args.max_seq_len) # 1024 is default
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_dir, use_flash_attention_2=True, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_dir, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None: # llama models, no padded token
        tokenizer.pad_token = tokenizer.eos_token
    
    ds = prepare_chat_data(tokenizer, max_length, args.data_path)

    if isinstance(ds, datasets.DatasetDict):
        split_ds = ds['train'].train_test_split(test_size=0.01, seed=42)
    else:
        split_ds = ds.train_test_split(test_size=0.01, seed=42)
        
    print (split_ds)
        
    data_collator = DataCollatorForTokenClassification(tokenizer, padding="max_length", return_tensors="pt", max_length=max_length)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_ds['train'],
        eval_dataset=split_ds['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None
    )

    if transformers.integrations.is_wandb_available():
        from transformers.integrations import WandbCallback
        trainer.remove_callback(WandbCallback)
        
    if training_args.do_train:
        trainer.train()
        trainer.save_state()
        
