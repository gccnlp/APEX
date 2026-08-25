import copy
import logging
import json
import torch
import transformers
import datasets
import random
from datasets import load_dataset
from torch.utils.data import Dataset
from functools import partial
from dataclasses import dataclass
from typing import Dict, Sequence
import os
from utils.utils import print_rank_0

IGNORE_INDEX = -100

BASE_PROMPT = """<s> Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{instruction}
                
### Response:
"""


BASE_PROMPT_WITH_INPUT = """<s> Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Input:
{input}
                
### Response:
"""

def encode_with_messages_format(example, tokenizer, max_seq_length, add_bos=False):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')
    
    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text
        
    example_text = _concat_messages(messages).strip()
    if add_bos:
        example_text = tokenizer.bos_token + example_text
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[:message_idx+1]) + "<|assistant|>\n"
            else:
                messages_so_far = _concat_messages(messages[:message_idx+1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt', 
                max_length=max_seq_length, 
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100
            
            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }
def load_json_data(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return [json.loads(line) for line in file]
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON: {e}")
            return None


def get_output_or_chosen(example):
    if "output" in example:
        return example["output"]
    elif "chosen" in example:
        return example["chosen"]
    else:
        raise ValueError("double check your data format")


def get_instruction_or_prompt(example):
    if "instruction" in example:
        return example["instruction"]
    elif "prompt" in example:
        return example["prompt"]
    else:
        raise ValueError("double check your data format")


def get_alpaca_prompt(example):
    if "input" in example and example["input"] != "":
        return BASE_PROMPT_WITH_INPUT.format_map(
            {"instruction": example["instruction"], "input": example["input"]}
        )
    else:
        return BASE_PROMPT.format_map({"instruction": example["instruction"]})


def get_output_or_chosen(example):
    if "output" in example:
        return example["output"]
    elif "chosen" in example:
        return example["chosen"]
    elif "answer" in example:
        return example["answer"].split("####")[0].strip()
    elif "Rationale" in example:
        return example["Rationale"]
    elif "rationale" in example:
        return example["rationale"]
    elif "solution" in example:
        return example["solution"]
    else:
        raise ValueError("double check your data format")


def get_instruction_or_prompt(example):
    if "input" in example and example["input"] != "":
        return example["input"]
    elif "instruction" in example:
        return example["instruction"]
    elif "prompt" in example:
        return example["prompt"]
    elif "question" in example:
        return example["question"]
    elif "Problem" in example:
        return example["Problem"]
    elif "problem" in example:
        return example["problem"]
    else:
        raise ValueError("double check your data format")


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    ids_list = tokenizer(
        strings,
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_attention_mask=False,
    )["input_ids"]

    input_ids = []
    input_ids_lens = []

    for ids in ids_list:
        input_ids.append(torch.tensor(ids))
        input_ids_lens.append(len(ids))

    return dict(
        input_ids=input_ids,
        input_ids_lens=input_ids_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    print_rank_0("-----------------")
    print_rank_0(examples[0])
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

def load_tulu_dataset(data_path: str, tokenizer: transformers.PreTrainedTokenizer, args):
    data_files = {}
    dataset_args = {}
    
    if args.select_data:
        print("Using Subset Dataset.")
        full_data_path = args.data_path[0]
        directory = os.path.dirname(full_data_path)
        # Define the path used to cache the subset.
        subset_data_path = os.path.join(directory, "subset_data.json")

        if os.path.exists(subset_data_path):
            print("Loading From Cache...")
            data_files["train"] = subset_data_path
            # Load the cached subset when it already exists.
           # Load the dataset.
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                **dataset_args,
            )
        else:
            print("Creating Subset...")
            data_files["train"] = args.data_path[0]
            # Load the full dataset.
            raw_datasets = load_dataset(
                "json",
                data_files=full_data_path,
                **dataset_args,
            )
            
            # Select the first 100,000 records.
            subset = raw_datasets['train'].select(range(100000))
            
            # Save the subset as a new file.
            subset.to_json(subset_data_path)
            raw_datasets['train'] = subset

    else:
        data_files["train"] = args.data_path[0]
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            **dataset_args,
        )
    print(args.max_seq_len)
    encode_function = partial(
        encode_with_messages_format,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_len,
        add_bos=False,
    )
    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=64,
        load_from_cache_file=True,
        remove_columns=[name for name in raw_datasets["train"].column_names if name not in ["input_ids", "labels", "attention_mask"]],
        desc="Tokenizing and reformatting instruction data",
    )
    torch.distributed.barrier()
    lm_datasets.set_format(type="pt")
    lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any())
    train_dataset = lm_datasets["train"]
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logging.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    return train_dataset
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        instruction_type: str,
        args,
    ):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = load_json_data(data_path)  # try both formats
        logging.warning("Formatting inputs...")

        # We might want to clean this up, it's a bit messy
        if instruction_type == "single":
            print_rank_0("single-round conversation", args.global_rank)
            if "chat" not in args.model_name_or_path:
                print_rank_0("base model", args.global_rank)
                if "alpaca" in data_path:
                    sources = [get_alpaca_prompt(example) for example in list_data_dict]
                else:
                    sources = [
                        BASE_PROMPT.format_map(
                            {"instruction": get_instruction_or_prompt(example)}
                        )
                        for example in list_data_dict
                    ]
            else:
                print_rank_0("chat model", args.global_rank)
                sources = []
                for example in list_data_dict:
                    chat = [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant. You will be given a user's question, and you need to answer it.",
                        },
                        {"role": "user", "content": get_instruction_or_prompt(example)},
                    ]
                    source = tokenizer.apply_chat_template(chat, tokenize=False)
                    source += " "
                    sources.append(source)

        targets = [
            f"{get_output_or_chosen(example).replace('</s>', '')} {tokenizer.eos_token}"
            for example in list_data_dict
        ]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
