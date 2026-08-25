import sys
import os
import gc
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import copy
import torch
import json
import random
import math
import time
import argparse
import deepspeed
import wandb
import torch.distributed as dist
# add this if enconter AttributeError: 'DeepSpeedCPUAdam' object has no attribute 'ds_opt_adam'
# deepspeed.ops.op_builder.CPUAdamBuilder().load()
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed import get_accelerator

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset

from torch.utils.data.distributed import DistributedSampler

from transformers import AutoModelForCausalLM, SchedulerType, get_scheduler, DataCollatorForSeq2Seq
from tqdm import tqdm
from utils.apex_utils import (
    convert_ffn_layer_to_apex,
    convert_mha_layer_to_apex,
    convert_apex_to_linear_layer,
    replace_mlp_activations,
    replace_attn_activations,
    restore_mlp_activations,
    only_optimize_apex_parameters,
    sync_activation_values,
    sync_head_values,
    compute_regu_loss,
    merge_index_dicts,
    remap_index_dict,
)
from utils.utils import (
    print_rank_0,
    to_device,
    set_random_seed,
    get_all_reduce_mean,
    int_or_float,
)

from utils.ds_utils import get_train_ds_config

from utils.model_utils import (
    load_hf_tokenizer,
    create_hf_model,
    save_hf_format,
    get_optimizer_grouped_parameters,
    make_model_gradient_checkpointing_compatible,
    print_throughput
)

from utils.data_utils import SupervisedDataset, DataCollatorForSupervisedDataset, load_tulu_dataset

def clear_hooks(model):
    for n, m in model.named_modules():         
        if hasattr(m, 'current_activations'):
            m.current_activations=m.current_activations.detach()
            del m.current_activations

def add_activation_statistics(model,args,merge_count):
    model = replace_mlp_activations(model,args,merge_count)
    model = replace_attn_activations(model,args,merge_count)
    # statistics_hook,model = replace_attn_activations(model)
    # global ATTN_HOOKS
    # ATTN_HOOKS = statistics_hook
    return model

def infer_large_act_channel(args, model, train_dataloader, device="cuda:0", seed=0):
    start_time = time.time()
    random.seed(seed)
     # 2. Create a deterministic data loader.
    if args.shuffle_infer_data:
        dataset_size = len(train_dataloader.dataset)
        all_indices = list(range(dataset_size))
        subset_indices = random.sample(all_indices, min(args.num_samples, dataset_size))
    else:
        subset_indices = list(range(min(args.num_samples, len(train_dataloader.dataset))))
    subset_dataset = Subset(train_dataloader.dataset, subset_indices)
    
    # Use DistributedSampler to ensure the same data distribution across ranks.
    sampler = DistributedSampler(
        subset_dataset,
        shuffle=False,
        drop_last=True,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank()
    )
    temp_dataloader = DataLoader(
        subset_dataset,
        batch_size=8 if train_dataloader.batch_size<8 else train_dataloader.batch_size,
        sampler=sampler,
        collate_fn=train_dataloader.collate_fn
    )

    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(temp_dataloader):
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
    end_time = time.time()
    print("infer cost time:",end_time-start_time)
    
def get_apex(args,model,ffn_topk_indexs,ffn_mink_indexs,attn_topk_indexs,attn_mink_indexs,s2_ffn_indexs=None,s2_attn_indexs=None):
    print("layer 0 FFN min 10 indexs:",ffn_mink_indexs[0][:10])
    print("layer 0 ATTN min indexs:",attn_mink_indexs[0])
    print("init_method:",args.init_method)
    attn_orders = {}
    ffn_orders = {}
    if args.v_ratio > 0:
        parameters_v = {}
        parameters_o = {}
        only_tuned_parameters = {}
        for layer_idx in attn_topk_indexs.keys():
            topk_index = attn_topk_indexs[layer_idx]
            mink_index = attn_mink_indexs[layer_idx]
            # Assert that the two index lists have equal lengths.
            assert len(topk_index) == len(mink_index), "Top-k and Min-k indices are not of the same length."
            parameters_v[layer_idx] = mink_index + topk_index
            parameters_o[layer_idx] = mink_index + topk_index   
        if s2_attn_indexs:
            for layer_idx in s2_attn_indexs.keys():
                only_tuned_parameters[layer_idx] = s2_attn_indexs[layer_idx]
        else:
            for layer_idx in attn_topk_indexs.keys():
                only_tuned_parameters[layer_idx] = []

        selected_parameters_mha = {"v_proj": parameters_v, "o_proj": parameters_o}

        attn_orders = convert_mha_layer_to_apex(model, selected_parameters_mha, only_tuned_parameters, args)
    

    if args.u_ratio > 0:
        parameters_u = {}
        parameters_d = {}
        only_tuned_parameters = {}
        for layer_idx in ffn_topk_indexs.keys():
            topk_index = ffn_topk_indexs[layer_idx]
            mink_index = ffn_mink_indexs[layer_idx]
            if s2_ffn_indexs:
                only_tuned_parameters[layer_idx] = s2_ffn_indexs[layer_idx]
            else:
                only_tuned_parameters[layer_idx] = []
            # Assert that the two index lists have equal lengths.
            assert len(topk_index) == len(mink_index), "Top-k and Min-k indices are not of the same length."
            parameters_u[layer_idx] = mink_index + topk_index
            parameters_d[layer_idx] = mink_index + topk_index            
        selected_parameters_ffn = {"up_proj": parameters_u, "down_proj": parameters_d}

        print("mix channels:",str(not args.not_mix_channels))
        ffn_orders = convert_ffn_layer_to_apex(model, selected_parameters_ffn, only_tuned_parameters, args)

    model = only_optimize_apex_parameters(model, args.train_embedding_and_lm_head, args.train_layernorm)
    model = make_model_gradient_checkpointing_compatible(model)

    print_rank_0(f"learning rate: {args.learning_rate}", args.global_rank)
    return model, ffn_orders, attn_orders

def _cleanup_gpus():

    print_rank_0("Clearing gpus before moving ahead ...")
    for _ in range(6):
        gc.collect()
        time.sleep(1)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

def parse_args():
    parser = argparse.ArgumentParser(description="APEX Training")
    parser.add_argument(
        "--data_path",
        nargs="*",
        default=["./LLM-Adapters/ft-training_set/commonsense_170k.json"],
        help="Path to the training dataset. Accepted format:"
        "1) a single data path, 2) multiple datasets in the"
        "form: dataset1-path dataset2-path ...",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1024,
        help="infer samples.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--val_set_size",
        type=int,
        default=0,
        help="Size of the validation set. If 0, no validation set is used.",
    )
    parser.add_argument(
        "--load_last_model", action="store_true", help="only save the last model"
    )
    parser.add_argument(
        "--eval_step",
        type=int,
        default=80,
        help="size of eval_step",
    )
    parser.add_argument(
        "--eval_delay",
        type=int_or_float,
        default=0,
        help="eval after certain steps if it is an integer, or eval after certain ratio of steps if it is a float",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--mixer_lr_scale",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float, default=0, help="Ratio of total training steps used for warmup.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="Training data type",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the model."
    )
    parser.add_argument(
        "--project_name", type=str, default=None
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable HF gradient checkpointing for model.",
    )
    parser.add_argument(
        "--not_save_checkpoints",
        action="store_true",
    )
    parser.add_argument(
        "--finetune_qk",
        action="store_true",
    )
    parser.add_argument(
        "--train_embedding_and_lm_head",
        action="store_true",
    )
    parser.add_argument(
        "--train_layernorm",
        action="store_true",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="dropout rate of the model."
    )
    # deepspeed features
    parser.add_argument(
        "--offload", action="store_true", help="Enable ZeRO Offload techniques."
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for Actor model (and clones).",
    )

    parser.add_argument(
        "--apex", action="store_true", help="use APEX for efficient training."
    )
    parser.add_argument(
        "--select_data", action="store_true", help="use subset"
    )
    parser.add_argument(
        "--merge_to_base_model", action="store_true", help="fuse APEX into the base model between stages."
    )
    parser.add_argument(
        "--track_old_min_values",
        action="store_true",
        help="retain earlier Min-K groups for direct training and exclude them from later selections.",
    )
    parser.add_argument(
        "--shuffle_infer_data", action="store_true", help="shuffle data used for advantage assessment."
    )
    parser.add_argument(
        "--shuffle_dataloader",action="store_true",
    )
    parser.add_argument(
        "--not_mix_channels", action="store_true", help="disable channel mixing in APEX."
    )
    parser.add_argument(
        "--reverse_topk_mink", action="store_true"
    )
    parser.add_argument(
        "--record_act_step",
        type=int,
        default=1000,
        help="size of record_act_step",
    )
    parser.add_argument(
        "--find_largest_divisor", action="store_true"
    )
    parser.add_argument(
        "--merge_by_steps_count",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--compute_cv_loss",
        action="store_true",
    )
    parser.add_argument(
        "--init_method",
        type=str,
        default="noise",
        choices=["noise", "random", "half", "zeros"],
        help="Training data type",
    )
    parser.add_argument(
        "--random_assessment",
        action="store_true",
    )
    parser.add_argument("--v_ratio", type=float, default=0.0)
    parser.add_argument("--u_ratio", type=float, default=0.0)
    parser.add_argument("--tuned_ffn_ratio", type=float, default=None)
    parser.add_argument("--tuned_attn_ratio", type=float, default=None)
    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=5e-5,
        help="lambda for regularization loss",
    )
    parser.add_argument(
        "--act_record_method", type=str, default="rank", choices=["rank", "avg", "ma","ma-rank"],
    )
    ## Tensorboard logging
    parser.add_argument(
        "--enable_tensorboard", action="store_true", help="Enable tensorboard logging"
    )
    parser.add_argument("--tensorboard_path", type=str, default="tensorboard")
    parser.add_argument(
        "--instruction_type", type=str, choices=["single", "multi"], default="single"
    )
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    ## init distributed training with deepspeed
    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()
    if args.local_rank==0 and "test" not in args.output_dir:
        max_retries = 3  # Maximum number of retries.
        retry_delay = 5   # Delay between retries in seconds.

        for _ in range(max_retries):
            try:
                wandb.init(
                    project=args.project_name,
                    name=args.output_dir.split("/")[-1]
                )
                break  # Exit the loop after successful initialization.
            except Exception as e:
                print(f"初始化失败: {e}")
                time.sleep(retry_delay)
        else:
            print("达到最大重试次数, 离线初始化")
            wandb.init(mode="offline", project=args.project_name,
                    name=args.output_dir.split("/")[-1])

    ds_config = get_train_ds_config(
        offload=args.offload,
        dtype=args.dtype,
        stage=args.zero_stage,
        enable_tensorboard=args.enable_tensorboard,
        tb_path=args.tensorboard_path,
        tb_name="APEX Training",
        profiler_path=args.output_dir,
    )

    ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
    ds_config["train_batch_size"] = (
        args.per_device_train_batch_size
        * torch.distributed.get_world_size()
        * args.gradient_accumulation_steps
    )

    set_random_seed(args.seed)
    torch.distributed.barrier()

    ## Load model and tokenizer
    tokenizer = load_hf_tokenizer(
        args.model_name_or_path, fast_tokenizer=True
    )  
    tokenizer.model_max_length = args.max_seq_len
    print_rank_0(f"Tokenizer: {tokenizer.model_max_length}", args.global_rank)

    print_rank_0(f"Loading model from {args.model_name_or_path}", args.global_rank)
    unwrapped_model = create_hf_model(
        AutoModelForCausalLM,
        args.model_name_or_path,
        tokenizer,
        dropout=args.dropout,
        use_vanilla_attention=args.apex
    )

    
    ## Load Data
    if len(args.data_path) == 1 and ".json" in args.data_path[0] and "tulu" not in args.data_path[0]:
        print_rank_0(f"------json Data: {args.data_path[0]}", args.global_rank)
        train_dataset = SupervisedDataset(
            data_path=args.data_path[0],
            tokenizer=tokenizer,
            instruction_type=args.instruction_type,
            args=args,
        )
        if args.val_set_size > 0:
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset,
                [len(train_dataset) - args.val_set_size, args.val_set_size],
            )
            print_rank_0(
                f"Train set size: {len(train_dataset)}, Val set size: {len(val_dataset)}",
                args.global_rank,
            )

        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    elif "tulu" in args.data_path[0]:
        train_dataset = load_tulu_dataset(
            data_path=args.data_path[0],
            tokenizer=tokenizer,
            args=args,
        )
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=unwrapped_model, padding="longest")
    else:
        raise ValueError(
            "Only json format is supported for now. Please check your data format."
        )

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        if args.val_set_size > 0:
            val_sampler = SequentialSampler(val_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        if args.val_set_size > 0:
            val_sampler = DistributedSampler(val_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
    )
    if args.val_set_size > 0:
        val_dataloader = DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=data_collator,
        )
    old_ffn_mink_indexs = {}
    old_attn_mink_indexs = {}
    if args.apex:
        merge_count = 1
        print_rank_0("------use APEX------", args.global_rank)
        unwrapped_model = add_activation_statistics(unwrapped_model,args,merge_count).to(device)
        infer_large_act_channel(
            args, unwrapped_model, train_dataloader, device, seed=args.seed
        )
        ffn_topk_indexs,ffn_mink_indexs,ffn_mink_extk_indexs = sync_activation_values(args, unwrapped_model,"",None,merge_count)
        attn_topk_indexs,attn_mink_indexs,attn_mink_extk_indexs = sync_head_values(args, unwrapped_model,"",None, merge_count)
        unwrapped_model = restore_mlp_activations(unwrapped_model)
        recorder_merge_count = merge_count + 1 if args.track_old_min_values else merge_count
        unwrapped_model = add_activation_statistics(unwrapped_model,args,recorder_merge_count)
        unwrapped_model.to('cpu')
        unwrapped_model, ffn_orders, attn_orders = get_apex(
            args,
            unwrapped_model,
            ffn_topk_indexs,
            ffn_mink_indexs,
            attn_topk_indexs,
            attn_mink_indexs,
            ffn_mink_extk_indexs,
            attn_mink_extk_indexs,
        )
        if args.track_old_min_values:
            if args.u_ratio > 0:
                old_ffn_mink_indexs = remap_index_dict(
                    merge_index_dicts(old_ffn_mink_indexs, ffn_mink_indexs),
                    ffn_orders,
                )
            if args.v_ratio > 0:
                old_attn_mink_indexs = remap_index_dict(
                    merge_index_dicts(old_attn_mink_indexs, attn_mink_indexs),
                    attn_orders,
                )
        _cleanup_gpus()
        
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        unwrapped_model, args.weight_decay, args.learning_rate, args.mixer_lr_scale
    )

    ## Init deepspeed optimizer
    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(
        optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95)
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    # if args.apex and args.merge_to_base_model:
    #     num_training_steps_for_scheduler = num_update_steps_per_epoch
    # else:
    num_training_steps_for_scheduler = args.num_train_epochs * num_update_steps_per_epoch
    if args.merge_by_steps_count>0:
        merge_steps = math.ceil(num_training_steps_for_scheduler * args.gradient_accumulation_steps / args.merge_by_steps_count)
        print("Merging by step:",merge_steps)
    else:
        merge_steps = num_update_steps_per_epoch * args.gradient_accumulation_steps # or len(train_dataloader)
        print("Merging by epoch.")
    #print("Total steps:",num_training_steps_for_scheduler)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
        num_training_steps=num_training_steps_for_scheduler,
    )
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=unwrapped_model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=False,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ## Training
    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            losses = get_all_reduce_mean(losses)
        except:
            pass
        try:
            perplexity = torch.exp(losses).item()
        except OverflowError:
            perplexity = float("inf")
        model.train()
        return perplexity, losses.item()

    print_rank_0("***** Running training *****", args.global_rank)
    args_dict = vars(args)
    formatted_args = json.dumps(args_dict, indent=4, sort_keys=True)
    print_rank_0(formatted_args, args.global_rank)

    # print trainable parameters
    num = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    print_rank_0(f"Number of trainable parameters: {num}", args.global_rank)
    num = sum(p.numel() for n,p in model.module.named_parameters() if p.requires_grad and "blkdiag" in n)
    print_rank_0(f"Number of trainable mixer parameters: {num}", args.global_rank)
    num = sum(p.numel() for n,p in model.module.named_parameters() if p.requires_grad and "s2" in n and "self_attn" in n)
    print_rank_0(f"Number of trainable attn s2 parameters: {num}", args.global_rank)
    num = sum(p.numel() for n,p in model.module.named_parameters() if p.requires_grad and "s2" in n and "mlp" in n)
    print_rank_0(f"Number of trainable mlp s2 parameters: {num}", args.global_rank)

    total_training_steps = args.num_train_epochs * num_update_steps_per_epoch
    current_step_count = 0
    mean_loss = 0.0
    lambda_reg = args.lambda_reg
    lr_plot = []
    best_val_loss = float("inf")
    final_saved_model_index = 0
    best_model = None
    args.eval_step = args.eval_step * args.gradient_accumulation_steps

    args.eval_delay = (
        args.eval_delay
        if isinstance(args.eval_delay, int)
        else int(args.eval_delay * total_training_steps)
    )

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank,
        )
        if args.shuffle_dataloader:
            print("Shuffle Train DataLoader By Epoch Number.")
            train_sampler.set_epoch(epoch)
        model.train()
        
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}")):
            lr_plot.append(
                lr_scheduler.get_last_lr()[1]
                if len(lr_scheduler.get_last_lr()) > 1
                else lr_scheduler.get_last_lr()[0]
            )
            start = time.time()
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            if args.apex:
                regu_loss = compute_regu_loss(model)
                # if args.only_cv_loss_for_debug:
                #     total_loss = args.lambda_reg * regu_loss
                # else:
                total_loss = loss + lambda_reg * regu_loss
            elif args.lambda_reg!=0:
                regu_loss = compute_regu_loss(model)
                total_loss = loss + lambda_reg * regu_loss
            else:
                total_loss = loss

            model.backward(total_loss)
            model.step()
            current_step_count += 1
            # Log metrics only on the main process.
            if torch.distributed.get_rank() == 0:
                wandb.log({
                    "loss": loss.item(),
                    "regu_loss": regu_loss.item() if args.compute_cv_loss else None,
                    "lr": optimizer.param_groups[0]['lr'],
                    "epoch": epoch
                })
            end = time.time()
            # print throughput every 100 steps
            if torch.distributed.get_rank() == 0 and step % 100 == 0:
                print_throughput(model.model, args, end - start, args.global_rank)
            mean_loss += loss.item()
            if current_step_count % 50 == 0:
                print_rank_0(
                    f"Mean Loss: {mean_loss/current_step_count}",
                    args.global_rank,
                )
                if args.apex:
                    print_rank_0(
                        f"Regu Loss: {regu_loss}",
                        args.global_rank,
                    )
            if current_step_count % args.record_act_step == 0:
                # infer_large_act_channel(args, model, train_dataloader, device, seed=current_step_count)
                # model.train()
                _cleanup_gpus()

            if (
                current_step_count % args.eval_step == 0
                and args.val_set_size > 0
                and not args.load_last_model
                and current_step_count >= args.eval_delay
            ):
                ppl, val_loss = evaluation(model, val_dataloader)
                print_rank_0(
                    f"Validation perplexity: {ppl}, Validation loss: {val_loss}",
                    args.global_rank,
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if args.global_rank == 0:
                        best_model = copy.deepcopy(model.module).to("cpu")
                    final_saved_model_index = current_step_count
            # if current_step_count%2==0:
            #     break
            merge_method_condition = args.apex and args.merge_to_base_model
            merge_step_condition = (current_step_count%merge_steps==0)
            # not merge at last step of last epoch
            if (epoch==(args.num_train_epochs-1)) and (step==(len(train_dataloader)-1)):
                merge_step_condition=False
            if merge_method_condition and merge_step_condition:
                print("Merge at step:",current_step_count)
                clear_hooks(model.module)
                if args.global_rank == 0:
                    checkpoint = copy.deepcopy(model.module).to("cpu")
                    if args.apex:
                        print_rank_0("fusing APEX into linear layers ...", args.global_rank)
                        checkpoint = convert_apex_to_linear_layer(checkpoint)
                    print_rank_0("saving the model ...", args.global_rank)
                    save_hf_format(checkpoint, tokenizer, args, "checkpoint-step{}".format(current_step_count))
                    del checkpoint
                torch.distributed.barrier()
                if args.apex:
                    if args.track_old_min_values:
                        merge_count += 1
                    ffn_topk_indexs, ffn_mink_indexs,ffn_mink_extk_indexs = sync_activation_values(
                        args,
                        model.module,
                        "checkpoint-step{}".format(current_step_count),
                        old_ffn_mink_indexs if args.track_old_min_values else None,
                        merge_count,
                    )
                    attn_topk_indexs, attn_mink_indexs,attn_mink_extk_indexs = sync_head_values(
                        args,
                        model.module,
                        "checkpoint-step{}".format(current_step_count),
                        old_attn_mink_indexs if args.track_old_min_values else None,
                        merge_count,
                    )

                current_learning_rates = [group['lr'] for group in optimizer.param_groups]
                print(current_learning_rates)
                optimizer.destroy()
                optimizer = None
                original_model = model.module
                original_model.cpu()
                model = None
                original_model = None
                lr_scheduler = None
                del model, optimizer, lr_scheduler, unwrapped_model
                torch.cuda.empty_cache()
                gc.collect()


                save_path =  os.path.join(args.output_dir,  "checkpoint-step{}".format(current_step_count))
                unwrapped_model = create_hf_model(
                    AutoModelForCausalLM,
                    save_path,
                    tokenizer,
                    dropout=args.dropout,
                    use_vanilla_attention=args.apex
                )
                if args.apex:
                    recorder_merge_count = merge_count + 1 if args.track_old_min_values else merge_count
                    unwrapped_model = add_activation_statistics(unwrapped_model,args,recorder_merge_count)
                    if args.track_old_min_values:
                        s2_ffn_indexs = merge_index_dicts(old_ffn_mink_indexs, ffn_mink_extk_indexs)
                        s2_attn_indexs = merge_index_dicts(old_attn_mink_indexs, attn_mink_extk_indexs)
                    else:
                        s2_ffn_indexs = ffn_mink_extk_indexs
                        s2_attn_indexs = attn_mink_extk_indexs
                    unwrapped_model, ffn_orders, attn_orders = get_apex(
                        args,
                        unwrapped_model,
                        ffn_topk_indexs,
                        ffn_mink_indexs,
                        attn_topk_indexs,
                        attn_mink_indexs,
                        s2_ffn_indexs,
                        s2_attn_indexs,
                    )
                    if args.track_old_min_values:
                        if args.u_ratio > 0:
                            old_ffn_mink_indexs = remap_index_dict(
                                merge_index_dicts(old_ffn_mink_indexs, ffn_mink_indexs),
                                ffn_orders,
                            )
                        if args.v_ratio > 0:
                            old_attn_mink_indexs = remap_index_dict(
                                merge_index_dicts(old_attn_mink_indexs, attn_mink_indexs),
                                attn_orders,
                            )
                torch.distributed.barrier()

                optimizer_grouped_parameters = get_optimizer_grouped_parameters(
                    unwrapped_model, args.weight_decay, args.learning_rate, args.mixer_lr_scale
                )
                optimizer = AdamOptimizer(
                    optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95)
                )
                for param_group, lr in zip(optimizer.param_groups, current_learning_rates):
                    param_group['lr'] = lr
                num_training_steps_for_scheduler_new = math.ceil((num_training_steps_for_scheduler*args.gradient_accumulation_steps - current_step_count)/args.gradient_accumulation_steps)
                lr_scheduler = get_scheduler(
                    name=args.lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=int(num_training_steps_for_scheduler_new * 0.005),
                    num_training_steps=num_training_steps_for_scheduler_new,
                )
                model, optimizer, _, lr_scheduler = deepspeed.initialize(
                    model=unwrapped_model,
                    optimizer=optimizer,
                    args=args,
                    config=ds_config,
                    lr_scheduler=lr_scheduler,
                    dist_init_required=False,
                )
                num = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
                print_rank_0(f"Number of trainable parameters: {num}", args.global_rank)
                if args.gradient_checkpointing:
                    model.gradient_checkpointing_enable()
                _cleanup_gpus()
                # Ensure all processes have finished clearing the optimizer state.
                torch.distributed.barrier()
                if args.global_rank==0:
                    for filename in os.listdir(save_path):
                        if filename.endswith(".bin") or filename.endswith(".model"):
                            file_path = os.path.join(save_path, filename)
                            try:
                                os.remove(file_path)
                                print(f"Deleted: {file_path}")
                            except Exception as e:
                                print(f"Error deleting {file_path}: {e}")
                                
                model.train()


        print_rank_0(
            f"Epoch {epoch+1}/{args.num_train_epochs} Train loss: {mean_loss/len(train_dataloader)}",
            args.global_rank,
        )
        if (epoch+1)!=args.num_train_epochs:
            if args.global_rank == 0 and not args.not_save_checkpoints:
                clear_hooks(model.module)
                checkpoint = copy.deepcopy(model.module).to("cpu")
                if args.apex:
                    print_rank_0("fusing APEX into linear layers ...", args.global_rank)
                    checkpoint = convert_apex_to_linear_layer(checkpoint)
                print_rank_0("saving the model ...", args.global_rank)
                save_hf_format(checkpoint, tokenizer, args, "checkpoint-epoch{}".format(epoch))
                
                del checkpoint
        model.tput_timer.update_epoch_count()
        
    # Evaluation and Save
    if args.output_dir is not None:
        # evaluate last model
        if args.val_set_size > 0 and not args.load_last_model:
            ppl, val_loss = evaluation(model, val_dataloader)
            print_rank_0(
                f"Validation perplexity: {ppl}, Validation loss: {val_loss}",
                args.global_rank,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if args.global_rank == 0:
                    best_model = copy.deepcopy(model.module).to("cpu")
                final_saved_model_index = "last"
            print_rank_0(f"Best validation loss: {best_val_loss}", args.global_rank)
            print_rank_0(
                f"Savings the best model at step {final_saved_model_index}",
                args.global_rank,
            )

        if args.load_last_model:
            print_rank_0("only load the last model ...", args.global_rank)

        model = best_model.to(device) if best_model else model
        clear_hooks(model.module)
        model = copy.deepcopy(model.module).to("cpu")
        if args.apex:
            print_rank_0("fusing APEX into linear layers ...", args.global_rank)
            model = convert_apex_to_linear_layer(model)

        if args.global_rank == 0:
            print_rank_0("saving the model ...", args.global_rank)
            save_hf_format(model, tokenizer, args)
            wandb.finish()
        torch.save(lr_plot, os.path.join(args.output_dir, "lr_plot.pt"))
        torch.distributed.barrier()

if __name__ == "__main__":
    main()
