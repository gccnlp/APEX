import copy
import torch
from apex_core import MixColumnLinear, MixRowLinear, S2ColumnLinear, S2RowLinear
import torch.nn as nn
import json
import os
import torch.distributed as dist
import random


def _ordered_unique(indices):
    unique_indices = []
    seen = set()
    for index in indices:
        if index not in seen:
            seen.add(index)
            unique_indices.append(index)
    return unique_indices


def merge_index_dicts(*index_dicts):
    merged = {}
    for index_dict in index_dicts:
        if not index_dict:
            continue
        for layer_idx, indices in index_dict.items():
            merged[layer_idx] = _ordered_unique(
                merged.get(layer_idx, []) + list(indices)
            )
    return merged


def remap_index_dict(index_dict, orders):
    remapped = {}
    for layer_idx, indices in index_dict.items():
        order = orders[layer_idx]
        inverse_order = [-1] * len(order)
        for new_position, old_position in enumerate(order):
            if not 0 <= old_position < len(order) or inverse_order[old_position] != -1:
                raise ValueError(f"Layer {layer_idx} order is not a permutation")
            inverse_order[old_position] = new_position
        if any(position == -1 for position in inverse_order):
            raise ValueError(f"Layer {layer_idx} order is not a permutation")
        remapped[layer_idx] = _ordered_unique(
            inverse_order[index] for index in indices
        )
    return remapped


class RecordingActivation(nn.Module):
    def __init__(self, original_act: nn.Module, layer_idx: int, act_record_method: str, k: int):
        super().__init__()
        self.original_act = original_act
        self.layer_idx = layer_idx
        self.recorded_activations = None
        self.act_record_method = act_record_method
        self.k = k
        self.forward_pass_count = 0
        self.compute_cv_loss=False
        self.enable_module_grad_checkpoint=False

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     output = self.original_act(x)
    #     with torch.no_grad():
    #         # Flatten the batch and sequence dimensions, then compute L2 norms over the hidden dimension.
    #         norms = torch.linalg.norm(output.view(-1,output.shape[-1]), dim=0).detach()
            
    #         if self.recorded_activations is None:
    #             self.recorded_activations = torch.zeros_like(norms)
    #         dist.all_reduce(norms, op=dist.ReduceOp.SUM)
    #         world_size = dist.get_world_size()
    #         # Divide the sum by world_size to obtain the average.
    #         current_activations = norms / world_size
    #         if self.is_record_avg_acts:
    #             self.recorded_activations = self.recorded_activations + current_activations
    #             self.forward_pass_count+=1
    #         else:
    #             self.recorded_activations = self.recorded_activations*0.9 + current_activations*0.1
    #         del norms
    #         del current_activations

    #     return output
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_module_grad_checkpoint and self.training:
            output = torch.utils.checkpoint.checkpoint(self.original_act,x,use_reentrant=False)  # Original FFN computation.
        else:
            output = self.original_act(x)  # Original FFN computation.
        if self.compute_cv_loss and self.training:
            self.current_activations = torch.linalg.norm(output.view(-1, output.shape[-1]), dim=0)
        with torch.no_grad():
            if self.compute_cv_loss and self.training:
                neuron_activations = self.current_activations.detach()
            else:
                neuron_activations = torch.linalg.norm(output.view(-1, output.shape[-1]), dim=0).detach()
            # Initialize the accumulator on the first call.
            if self.recorded_activations is None:
                self.recorded_activations = torch.zeros_like(
                    neuron_activations
                )
            if self.act_record_method=="avg":
                dist.all_reduce(neuron_activations, op=dist.ReduceOp.SUM)
                world_size = dist.get_world_size()
                # Divide the sum by world_size to obtain the average.
                batch_activations = neuron_activations / world_size
                self.recorded_activations = self.recorded_activations + batch_activations
                self.forward_pass_count+=1
                del neuron_activations
                del batch_activations

                return output
            elif self.act_record_method=="ma":
                dist.all_reduce(neuron_activations, op=dist.ReduceOp.SUM)
                world_size = dist.get_world_size()
                # Divide the sum by world_size to obtain the average.
                batch_activations = neuron_activations / world_size
                self.recorded_activations = self.recorded_activations*0.9 + batch_activations*0.1
                self.forward_pass_count+=1
                del neuron_activations
                del batch_activations

                return output
            elif self.act_record_method=="ma-rank":           
                topk_mask = torch.zeros_like(neuron_activations)
                _, topk_indices = torch.topk(neuron_activations, k=self.k)
                topk_mask[topk_indices] = True  # Mark the top-k entries.

                mink_mask = torch.zeros_like(neuron_activations)
                _, mink_indices = torch.topk(-neuron_activations, k=self.k)
                mink_mask[mink_indices] = True  # Mark the bottom-k entries.

                dist.all_reduce(topk_mask, op=dist.ReduceOp.SUM)
                dist.all_reduce(mink_mask, op=dist.ReduceOp.SUM)
            
                
                # Update the moving rank score.
                self.recorded_activations = self.recorded_activations * 0.9 + (topk_mask.float()- mink_mask.float())*0.1        

                # Release temporary tensors.
                neuron_activations=None
                topk_indices=None
                topk_mask=None
                mink_mask = None
            else:            
                topk_mask = torch.zeros_like(neuron_activations)
                _, topk_indices = torch.topk(neuron_activations, k=self.k)
                topk_mask[topk_indices] = True  # Mark the top-k entries.

                mink_mask = torch.zeros_like(neuron_activations)
                _, mink_indices = torch.topk(-neuron_activations, k=self.k)
                mink_mask[mink_indices] = True  # Mark the bottom-k entries.


                if dist.is_initialized():
                    dist.all_reduce(topk_mask, op=dist.ReduceOp.SUM)
                    dist.all_reduce(mink_mask, op=dist.ReduceOp.SUM)
            
                
                # Update the cumulative rank score.
                self.recorded_activations += topk_mask
                self.recorded_activations -= mink_mask           

                # Release temporary tensors.
                neuron_activations=None
                topk_indices=None
                topk_mask=None
                mink_mask = None
        
        return output

def only_optimize_apex_parameters(model, train_embedding_and_lm_head=False, train_layernorm=False):
    # Turn off gradients for parameters outside the APEX training set.
    print("train_embedding_and_lm_head:",train_embedding_and_lm_head)
    print("train_layernorm:",train_layernorm)
    for name, param in model.named_parameters():
        if "s2" in name:
            param.requires_grad = True
        elif "mixer" in name:
            param.requires_grad = True
        elif "blkdiag" in name:
            param.requires_grad = True
        elif "embed_tokens" in name and train_embedding_and_lm_head:
            param.requires_grad = True
        elif "lm_head" in name and train_embedding_and_lm_head:
            param.requires_grad = True
        elif "norm" in name and train_layernorm:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    print_count = 0
    for name, module in model.named_modules():
        if isinstance(module, MixColumnLinear) or isinstance(module, MixRowLinear) or isinstance(module, S2ColumnLinear) or isinstance(module, S2RowLinear):
            if isinstance(module, MixColumnLinear) and print_count < 1:
                print(module.mixer.blkdiag2)
                print(module.mixer_vec)
                print(module.mixer_scale)
                print_count+=1
    return model

# Convert the linear layers in the MHA module to APEX layers.
def convert_mha_layer_to_apex(model, selected_parameters, only_tuned_parameters=None, args=None):
    finetune_qk=args.finetune_qk
    mix_channels=not args.not_mix_channels
    find_largest_divisor=args.find_largest_divisor
    init_method=args.init_method
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    orders = {}
    for i in range(model.config.num_hidden_layers):
        layer = model.model.layers[i]
        o_indices = set(selected_parameters["o_proj"][i])
        vo = [
            index
            for index in selected_parameters["v_proj"][i]
            if index in o_indices
        ]
        paired_indices = set(vo)
        only_tuned = _ordered_unique(
            index
            for index in (only_tuned_parameters or {}).get(i, [])
            if index not in paired_indices
        )
        order = vo + only_tuned
        for j in range(model.config.num_attention_heads):
            if j not in order:
                order.append(j)
        orders[i] = order.copy()
        if len(vo) > 0:
            module = layer.self_attn.v_proj
            checkpoint = copy.deepcopy(module.state_dict())
            layer.self_attn.v_proj = MixColumnLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias,
                start=0,
                end=(len(vo)) * head_dim,
                tuned_end=(len(vo) + len(only_tuned)) * head_dim,
                device=next(module.parameters()).device,
                dtype=next(module.parameters()).dtype,
                mix_channels=mix_channels,
                find_largest_divisor=find_largest_divisor,
                init_method=init_method
            )
            layer.self_attn.v_proj.load_state_dict(checkpoint, strict=False)
            del module
            del checkpoint
            module = layer.self_attn.o_proj
            checkpoint = copy.deepcopy(module.state_dict())
            layer.self_attn.o_proj = MixRowLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias,
                start=0,
                end=(len(vo)) * head_dim,
                tuned_end=(len(vo) +len(only_tuned)) * head_dim,
                device=next(module.parameters()).device,
                dtype=next(module.parameters()).dtype,
                mix_channels=mix_channels,
                find_largest_divisor=find_largest_divisor,
                init_method=init_method
            )
            layer.self_attn.o_proj.load_state_dict(checkpoint, strict=False)
            layer.self_attn.o_proj.mixer = layer.self_attn.v_proj.mixer
            layer.self_attn.o_proj.mixer_vec = layer.self_attn.v_proj.mixer_vec
            del module
            del checkpoint
            if finetune_qk:
                module = layer.self_attn.q_proj
                checkpoint = copy.deepcopy(module.state_dict())
                layer.self_attn.q_proj = S2ColumnLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias,
                    start=0,
                    end=(len(vo) +len(only_tuned)) * head_dim,
                    device=next(module.parameters()).device,
                    dtype=next(module.parameters()).dtype,
                )
                layer.self_attn.q_proj.load_state_dict(checkpoint, strict=False)
                del module
                del checkpoint
                module = layer.self_attn.k_proj
                checkpoint = copy.deepcopy(module.state_dict())
                layer.self_attn.k_proj = S2ColumnLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias,
                    start=0,
                    end=(len(vo) +len(only_tuned)) * head_dim,
                    device=next(module.parameters()).device,
                    dtype=next(module.parameters()).dtype,
                )
                layer.self_attn.k_proj.load_state_dict(checkpoint, strict=False)
                del module
                del checkpoint
        if len(vo)==0 and len(only_tuned)>0:
            module = layer.self_attn.v_proj
            checkpoint = copy.deepcopy(module.state_dict())
            layer.self_attn.v_proj = S2ColumnLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias,
                start=0,
                end=(len(only_tuned)) * head_dim,
                device=next(module.parameters()).device,
                dtype=next(module.parameters()).dtype,
            )
            layer.self_attn.v_proj.load_state_dict(checkpoint, strict=False)
            del module
            del checkpoint
            module = layer.self_attn.o_proj
            checkpoint = copy.deepcopy(module.state_dict())
            layer.self_attn.o_proj = S2RowLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias,
                start=0,
                end=(len(only_tuned)) * head_dim,
                device=next(module.parameters()).device,
                dtype=next(module.parameters()).dtype,
            )
            layer.self_attn.o_proj.load_state_dict(checkpoint, strict=False)
            del module
            del checkpoint
            if finetune_qk:
                module = layer.self_attn.q_proj
                checkpoint = copy.deepcopy(module.state_dict())
                layer.self_attn.q_proj = S2ColumnLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias,
                    start=0,
                    end=(len(only_tuned)) * head_dim,
                    device=next(module.parameters()).device,
                    dtype=next(module.parameters()).dtype,
                )
                layer.self_attn.q_proj.load_state_dict(checkpoint, strict=False)
                del module
                del checkpoint
                module = layer.self_attn.k_proj
                checkpoint = copy.deepcopy(module.state_dict())
                layer.self_attn.k_proj = S2RowLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias,
                    start=0,
                    end=(len(only_tuned)) * head_dim,
                    device=next(module.parameters()).device,
                    dtype=next(module.parameters()).dtype,
                )
                layer.self_attn.k_proj.load_state_dict(checkpoint, strict=False)
                del module
                del checkpoint

        q_weight = layer.self_attn.q_proj.weight.data
        q_weight = q_weight.reshape(
            model.config.num_key_value_heads, -1, q_weight.shape[-1]
        )
        layer.self_attn.q_proj.weight.data = q_weight[order, :, :].reshape(
            -1, q_weight.shape[-1]
        )
        k_weight = layer.self_attn.k_proj.weight.data
        k_weight = k_weight.reshape(
            model.config.num_key_value_heads, -1, k_weight.shape[-1]
        )
        layer.self_attn.k_proj.weight.data = k_weight[order, :, :].reshape(
            -1, k_weight.shape[-1]
        )
        v_weight = layer.self_attn.v_proj.weight.data
        v_weight = v_weight.reshape(
            model.config.num_key_value_heads, -1, v_weight.shape[-1]
        )
        layer.self_attn.v_proj.weight.data = v_weight[order, :, :].reshape(
            -1, v_weight.shape[-1]
        )
        o_weight = layer.self_attn.o_proj.weight.data
        o_weight = o_weight.reshape(
            o_weight.shape[0], model.config.num_attention_heads, -1
        )
        layer.self_attn.o_proj.weight.data = o_weight[:, order, :].reshape(
            o_weight.shape[0], -1
        )

        del v_weight, o_weight
    return orders

def replace_attn_activations(model: nn.Module, args, merge_count):
    act_record_method = args.act_record_method
    print("act_record_method:",act_record_method)
    k = int(model.config.num_attention_heads * args.v_ratio / (2 ** merge_count))
    k = max(1, k)  # Select at least one head.
    for i in range(model.config.num_hidden_layers):
        layer = model.model.layers[i]
        layer.self_attn.record_attn_weights=True
        layer.self_attn.k = k
        layer.self_attn.act_record_method = act_record_method
        layer.self_attn.enable_module_grad_checkpoint=True
        if i>=5 and i<=25:
            layer.self_attn.compute_cv_loss = args.compute_cv_loss
    return model

def replace_mlp_activations(model: nn.Module, args, merge_count) -> None:
    """
    Replace each MLP activation function with a wrapper that records gate-projection activations.
    
    Args:
        model (nn.Module): Llama model instance.
    """
    act_record_method = args.act_record_method
    print("act_record_method:",act_record_method)
    k = int(model.config.intermediate_size * args.u_ratio / (2 ** merge_count))  # Compute the stage-adjusted selection count.
    k = max(1, k)  # Select at least one neuron.
    for i in range(model.config.num_hidden_layers):
        layer = model.model.layers[i]
        mlp = layer.mlp  # Get the MLP module for the current layer.
        # Check for the activation attribute used by Hugging Face LlamaMLP.
        if hasattr(mlp, 'act_fn'):
            original_act = mlp.act_fn
            # Wrap and replace the original activation function.
            mlp.act_fn = RecordingActivation(original_act, i, act_record_method, k)
            mlp.enable_module_grad_checkpoint = True
            mlp.act_fn.enable_module_grad_checkpoint=True
            if i>=5 and i<=25:
                mlp.act_fn.compute_cv_loss = args.compute_cv_loss
        else:
            raise AttributeError(f"MLP layer {i} has no attribute 'act_fn'")
    return model

def sync_activation_values(args, model, sub_folder="",old_mink_indexs=None, merge_count=1):
    topk_indexs = {}
    mink_indexs = {}
    mink_values_dict = {}
    topk_values_dict = {}
    mink_extk_indexs = {}
    extend_k=0
    k = int(model.config.intermediate_size * args.u_ratio / (2**merge_count))  # Compute the stage-adjusted selection count.
    if args.tuned_ffn_ratio is not None:
        print("tuned_ffn_ratio:",args.tuned_ffn_ratio)
        extend_k = int(model.config.intermediate_size * args.tuned_ffn_ratio)-k
    for i in range(model.config.num_hidden_layers):
        layer = model.model.layers[i]
        mlp = layer.mlp
        
        if not (hasattr(mlp, 'act_fn') and isinstance(mlp.act_fn, RecordingActivation)):
            continue
            
        module = mlp.act_fn
        activation = module.recorded_activations
        if args.random_assessment:
            if i==0:
                print("before:",activation)
            idx = torch.randperm(activation.size(0))
            activation = activation[idx]
            if i==0:
                print("after:",activation)
        
        # Prepare the index filter.
        all_indices = torch.arange(activation.size(0))            
        if old_mink_indexs and i in old_mink_indexs:
            mask = torch.ones_like(all_indices, dtype=bool)
            mask[old_mink_indexs[i]] = False
            filtered_indices = all_indices[mask]
            filtered_activations = activation[filtered_indices]
            cur_extend_k = extend_k - len(old_mink_indexs[i])
        else:
            filtered_indices = all_indices
            filtered_activations = activation
            cur_extend_k = extend_k
        
        # Select the top-k entries.
        topk_values, topk_indices_original = torch.topk(filtered_activations, k)
        topk_indices = filtered_indices[topk_indices_original.cpu()]
        topk_indexs[i] = topk_indices.cpu().tolist()
        topk_values_dict[i] = topk_values.cpu().tolist()
        
        # Select the bottom-k entries using the same filtered set.
        mink_values, mink_indices_original = torch.topk(-filtered_activations, k)
        mink_indices = filtered_indices[mink_indices_original.cpu()]
        mink_indices = mink_indices.cpu().tolist()
        mink_indexs[i] = mink_indices
        mink_values_dict[i] = (-mink_values).cpu().tolist()
        if args.tuned_ffn_ratio is not None:
            _, mink_indices_extk = torch.topk(-filtered_activations, cur_extend_k)
            min_extk_indices = filtered_indices[mink_indices_extk.cpu()]
            min_extk_indices = min_extk_indices.cpu().tolist()
            mink_extk_indexs[i] = list(set(min_extk_indices) - set(mink_indices))


    if args.reverse_topk_mink:
        print("Reverse topk mink !")
        temp = topk_indexs
        topk_indexs = mink_indexs
        mink_indexs = temp
     # Ensure that the output directory exists.
    os.makedirs(os.path.join(args.output_dir, sub_folder), exist_ok=True)

    # Save the dictionaries as JSON files.
    with open(os.path.join(os.path.join(args.output_dir, sub_folder), 'ffn_topk_indexs.json'), 'w') as f:
        json.dump(topk_indexs, f)

    with open(os.path.join(os.path.join(args.output_dir, sub_folder), 'ffn_mink_indexs.json'), 'w') as f:
        json.dump(mink_indexs, f)

    with open(os.path.join(os.path.join(args.output_dir, sub_folder), 'ffn_topk_values.json'), 'w') as f:
        json.dump(topk_values_dict, f)

    with open(os.path.join(os.path.join(args.output_dir, sub_folder), 'ffn_mink_values.json'), 'w') as f:
        json.dump(mink_values_dict, f)
    if args.u_ratio!=0:
        print("layer 0 FFN min 10 indexs:",mink_indexs[0][:10])
    if args.tuned_ffn_ratio is not None:
        with open(os.path.join(os.path.join(args.output_dir, sub_folder), 'ffn_mink_extk_indexs.json'), 'w') as f:
            json.dump(mink_extk_indexs, f)
        if args.u_ratio!=0:
            print("layer 0 FFN tuned 10 indexs:",mink_extk_indexs[0][:10])
    return topk_indexs, mink_indexs, mink_extk_indexs

def sync_head_values(args, model, sub_folder="",old_mink_indexs=None, merge_count=1):
    topk_indexs = {}
    mink_indexs = {}
    mink_values_dict = {}
    topk_values_dict = {}
    mink_extk_indexs = {}
    extend_k=0
    trainable_heads_budget = model.config.num_attention_heads * model.config.num_hidden_layers * args.v_ratio

    if old_mink_indexs is not None:
        print(sum(len(lst) for lst in old_mink_indexs.values()))
        trainable_heads_budget = trainable_heads_budget - sum(len(lst) for lst in old_mink_indexs.values())
        print(trainable_heads_budget)
    # Determine whether the budget permits selecting a pair of heads per layer.
    if trainable_heads_budget / model.config.num_hidden_layers < 2:
        # Compute how many layers receive a pair of selected heads.
        layers_to_allocate = int(trainable_heads_budget // 2)
        selected_layers = random.sample(range(model.config.num_hidden_layers), layers_to_allocate)
        k = 1  # Select one advantageous and one disadvantageous head per chosen layer.
    else:
        selected_layers = range(model.config.num_hidden_layers)
        # Compute the number of heads selected per layer.
        k = int(model.config.num_attention_heads * args.v_ratio / (2 ** merge_count))
    if args.tuned_attn_ratio is not None:
        print("tuned_attn_ratio:",args.tuned_attn_ratio)
        extend_k = int(model.config.num_attention_heads * args.tuned_attn_ratio)-k
    for i in range(model.config.num_hidden_layers):
        if trainable_heads_budget / model.config.num_hidden_layers < 2 and i not in selected_layers:
            topk_indexs[i] = []
            mink_indexs[i] = []
            continue
            
        layer = model.model.layers[i]
        head_importances = layer.self_attn.head_importances
        if args.random_assessment:
            if i==0:
                print("before:",head_importances)
            idx = torch.randperm(head_importances.size(0))
            head_importances = head_importances[idx]
            if i==0:
                print("after:",head_importances)
        
        # Enumerate all candidate indices.
        all_indices = torch.arange(model.config.num_attention_heads)
        
        # Exclude indices selected as disadvantageous in earlier stages.
        if old_mink_indexs and i in old_mink_indexs:
            mask = torch.ones_like(all_indices, dtype=bool)
            mask[old_mink_indexs[i]] = False
            filtered_indices = all_indices[mask]
            filtered_importances = head_importances[filtered_indices]
            cur_extend_k = extend_k - len(old_mink_indexs[i])
        else:
            filtered_indices = all_indices
            filtered_importances = head_importances
            cur_extend_k = extend_k
        
        # Select the top-k heads.
        topk_values, topk_indices_original = torch.topk(filtered_importances, k)
        topk_indices = filtered_indices[topk_indices_original.cpu()]
        topk_indexs[i] = topk_indices.tolist()
        topk_values_dict[i] = topk_values.cpu().tolist()
        
        # Select the bottom-k heads using the same filtered set.
        mink_values, mink_indices_original = torch.topk(-filtered_importances, k)
        mink_indices = filtered_indices[mink_indices_original.cpu()]
        mink_indices = mink_indices.cpu().tolist()
        mink_indexs[i] = mink_indices
        mink_values_dict[i] = (-mink_values).cpu().tolist()

        if args.tuned_attn_ratio is not None:
            _, mink_indices_extk = torch.topk(-filtered_importances, cur_extend_k)
            min_extk_indices = filtered_indices[mink_indices_extk.cpu()]
            min_extk_indices = min_extk_indices.cpu().tolist()
            mink_extk_indexs[i] = list(set(min_extk_indices) - set(mink_indices))
     # Ensure that the output directory exists.
    os.makedirs(os.path.join(args.output_dir, sub_folder), exist_ok=True)

    # Save the dictionaries as JSON files.
    with open(os.path.join(os.path.join(args.output_dir, sub_folder), 'attn_topk_indexs.json'), 'w') as f:
        json.dump(topk_indexs, f)

    with open(os.path.join(os.path.join(args.output_dir, sub_folder), 'attn_mink_indexs.json'), 'w') as f:
        json.dump(mink_indexs, f)
    
    with open(os.path.join(os.path.join(args.output_dir, sub_folder), 'attn_topk_values.json'), 'w') as f:
        json.dump(topk_values_dict, f)

    with open(os.path.join(os.path.join(args.output_dir, sub_folder), 'attn_mink_values.json'), 'w') as f:
        json.dump(mink_values_dict, f)
    if args.v_ratio!=0:
        print("layer 0 ATTN min indexs:",mink_indexs[0])
    if args.tuned_attn_ratio is not None and  args.tuned_attn_ratio!=0:
        with open(os.path.join(os.path.join(args.output_dir, sub_folder), 'attn_mink_extk_indexs.json'), 'w') as f:
            json.dump(mink_extk_indexs, f)
        if args.v_ratio!=0:
            print("layer 0 ATTN tuned 10 indexs:",mink_extk_indexs[0])
    return topk_indexs, mink_indexs, mink_extk_indexs

def restore_mlp_activations(model: nn.Module) -> None:
    """
    Restore the original activation function in every wrapped MLP layer.
    
    Args:
        model (nn.Module): Llama model instance.
    """
    for i in range(model.config.num_hidden_layers):
        layer = model.model.layers[i]
        mlp = layer.mlp  # Get the MLP module for the current layer.
        # Check whether the activation function is wrapped for recording.
        if hasattr(mlp, 'act_fn') and isinstance(mlp.act_fn, RecordingActivation):
            # Restore the original activation function.
            mlp.act_fn = mlp.act_fn.original_act
    return model

# Convert the linear layers in the FFN module to APEX layers.
def convert_ffn_layer_to_apex(model, selected_parameters, only_tuned_parameters=None, args=None):
    mix_channels=not args.not_mix_channels
    find_largest_divisor=args.find_largest_divisor
    init_method=args.init_method
    orders = {}
    for i in range(model.config.num_hidden_layers):
        layer = model.model.layers[i]
        ud = [
            j
            for j in selected_parameters["up_proj"][i]
            if j in selected_parameters["down_proj"][i]
        ]
        paired_indices = set(ud)
        only_tuned = _ordered_unique(
            index
            for index in (only_tuned_parameters or {}).get(i, [])
            if index not in paired_indices
        )
        order = ud + only_tuned
        for j in range(model.config.intermediate_size):
            if j not in order:
                order.append(j)
        orders[i] = order.copy()
        if len(ud) > 0:
            module = layer.mlp.up_proj
            checkpoint = copy.deepcopy(module.state_dict())
            layer.mlp.up_proj = MixColumnLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias,
                start=0,
                end=len(ud),
                tuned_end=len(only_tuned) + len(ud),
                device=next(module.parameters()).device,
                dtype=next(module.parameters()).dtype,
                mix_channels=mix_channels,
                find_largest_divisor=find_largest_divisor,
                init_method=init_method
            )
            layer.mlp.up_proj.load_state_dict(checkpoint, strict=False)
            del module
            del checkpoint

            # gate
            module = layer.mlp.gate_proj
            checkpoint = copy.deepcopy(module.state_dict())
            layer.mlp.gate_proj = MixColumnLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias,
                start=0,
                end=len(ud),
                tuned_end=len(only_tuned) + len(ud),
                device=next(module.parameters()).device,
                dtype=next(module.parameters()).dtype,
                mix_channels=mix_channels,
                find_largest_divisor=find_largest_divisor,
                init_method=init_method
            )
            layer.mlp.gate_proj.load_state_dict(checkpoint, strict=False)
            #layer.mlp.gate_proj.mixer.weight = layer.mlp.up_proj.mixer.weight
            layer.mlp.gate_proj.mixer = layer.mlp.up_proj.mixer
            layer.mlp.gate_proj.mixer_vec = layer.mlp.up_proj.mixer_vec
            del module
            del checkpoint

            # down_proj
            module = layer.mlp.down_proj
            checkpoint = copy.deepcopy(module.state_dict())
            layer.mlp.down_proj = MixRowLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias,
                start=0,
                end=len(ud),
                tuned_end=len(only_tuned) + len(ud),
                device=next(module.parameters()).device,
                dtype=next(module.parameters()).dtype,
                mix_channels=mix_channels,
                find_largest_divisor=find_largest_divisor,
                init_method=init_method
            )
            layer.mlp.down_proj.load_state_dict(checkpoint, strict=False)
            layer.mlp.down_proj.mixer = layer.mlp.up_proj.mixer
            layer.mlp.down_proj.mixer_vec = layer.mlp.up_proj.mixer_vec
            del module
            del checkpoint
        u_weight = layer.mlp.up_proj.weight.data
        layer.mlp.up_proj.weight.data = u_weight[order, :]
        g_weight = layer.mlp.gate_proj.weight.data
        layer.mlp.gate_proj.weight.data = g_weight[order, :]
        d_weight = layer.mlp.down_proj.weight.data
        layer.mlp.down_proj.weight.data = d_weight[:, order]
    return orders

# Fuse APEX layers back into linear layers.
def convert_apex_to_linear_layer(model):
    print_count = 0
    for name, module in model.named_modules():
        if isinstance(module, MixColumnLinear) or isinstance(module, MixRowLinear) or isinstance(module, S2ColumnLinear) or isinstance(module, S2RowLinear):
            if isinstance(module, MixColumnLinear) and print_count < 1:
                print(module.mixer.blkdiag2)
                print(module.mixer_vec)
                print(module.mixer_scale)
                print_count+=1
            module.fuse_s2_weight()
    # for i in range(model.config.num_hidden_layers):
    #     layer = model.model.layers[i]
    #     module = layer.mlp.gate_proj
    #     checkpoint = copy.deepcopy(module.state_dict())
    #     layer.mlp.gate_proj = nn.Linear(in_features=module.in_features,
    #                         out_features=module.out_features,bias=False)
    #     layer.mlp.gate_proj.load_state_dict(checkpoint, strict=False)
    #     del checkpoint
    #     del module

    #     module = layer.mlp.up_proj
    #     checkpoint = copy.deepcopy(module.state_dict())
    #     layer.mlp.up_proj = nn.Linear(in_features=module.in_features,
    #                         out_features=module.out_features,bias=False)
    #     layer.mlp.up_proj.load_state_dict(checkpoint, strict=False)
    #     del checkpoint
    #     del module

    #     module = layer.mlp.down_proj
    #     checkpoint = copy.deepcopy(module.state_dict())
    #     layer.mlp.down_proj = nn.Linear(in_features=module.in_features,
    #                         out_features=module.out_features,bias=False)
    #     layer.mlp.down_proj.load_state_dict(checkpoint, strict=False)
    #     del checkpoint
    #     del module
    #     layer.self_attn.head_importances=None
    #     module = layer.self_attn.v_proj
    #     checkpoint = copy.deepcopy(module.state_dict())
    #     layer.self_attn.v_proj = nn.Linear(in_features=module.in_features,
    #                         out_features=module.out_features,bias=False)
    #     layer.self_attn.v_proj.load_state_dict(checkpoint, strict=False)
    #     del checkpoint
    #     del module
        
    #     module = layer.self_attn.o_proj
    #     checkpoint = copy.deepcopy(module.state_dict())
    #     layer.self_attn.o_proj = nn.Linear(in_features=module.in_features,
    #                         out_features=module.out_features,bias=False)
    #     layer.self_attn.o_proj.load_state_dict(checkpoint, strict=False)
    #     del checkpoint
    #     del module
        
    return model

def compute_regu_loss(model):
    num_modules = 0
    epsilon = 1e-7       # Small constant that prevents division by zero.
    cv_penalty = 0.0
    for name, module in model.named_modules():         
        if hasattr(module, 'current_activations'):
            mean_neurons = torch.mean(module.current_activations)
            std_neurons = torch.std(module.current_activations)
            cv_loss = std_neurons / (mean_neurons + epsilon)
            num_modules+=1          
            cv_penalty += cv_loss
    return cv_penalty / (num_modules+epsilon)
#  cv loss old version
    # start_idx = module.up_proj.start
    # tuned_end_idx = module.up_proj.tuned_end
    # activations = module.act_fn.recorded_activations[start_idx:tuned_end_idx]
    # num_modules+=1 
    # # Compute the mean and standard deviation across neurons.
    # mean_neurons = torch.mean(activations)
    # std_neurons = torch.std(activations)
    
    # # Compute the coefficient of variation.
    # cv_neurons = std_neurons / (mean_neurons + epsilon)
    
    # # Average the current layer's coefficient of variation and accumulate the penalty.
    # cv_penalty += cv_neurons.mean()
# Use an exponential penalty for numerical stability.
    # regu_loss = 0.0
    # temperature = 1e-4
    # out_mean = 0
    # for name, module in model.named_modules():
    #     if hasattr(module, "mixer"):
    #         num_modules+=1 
    #         W1 = module.mixer_vec
    #         W2 = module.mixer.convert_to_dense_weight()
    #         #sum1 = torch.sum(W1)  # Compute the sum.
    #         #mean1 = torch.mean(torch.abs(W1))  # Compute the mean absolute value.
    #         #mean2 = torch.mean(torch.abs(W2))  # Compute the mean absolute value.
    #         mean2 = torch.mean(torch.pow(W2, 2))
    #         out_mean = mean2
            
    #         #penalty = sum1 - sum2
    #         penalty = torch.exp(-(mean2/temperature))

    #         regu_loss += penalty
    # #print(nonneg_loss)
    # #print(out_mean)
    # return lambda_reg * regu_loss/num_modules
# Frobenius-norm penalty.
    #     if hasattr(module, "mixer"):
    #         W1 = module.mixer_vec
    #         W2 = module.mixer
    #         norm1 = torch.norm(W1, p='fro')  # Compute the Frobenius norm.
    #         norm2 = torch.norm(W2, p='fro')  # Compute the Frobenius norm.

            
    #         # Compute a per-layer hinge-style penalty.
    #         penalty = torch.clamp(norm1 - norm2, min=0) ** 2
    #         # Add a penalty for negative mixer values.
    #         nonneg_penalty = torch.relu(-W2).pow(2).sum()  # Penalize all negative values in W2.
    #         regu_loss += penalty
    #         nonneg_loss += nonneg_penalty
    # #print(nonneg_loss)
    # return lambda_reg * regu_loss + lambda_nonneg * nonneg_loss
# Also penalize the mixer vector.
    #     regu_loss = 0.0
    #     nonneg_loss = 0.0
    #     lambda_nonneg = 100
    #     for name, module in model.named_modules():
    #         if hasattr(module, "mixer"):
    #             W1 = module.mixer_vec
    #             W2 = module.mixer.convert_to_dense_weight()
    #             sum1 = torch.sum(W1)  # Compute the sum.
    #             sum2 = torch.sum(torch.abs(W2))  # Compute the sum of absolute values.
    #             penalty = sum1 - sum2

    #             regu_loss += penalty
    #     #print(nonneg_loss)
    #     return lambda_reg * regu_loss
