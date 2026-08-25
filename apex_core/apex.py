import torch
import math

from torch import nn
from torch.nn.modules import Module
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from .structured_linear import StructuredLinear
from .blockdiag_butterfly_multiply import blockdiag_butterfly_multiply

def find_divisor(k, find_largest_divisor=False):
    sqrt_k = math.sqrt(k)
    if find_largest_divisor:
        if k == 1:
            return None  # 1 has no proper divisors
    
        # Find the smallest factor starting from 2
        for i in range(2, int(sqrt_k) + 1):
            if k % i == 0:
                return k // i  # Return the corresponding larger factor
        
        # If k is prime, its largest proper divisor is 1
        return 1
    else:  
        # Return immediately if the square root is an integer
        if sqrt_k.is_integer():
            return int(sqrt_k)
        
        # Collect all factors
        factors = set()
        for i in range(1, int(sqrt_k) + 1):
            if k % i == 0:
                factors.add(i)
                factors.add(k // i)
        factors = sorted(factors)
        
        # Find the factor closest to the square root
        min_diff = float('inf')
        result = None
        for factor in factors:
            diff = abs(factor - sqrt_k)
            if diff < min_diff:
                min_diff = diff
                result = factor
            elif diff == min_diff:
                # Choose the smaller factor when distances are equal
                if factor < result:
                    result = factor
    
        return result

class MonarchLinear(StructuredLinear):

    def __init__(self, *args, nblocks=4, init_method="noise", **kwargs):
        super().__init__(*args, **kwargs)
        in_blksz = int(math.ceil(self.in_features / nblocks))
        out_blksz = int(math.ceil(self.out_features / nblocks))
        self.in_features_extended = in_blksz * nblocks
        self.out_features_extended = out_blksz * nblocks
        self.init_method = init_method

        if self.in_features_extended < self.out_features_extended:
            self.blkdiag1 = nn.Parameter(torch.empty(nblocks, in_blksz, in_blksz), requires_grad=True)
            self.blkdiag2 = nn.Parameter(torch.empty(nblocks, out_blksz, in_blksz), requires_grad=True)
        else:
            self.blkdiag1 = nn.Parameter(torch.empty(nblocks, out_blksz, in_blksz), requires_grad=True)
            self.blkdiag2 = nn.Parameter(torch.empty(nblocks, out_blksz, out_blksz), requires_grad=True)
        self.reset_parameters()
        #logger.info(f'Linear class {self.__class__}: saving={self.saving}')

    def reset_parameters(self) -> None:
        # Mimic init.kaiming_uniform: https://github.com/pytorch/pytorch/blob/24087d07ca7ffa244575d259711dd7c99245a67a/torch/nn/init.py#L360
        if self.init_method=="zeros":
            nn.init.zeros_(self.blkdiag1)
            nn.init.zeros_(self.blkdiag2)
        elif self.init_method=="half":
            raise NotImplementedError('waiting for implementation')
        elif self.init_method=="random":
            nn.init.kaiming_uniform_(self.blkdiag1, a=math.sqrt(5))
            nn.init.zeros_(self.blkdiag2)
        else:
            for blkdiag in [self.blkdiag1, self.blkdiag2]:
                fan_in = blkdiag.shape[-1]
                gain = init.calculate_gain(nonlinearity='leaky_relu', param=math.sqrt(5))
                std = gain / math.sqrt(fan_in)
                bound = math.sqrt(0.0003) * std  # Calculate uniform bounds from standard deviation
                with torch.no_grad():
                    blkdiag.uniform_(-bound, bound)
                    
        self.reset_parameters_bias()

    @property
    def saving(self):
        return ((self.blkdiag1.numel() + self.blkdiag2.numel())
                / (self.in_features * self.out_features))

    def forward_matmul(self, x):
        output = blockdiag_butterfly_multiply(self.preprocess(x), self.blkdiag1, self.blkdiag2)
        return self.postprocess(output)

class MixColumnLinear(Module):

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        start=None,
        end=None,
        low_end=None,
        tuned_end=None,
        device=None,
        dtype=None,
        mix_channels=True,
        find_largest_divisor=False,
        init_method="noise"
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs),
            requires_grad=True,
        )
        if bias:
            self.bias = Parameter(
                torch.empty(out_features, **factory_kwargs), requires_grad=True
            )
        else:
            self.register_parameter("bias", None)
        if (end-start)%2!=0:
            end = end+1
        self.start = start
        if low_end is None:
            low_end = (end-start) // 2
        if tuned_end is None:
            tuned_end = end
        self.tuned_end = tuned_end
        self.low_end = low_end
        self.end = end
        self.s2 = nn.Parameter(
            torch.zeros(tuned_end - start, in_features), requires_grad=True
        )
        self.mixer_vec = nn.Parameter(torch.ones(1, low_end - start), requires_grad=True)
        self.nblocks = find_divisor(low_end-start, find_largest_divisor)
        self.init_method=init_method
        self.mixer = MonarchLinear(in_features=low_end - start, out_features=low_end - start, bias=False, nblocks=self.nblocks, init_method=self.init_method)
        #self.mixer = nn.Parameter(torch.zeros(1, low_end - start), requires_grad=True)
        self.mixer_scale = nn.Parameter(torch.ones(1, low_end - start), requires_grad=True)
        #self.mixer = nn.Linear(end - low_end, low_end - start, bias=False)
        self.mix_channels=mix_channels
        
        #print("merge channels:",str(self.mix_channels))
        self.reset_parameters()
        self.weight.requires_grad = False
        self.fused = False

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        self.mixer.reset_parameters()

        
        # for i in range((self.end - self.start)//2):
        #     weight[i, i] = 1
        # with torch.no_grad():
        #     self.mixer.weight.copy_(weight)
        

    def fuse_s2_weight(self):
        if self.fused == True:
            return
        self.weight.data[self.start : self.tuned_end, :] += self.s2
        if not self.mix_channels:
            with torch.no_grad():
                monarch_weight = self.mixer.convert_to_dense_weight()
                self.weight.data[self.start : self.low_end, :] = self.weight.data[self.start : self.low_end, :].mul(self.mixer_vec.data.T)
                self.weight.data[self.low_end : self.end, :] = monarch_weight @ self.weight.data[self.low_end : self.end, :]+self.weight.data[self.low_end : self.end, :].mul(self.mixer_scale.data.T)
        else:
            with torch.no_grad():
                monarch_weight = self.mixer.convert_to_dense_weight()
                self.weight.data[self.start : self.low_end, :] = self.weight.data[self.start : self.low_end, :].mul(self.mixer_vec.data.T) + monarch_weight @ self.weight.data[self.low_end : self.end, :]
                self.weight.data[self.low_end : self.end, :] = self.weight.data[self.low_end : self.end, :].mul(self.mixer_scale.data.T)
        self.fused = True

    def unfuse_s2_weight(self):
        if self.fused == False:
            return
        self.weight[self.start : self.tuned_end, :] -= self.s2
        self.fused = False

    def forward(self, input: Tensor) -> Tensor:
        base_output = torch.nn.functional.linear(input, self.weight, self.bias)
        if self.fused:
            return base_output
        else:
            if not self.mix_channels:
                s2_output = torch.nn.functional.linear(input, self.s2, None)
                # Extract the middle segment and add s2_output
                modified_channel = base_output[..., self.start:self.tuned_end] + s2_output
                # Split the middle segment into low and high halves, retaining the updated high half
                modified_high_channel = modified_channel[..., self.low_end:self.end]
                modified_low_channel = modified_channel[..., :self.low_end]
                # Use the mixer to produce the new low half
                mixed_low_part =  modified_low_channel * self.mixer_vec
                #mixed_low_part = modified_high_channel * self.mixer + modified_low_channel * self.mixer_vec
                mixed_high_part = self.mixer(modified_high_channel) + modified_high_channel * self.mixer_scale
                # Construct the tuned segment from the mixed pair and direct-update tail
                direct_tuned_part = modified_channel[..., self.end - self.start :]
                new_middle = torch.cat(
                    [mixed_low_part, mixed_high_part, direct_tuned_part], dim=-1
                )
                
                # Concatenate all segments
                output = torch.cat([
                    base_output[..., :self.start],
                    new_middle,
                    base_output[..., self.tuned_end:]
                ], dim=-1)
                return output
            else:
                s2_output = torch.nn.functional.linear(input, self.s2, None)

                # Extract the middle segment and add s2_output
                modified_channel = base_output[..., self.start:self.tuned_end] + s2_output
                
                # Split the middle segment into low and high halves, retaining the updated high half
                modified_high_channel = modified_channel[..., self.low_end:self.end]
                modified_low_channel = modified_channel[..., :self.low_end]
                
                # Use the mixer to produce the new low half
                #mixed_high = checkpoint(self.mixer, modified_high_channel)  # Recompute instead of storing
                #mixed_low_part = mixed_high + modified_low_channel * self.mixer_vec
                mixed_low_part = self.mixer(modified_high_channel) + modified_low_channel * self.mixer_vec
                mixed_high_part = modified_high_channel * self.mixer_scale
                
                # Construct the tuned segment from the mixed pair and direct-update tail
                direct_tuned_part = modified_channel[..., self.end - self.start :]
                new_middle = torch.cat(
                    [mixed_low_part, mixed_high_part, direct_tuned_part], dim=-1
                )
                
                # Concatenate all segments
                output = torch.cat([
                    base_output[..., :self.start],
                    new_middle,
                    base_output[..., self.tuned_end:]
                ], dim=-1)
                return output
            # base_output[:, :, self.start : self.end] += s2_output
            # # Use the mixer output without in-place assignment
            # mixer_output = self.mixer(base_output[:, :, self.start : self.end])
            # base_output[:, :, self.start : (self.end+1)//2] = mixer_output
            #base_output[:, :, self.start : (self.end+1)//2] = self.mixer(base_output[:, :, self.start : self.end])
            

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

class MixRowLinear(Module):

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        start=None,
        end=None,
        tuned_end=None,
        low_end=None,
        device=None,
        dtype=None,
        mix_channels=True,
        find_largest_divisor=False,
        init_method="noise"
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs),
            requires_grad=True,
        )
        if bias:
            self.bias = Parameter(
                torch.empty(out_features, **factory_kwargs), requires_grad=True
            )
        else:
            self.register_parameter("bias", None)
        
        if (end-start)%2!=0:
            end = end+1
        self.start = start
        if low_end is None:
            low_end = (end-start) // 2
        if tuned_end is None:
            tuned_end = end
        self.tuned_end = tuned_end
        self.low_end = low_end
        self.end = end
        self.mixer_vec = nn.Parameter(torch.ones(1, low_end - start), requires_grad=True)
        #self.mixer = nn.Parameter(torch.zeros(1, low_end - start), requires_grad=True)
        self.nblocks = find_divisor(low_end-start, find_largest_divisor)
        # print("shape:",low_end-start)
        # print("nblocks:",self.nblocks)
        self.init_method=init_method
        self.mixer = MonarchLinear(in_features=low_end - start, out_features=low_end - start, bias=False, nblocks=self.nblocks, init_method=self.init_method)
        self.s2 = nn.Parameter(
            torch.zeros(out_features, tuned_end - start), requires_grad=True
        )
        self.mix_channels = mix_channels
        
        #print("merge channels:",str(self.mix_channels))
        self.reset_parameters()
        self.weight.requires_grad = False
        self.fused = False

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        self.mixer.reset_parameters()

    def fuse_s2_weight(self):
        if self.fused == True:
            return
        with torch.no_grad():
            self.weight.data[:,self.start : self.tuned_end] += self.s2
            # self.weight.data[:, self.start : self.low_end] = (
            #     self.weight.data[:, self.start : self.low_end] * self.mixer_vec.data
            #     + self.weight.data[:, self.low_end : self.end] * self.mixer.data
            # )
            if not self.mix_channels:
                monarch_weight = self.mixer.convert_to_dense_weight()
                self.weight.data[:, self.start : self.low_end] = self.weight.data[:, self.start : self.low_end] * self.mixer_vec.data
                self.weight.data[:, self.low_end : self.end] = self.weight.data[:, self.low_end : self.end] @ monarch_weight.t() + self.weight.data[:, self.low_end : self.end]
            else:
                monarch_weight = self.mixer.convert_to_dense_weight()
                self.weight.data[:, self.start : self.low_end] = (
                    self.weight.data[:, self.start : self.low_end] * self.mixer_vec.data
                    + self.weight.data[:, self.low_end : self.end] @ monarch_weight.t()
                )
            
        self.fused = True

    def unfuse_s2_weight(self):
        if self.fused == False:
            return
        self.weight[:, self.start : self.tuned_end] -= self.s2
        self.fused = False

    def forward(self, input: Tensor) -> Tensor:            
        if self.fused:
            base_output = torch.nn.functional.linear(input, self.weight, self.bias)
            return base_output
        else:
            if not self.mix_channels:
                tuned_weight = self.weight[:, self.start : self.tuned_end] + self.s2
                mixed_weight = torch.cat([tuned_weight[:, self.start : self.low_end] * self.mixer_vec,self.mixer(tuned_weight[:, self.low_end : self.end])+tuned_weight[:, self.low_end : self.end]],dim=1)
                # Build the full weight matrix by concatenation to avoid cloning a large matrix
                full_weight = torch.cat([
                    self.weight[:, :self.start],       # Original leading weights
                    mixed_weight,
                    tuned_weight[:, self.end :self.tuned_end],  # Remaining tuned_part
                    self.weight[:, self.tuned_end:]     # Original trailing weights
                ], dim=1)
                
                # Apply one linear operation, including bias
                return torch.nn.functional.linear(input, full_weight, self.bias)
            else:
                tuned_weight = self.weight[:, self.start : self.tuned_end] + self.s2
                # Compute the dynamic mixture without modifying the weights
                mixed_weight_low = (
                    tuned_weight[:, self.start : self.low_end] * self.mixer_vec
                    + self.mixer(tuned_weight[:, self.low_end : self.end])
                )
                
                # Build the full weight matrix by concatenation to avoid cloning a large matrix
                full_weight = torch.cat([
                    self.weight[:, :self.start],       # Original leading weights
                    mixed_weight_low,
                    tuned_weight[:, self.low_end :self.tuned_end],  # Remaining tuned_part
                    self.weight[:, self.tuned_end:]     # Original trailing weights
                ], dim=1)
                
                # Apply one linear operation, including bias
                return torch.nn.functional.linear(input, full_weight, self.bias)
            # # Compute the mixed output segment
            # mixed_output = torch.nn.functional.linear(
            #     input[..., self.start : self.low_end], 
            #     mixed_weight_low, 
            #     None
            # )
            # # Compute the mixed output segment
            # high_output = torch.nn.functional.linear(
            #     input[..., self.low_end : self.tuned_end], 
            #     tuned_weight[:, self.low_end :], 
            #     None
            # )
            # # Compute the mixed output segment
            # other_output = torch.nn.functional.linear(
            #     input[..., self.tuned_end : ], 
            #     self.weight[:, self.tuned_end :], 
            #     None
            # )
            # base_output = mixed_output+high_output+other_output
            # s2_output = torch.nn.functional.linear(
            #     input[:, :, self.start : self.tuned_end], self.s2, None
            # )
            # base_output += s2_output
            # return base_output

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

# class MixColumnLinear(Module):

#     __constants__ = ["in_features", "out_features"]
#     in_features: int
#     out_features: int
#     weight: Tensor

#     def __init__(
#         self,
#         in_features: int,
#         out_features: int,
#         bias: bool = True,
#         start=None,
#         end=None,
#         low_end=None,
#         device=None,
#         dtype=None,
#     ) -> None:
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(
#             torch.empty((out_features, in_features), **factory_kwargs),
#             requires_grad=True,
#         )
#         if bias:
#             self.bias = Parameter(
#                 torch.empty(out_features, **factory_kwargs), requires_grad=True
#             )
#         else:
#             self.register_parameter("bias", None)
#         if (end-start)%2!=0:
#             end = end+1
#         self.start = start
#         if low_end is None:
#             low_end = (end-start) // 2
#         self.low_end = low_end
#         self.end = end
#         self.s2 = nn.Parameter(
#             torch.zeros(end - start, in_features), requires_grad=True
#         )
#         self.mixer_vec = nn.Parameter(torch.ones(1, low_end - start), requires_grad=True)
#         self.mixer = nn.Linear(end - low_end, low_end - start, bias=False)
#         self.reset_parameters()
#         self.mixer.weight.requires_grad = True
#         self.weight.requires_grad = False
#         self.fused = False

#     def reset_parameters(self) -> None:
#         # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
#         # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
#         # https://github.com/pytorch/pytorch/issues/57109
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#             init.uniform_(self.bias, -bound, bound)
#         init.kaiming_uniform_(self.mixer.weight, a=math.sqrt(5))*0.00001
        
#         # for i in range((self.end - self.start)//2):
#         #     weight[i, i] = 1
#         # with torch.no_grad():
#         #     self.mixer.weight.copy_(weight)
        

#     def fuse_s2_weight(self):
#         if self.fused == True:
#             return
#         self.weight.data[self.start : self.end, :] += self.s2
#         with torch.no_grad():
#             mid = self.low_end
#             self.weight[self.start : mid, :] = self.weight[self.start : mid, :].mul(self.mixer_vec.T) + self.mixer.weight @ self.weight[mid:self.end, :] 
#         self.fused = True

#     def unfuse_s2_weight(self):
#         if self.fused == False:
#             return
#         self.weight[self.start : self.end, :] -= self.s2
#         self.fused = False

#     def forward(self, input: Tensor) -> Tensor:
#         base_output = torch.nn.functional.linear(input, self.weight, self.bias)
#         if self.fused:
#             return base_output
#         else:
#             s2_output = torch.nn.functional.linear(input, self.s2, None)
#             mid = self.low_end

#             # Extract the middle segment and add s2_output
#             modified_channel = base_output[..., self.start:self.end] + s2_output
            
#             # Split the middle segment into low and high halves, retaining the updated high half
#             modified_high_channel = modified_channel[..., mid:]
#             modified_low_channel = modified_channel[..., :mid]

#             # Use the mixer to produce the new low half
#             mixed_part = self.mixer(modified_high_channel) + modified_low_channel * self.mixer_vec
            
#             # Construct the new middle segment from the mixer result and high half
#             new_middle = torch.cat([mixed_part, modified_high_channel], dim=-1)
            
#             # Concatenate all segments
#             output = torch.cat([
#                 base_output[..., :self.start],
#                 new_middle,
#                 base_output[..., self.end:]
#             ], dim=-1)
#             # base_output[:, :, self.start : self.end] += s2_output
#             # # Use the mixer output without in-place assignment
#             # mixer_output = self.mixer(base_output[:, :, self.start : self.end])
#             # base_output[:, :, self.start : (self.end+1)//2] = mixer_output
#             #base_output[:, :, self.start : (self.end+1)//2] = self.mixer(base_output[:, :, self.start : self.end])
#             return output

#     def extra_repr(self) -> str:
#         return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


# class MixRowLinear(Module):

#     __constants__ = ["in_features", "out_features"]
#     in_features: int
#     out_features: int
#     weight: Tensor

#     def __init__(
#         self,
#         in_features: int,
#         out_features: int,
#         bias: bool = True,
#         start=None,
#         end=None,
#         device=None,
#         dtype=None,
#     ) -> None:
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(
#             torch.empty((out_features, in_features), **factory_kwargs),
#             requires_grad=True,
#         )
#         if bias:
#             self.bias = Parameter(
#                 torch.empty(out_features, **factory_kwargs), requires_grad=True
#             )
#         else:
#             self.register_parameter("bias", None)
#         self.reset_parameters()
#         self.start = start
#         self.end = end

#         self.s2 = nn.Parameter(
#             torch.zeros(out_features, end - start), requires_grad=True
#         )
#         self.weight.requires_grad = False
#         self.fused = False

#     def reset_parameters(self) -> None:
#         # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
#         # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
#         # https://github.com/pytorch/pytorch/issues/57109
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#             init.uniform_(self.bias, -bound, bound)

#     def fuse_s2_weight(self):
#         if self.fused == True:
#             return
#         self.weight.data[:, self.start : self.end] += self.s2
#         self.fused = True

#     def unfuse_s2_weight(self):
#         if self.fused == False:
#             return
#         self.weight[:, self.start : self.end] -= self.s2
#         self.fused = False

#     def forward(self, input: Tensor) -> Tensor:
#         base_output = torch.nn.functional.linear(input, self.weight, self.bias)
#         if self.fused:
#             return base_output
#         else:
#             s2_output = torch.nn.functional.linear(
#                 input[:, :, self.start : self.end], self.s2, None
#             )
#             base_output += s2_output
#             return base_output

#     def extra_repr(self) -> str:
#         return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
