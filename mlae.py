import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List
import random
from transformers import set_seed

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: float, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


# LoRA 
class Linear(nn.Linear, LoRALayer):

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        self.lora_features = []

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
    
    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                out= (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
                result += out
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
        
        
# MLAE
class MLAE_Linear(nn.Linear, LoRALayer):
    def __init__(self, in_features: int, out_features: int, r: int = 0,
                 lora_alpha: float = 1.0, lora_dropout: float = 0.,
                 drop_rate: float = 0.0, fan_in_fan_out: bool = False,
                 merge_weights: bool = True, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha,
                           lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.drop_rate = drop_rate
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros(r, in_features))
            self.lora_B = nn.Parameter(self.weight.new_zeros(r, out_features))

            #Adaptive cofficient
            self.lora_lamda = nn.Parameter(torch.full((r,), lora_alpha, dtype=torch.float), requires_grad=True)

            self.weight.requires_grad = False  
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            if self.r > 0:
                delta_weight = torch.einsum('ro,ri,r->oi', self.lora_B, self.lora_A, self.lora_lamda)
                self.weight.data -= T(delta_weight)
            self.merged = False

    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            if self.r > 0:
                delta_weight = torch.einsum('ro,ri,r->oi', self.lora_B, self.lora_A, self.lora_lamda)
                self.weight.data += T(delta_weight)
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w      
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)

            # expert-level masking
            drop_lamda = F.dropout(self.lora_lamda, p=self.drop_rate, training=self.training)

            lora_activation = self.lora_dropout(x) @ self.lora_A.T  # (batch_size, r)
             # apply masking to LoRA
            lora_activation = lora_activation * drop_lamda  
            lora_output = lora_activation @ self.lora_B  # (batch_size, out_features)
            result += lora_output
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

