#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Ravi Krishna 07/25/21

# Import statements.
import torch
import torch.nn as nn
import numpy as np

#try:
#    import intel_pytorch_extension as ipex
#except:
#    pass
#from lamb_bin import Lamb, log_lamb_rs


class LinearDLRM(nn.Module):
    """
    This layer is essentially equivalent
    to a regular nn.Linear layer, except
    that it uses the weight initialization
    from dlrm_s_pytorch.py.
    """

    def __init__(self,
                in_feat,
                out_feat,
                bias=False,
                output_stays_blocked=False, 
                use_bf16=False):

        # Superclass initialization.
        super(LinearDLRM, self).__init__()

        # Store for later.
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias

        # Create the layer.
        if use_bf16:
            self.linear_layer = ipex.IpexMLPLinear(
                 int(self.in_feat),
                 int(self.out_feat),
                 output_stays_blocked=output_stays_blocked, default_blocking=32)
        else:
            self.linear_layer = nn.Linear(self.in_feat,
                                            self.out_feat,
                                            bias=self.bias)

        # Change initialization to match DLRM code
        # from dlrm_s_pytorch.py.
        with torch.no_grad():
            mean = 0.0

            std_dev = np.sqrt(2.0 / (float(self.in_feat) + float(self.out_feat)))

            W = np.random.normal(mean,
                                std_dev,
                                size=(self.out_feat, self.in_feat)).astype(np.float32)

            std_dev = np.sqrt(1.0 / float(self.out_feat))

            bt = np.random.normal(mean,
                                std_dev,
                                size=self.out_feat).astype(np.float32)

            self.linear_layer.weight.data = torch.tensor(W, requires_grad=True)

            if self.bias is True:
                self.linear_layer.bias.data = torch.tensor(bt, requires_grad=True)

        if use_bf16:
            self.linear_layer.to(torch.bfloat16)
            # prepack weight for IPEX Linear
            if hasattr(self.linear_layer, 'reset_weight_shape'):
                self.linear_layer.reset_weight_shape(block_for_dtype=torch.bfloat16)

    def forward(self, input_vectors):
        """
        Run input_vectors through the layer.
        """

        return self.linear_layer(input_vectors)
