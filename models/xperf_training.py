# Copyright (2023) Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from torch import nn, autograd
import logging
from einops import rearrange, repeat


class IndexFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        # return input[indices]
        return torch.gather(rearrange(input, 'b ... -> b (...)'), 0,
                            repeat(indices, 'z -> z d', d=second_dim)).reshape(-1, *other_shape)
    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, 'b ... -> b (...)')
        grad_input = torch.zeros([ctx.first_axis_dim, grad_output.shape[1]],
                                device=grad_output.device, dtype=grad_output.dtype)
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # grad_input[indices] = grad_output
        grad_input.scatter_(0, repeat(indices, 'z -> z d', d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device,
                            dtype=values.dtype)
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        output[indices] = values
        # output.scatter_(0, repeat(indices, 'z -> z d', d=values.shape[1]), values)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        grad_values = grad_output[indices]
        # grad_values = torch.gather(grad_output, 0, repeat(indices, 'z -> z d', d=grad_output.shape[1]))
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply


class IndexFirstAxisResidual(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        output = input[indices]
        # We don't want to reshape input (b ... -> b (...)) since it could change the channel_last
        # memory format to channel_first. In other words, input might not be contiguous.
        # If we don't detach, Pytorch complains about output being a view and is being modified inplace
        return output, input.detach()

    @staticmethod
    def backward(ctx, grad_output, grad_residual):
        indices, = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        assert grad_residual.shape[1:] == other_shape
        grad_input = grad_residual
        # grad_input[indices] += grad_output
        indices = indices.reshape(indices.shape[0], *((1,) * (grad_output.ndim - 1)))
        indices = indices.expand_as(grad_output)
        grad_input.scatter_add_(0, indices, grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis_residual = IndexFirstAxisResidual.apply


def unpad_input(hidden_states, attention_mask):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    return (index_first_axis(rearrange(hidden_states, 'b s ... -> (b s) ...'), indices), indices,
            cu_seqlens, max_seqlen_in_batch)


def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz)
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    dim = hidden_states.shape[-1]
    # output = torch.zeros((batch * seqlen), dim, device=hidden_states.device, dtype=hidden_states.dtype)
    # output[indices] = hidden_states
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, '(b s) ... -> b s ...', b=batch)


try:
    # import xperf_training
    # fast_transformer_dir = xperf_training.__path__[0]
    # fast_transformer_lib = os.path.join(fast_transformer_dir, 'libxperf_training_torch_ops_dyn.so')
    # torch.ops.load_library(fast_transformer_lib)
    import lego_ops
    lego_ops.load_ft_torch()

    class LayerNormFunction(autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, gamma, beta, residual=None, eps=1e-6):
            output, mean_var_rsqrt = torch.ops.FasterTransformer.LayerNorm_forward(
                input_tensor, gamma, beta, residual, eps)
            ctx.save_for_backward(input_tensor, gamma, mean_var_rsqrt, residual)
            return output

        @staticmethod
        def backward(ctx, grad_out):
            grad_in, grad_gamma, grad_beta, grad_residual = torch.ops.FasterTransformer.LayerNorm_backward(
                grad_out, *ctx.saved_tensors)
            return grad_in, grad_gamma, grad_beta, grad_residual, None

    class FasterLayerNorm(nn.Module):
        def __init__(self, hidden_dim, eps=1e-6):
            super(FasterLayerNorm, self).__init__()

            self.hidden_dim = hidden_dim
            self.weight = nn.Parameter(torch.Tensor(hidden_dim))
            self.bias = nn.Parameter(torch.Tensor(hidden_dim))
            self.eps = eps
            nn.init.constant_(self.weight, 1.0)
            nn.init.constant_(self.bias, 0.0)

        def forward(self, input_tensor, residual=None):
            if self.training:
                return LayerNormFunction.apply(input_tensor, self.weight, self.bias, residual, self.eps)
            else:
                output, mean_var_rsqrt = torch.ops.FasterTransformer.LayerNorm_forward(
                    input_tensor, self.weight, self.bias, residual, self.eps)
                return output

        def extra_repr(self):
            return 'hidden_dim={}'.format(self.hidden_dim)

    FTLayerNorm = FasterLayerNorm

    class LinearFunction(autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, residual, weight, bias, act_gelu = False, dropout_rate = 0.0):
            bias_out = torch.Tensor(0)
            dropout_mask = torch.Tensor(0)
            if act_gelu == True or dropout_rate > 0.0:
                output, bias_out, dropout_mask = torch.ops.FasterTransformer.Linear_forward_gelu_dropout(input_tensor, weight, bias, act_gelu, dropout_rate)
            else:
                output = torch.ops.FasterTransformer.Linear_forward(input_tensor, weight, bias, residual)
            ctx.save_for_backward(input_tensor, weight, bias_out, dropout_mask)
            ctx.act_gelu = act_gelu
            ctx.dropout_rate = dropout_rate
            ctx.has_residual = residual is not None
            return output

        @staticmethod
        def backward(ctx, grad_out):
            input_tensor, weight, bias_out, dropout_mask = ctx.saved_tensors
            if ctx.act_gelu == True or ctx.dropout_rate > 0.0:
                grad_in, grad_weight, grad_bias = torch.ops.FasterTransformer.Linear_backward_gelu_dropout(
                    grad_out, input_tensor, weight, ctx.act_gelu, ctx.dropout_rate, bias_out, dropout_mask)
            else:
                grad_in, grad_weight, grad_bias = torch.ops.FasterTransformer.Linear_backward(
                    grad_out, input_tensor, weight)
            return grad_in, grad_out.detach().clone() if ctx.has_residual else None, grad_weight, grad_bias, None, None

    class FasterLinear(nn.Module):
        def __init__(self, in_features, out_features, initializer_range=0.02, act_gelu=False, dropout_rate=0.0):
            super(FasterLinear, self).__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.weight.data.normal_(mean=0.0, std=initializer_range)
            self.bias = nn.Parameter(torch.Tensor(out_features))
            self.bias.data.zero_()
            self.act_gelu = act_gelu
            self.dropout_rate = dropout_rate

        def forward(self, input_tensor, residual=None):
            if self.training:
                return LinearFunction.apply(input_tensor, residual, self.weight, self.bias, self.act_gelu, self.dropout_rate)
            else:
                if self.act_gelu:
                    output, bias_out, dropout_mask = torch.ops.FasterTransformer.Linear_forward_gelu_dropout(input_tensor, self.weight, self.bias, self.act_gelu, 0.0)
                else:
                    output = torch.ops.FasterTransformer.Linear_forward(input_tensor, self.weight, self.bias, residual)
                return output

        def extra_repr(self):
            return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

    FTLinear = FasterLinear

    class FasterLinearWeightTransposed(nn.Module):
        def __init__(self, in_features, out_features, initializer_range=0.02, act_gelu=False, dropout_rate=0.0):
            super(FasterLinearWeightTransposed, self).__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features))  # the weight is transposed
            self.weight.data.normal_(mean=0.0, std=initializer_range)
            self.bias = nn.Parameter(torch.Tensor(out_features))
            self.bias.data.zero_()
            self.act_gelu = act_gelu
            self.dropout_rate = dropout_rate

        def forward(self, input_tensor, residual=None):
            weight_normal = self.weight.transpose(1, 0).contiguous()
            if self.training:
                return LinearFunction.apply(input_tensor, residual, weight_normal, self.bias, self.act_gelu, self.dropout_rate)
            else:
                if self.act_gelu:
                    output, bias_out, dropout_mask = torch.ops.FasterTransformer.Linear_forward_gelu_dropout(input_tensor, weight_normal, self.bias, self.act_gelu, 0.0)
                else:
                    output = torch.ops.FasterTransformer.Linear_forward(input_tensor, weight_normal, self.bias, residual)
                return output

        def extra_repr(self):
            return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

    FTLinearWeightTransposed = FasterLinearWeightTransposed


    class LinearTransposeFunction(autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, weight, bias, head_num, transpose_type):
            output = torch.ops.FasterTransformer.LinearTranspose_forward(input_tensor, weight, bias, head_num, transpose_type)
            ctx.head_num = head_num
            ctx.transpose_type = transpose_type
            ctx.save_for_backward(input_tensor, weight)
            return output

        @staticmethod
        def backward(ctx, grad_out):
            input_tensor, weight = ctx.saved_tensors
            grad_in, grad_weight, grad_bias = torch.ops.FasterTransformer.LinearTranspose_backward(grad_out, input_tensor, weight, ctx.head_num, ctx.transpose_type)
            return grad_in, grad_weight, grad_bias, None, None

    class FasterLinearTranspose(nn.Module):
        def __init__(self, in_features, out_features, head_num, transpose_type="0213", initializer_range=0.02):
            super(FasterLinearTranspose, self).__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.head_num = head_num
            self.transpose_type = transpose_type
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.bias = nn.Parameter(torch.Tensor(out_features))
            self.weight.data.normal_(mean=0.0, std=initializer_range)
            self.bias.data.zero_()

        def forward(self, input_tensor):
            return LinearTransposeFunction.apply(input_tensor, self.weight, self.bias, self.head_num, self.transpose_type)

        def extra_repr(self):
            return 'in_features={}, out_features={}, head_num={}'.format(self.in_features, self.out_features, self.head_num)

    FTLinearTranspose = FasterLinearTranspose

    class LinearSplitTransposeFunction(autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, weight, bias, head_num, transpose_type):
            q_output, k_output, v_output = torch.ops.FasterTransformer.LinearSplitTranspose_forward(input_tensor, weight, bias, head_num, transpose_type)
            ctx.head_num = head_num
            ctx.transpose_type = transpose_type
            ctx.save_for_backward(input_tensor, weight)
            return q_output, k_output, v_output

        @staticmethod
        def backward(ctx, q_grad_out, k_grad_out, v_grad_out):
            input_tensor, weight = ctx.saved_tensors
            grad_in, grad_weight, grad_bias = torch.ops.FasterTransformer.LinearSplitTranspose_backward(q_grad_out, k_grad_out, v_grad_out, input_tensor, weight, ctx.head_num, ctx.transpose_type)
            return grad_in, grad_weight, grad_bias, None, None

    class FasterLinearSplitTranspose(nn.Module):
        def __init__(self, in_features, out_features, head_num, transpose_type = "0213"):
            super(FasterLinearSplitTranspose, self).__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.head_num = head_num
            self.transpose_type = transpose_type
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.bias   = nn.Parameter(torch.Tensor(out_features))
            torch.nn.init.normal_(self.weight, mean=0, std=1)
            torch.nn.init.normal_(self.bias,   mean=0, std=1)

        def forward(self, input_tensor):
            return LinearSplitTransposeFunction.apply(input_tensor, self.weight, self.bias, self.head_num, self.transpose_type)

        def extra_repr(self):
            return 'in_features={}, out_features={}, head_num={}'.format(self.in_features, self.out_features, self.head_num)

    class TorchGatherFunction(autograd.Function):
        @staticmethod
        def forward(ctx, c2p_tensor, p2c_tensor, score_tensor, score_scaler):
            output = torch.ops.FasterTransformer.TorchGather_forward(c2p_tensor, p2c_tensor, score_tensor, score_scaler)
            ctx.score_scaler = score_scaler
            return output[0]

        @staticmethod
        def backward(ctx, grad_out):
            c2p_tensor_grad, p2c_tensor_grad, score_tensor_grad = torch.ops.FasterTransformer.TorchGather_backward(grad_out, ctx.score_scaler)
            return c2p_tensor_grad, p2c_tensor_grad, score_tensor_grad, None

    class FasterTorchGather(nn.Module):
        def __init__(self, score_scaler):
            super(FasterTorchGather, self).__init__()
            self.score_scaler = score_scaler

        def forward(self, c2p_tensor, p2c_tensor, score_tensor):
            return TorchGatherFunction.apply(c2p_tensor, p2c_tensor, score_tensor, self.score_scaler)

        def extra_repr(self):
            return 'score_scaler={}'.format(self.score_scaler)

    class FTDAGather(nn.Module):
        def __init__(self, score_scaler):
            super().__init__()

            self.score_scaler = score_scaler

        def forward(self, c2p_tensor, p2c_tensor, score_tensor):
            return TorchGatherFunction.apply(c2p_tensor, p2c_tensor, score_tensor, self.score_scaler)

        def extra_repr(self):
            return 'score_scaler={}'.format(self.score_scaler)

    class SoftmaxFunction(autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, mask_tensor, head_num = 1, dropout_rate = 0.0, batch_first = True):
            mask_tensor = mask_tensor.to(input_tensor.dtype)
            softmax_out, softmax_dropout_out, dropout_mask = torch.ops.FasterTransformer.Softmax_forward(
                input_tensor, mask_tensor, head_num, dropout_rate, batch_first)
            ctx.save_for_backward(softmax_out, dropout_mask)
            ctx.dropout_rate = dropout_rate
            return softmax_dropout_out if dropout_rate != 0.0 else softmax_out

        @staticmethod
        def backward(ctx, grad_out):
            softmax_out, dropout_mask = ctx.saved_tensors
            grad_in = torch.ops.FasterTransformer.Softmax_backward(
                grad_out, softmax_out, dropout_mask, ctx.dropout_rate)
            return grad_in, None, None, None, None

    def faster_softmax(input_tensor, mask_tensor = None, head_num = 1, dropout_rate = 0.0, batch_first = True):
        if torch.jit.is_tracing():
            return torch.ops.FasterTransformer.Softmax_infer(input_tensor, mask_tensor, head_num, dropout_rate, batch_first)
        else:
            return SoftmaxFunction.apply(input_tensor, mask_tensor, head_num, dropout_rate, batch_first)

    class FTSoftmax(nn.Module):
        def forward(self, input_tensor, mask_tensor, head_num, dropout_rate, batch_first):
            return SoftmaxFunction.apply(input_tensor, mask_tensor, head_num, dropout_rate if self.training else 0, batch_first)

    class TransposeFunction(autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, transpose_type):
            transpose_out = torch.ops.FasterTransformer.Transpose4d_forward(input_tensor, transpose_type)
            ctx.transpose_type = transpose_type
            return transpose_out

        @staticmethod
        def backward(ctx, grad_out):
            grad_in = torch.ops.FasterTransformer.Transpose4d_backward(grad_out, ctx.transpose_type)
            return grad_in, None


    def faster_transpose(input_tensor, transpose_type = "0213"):
        return TransposeFunction.apply(input_tensor, transpose_type)

    def FTTransposeV1(transpose_type="0213"):
        default_transpose_type = transpose_type

        def faster_transpose(input_tensor, transpose_type = default_transpose_type):
            return TransposeFunction.apply(input_tensor, transpose_type)

        return faster_transpose


    class MatMulFunction(autograd.Function):
        @staticmethod
        def forward(ctx, input_A, input_B, transpose_a = False, transpose_b = False, scale = 1.0):
            matmul_out = torch.ops.FasterTransformer.MatMul_forward(
                input_A, input_B, transpose_a, transpose_b, scale)
            ctx.transpose_a = transpose_a
            ctx.transpose_b = transpose_b
            ctx.scale = scale
            ctx.save_for_backward(input_A, input_B)
            return matmul_out

        @staticmethod
        def backward(ctx, grad_out):
            input_A, input_B = ctx.saved_tensors
            grad_A, grad_B = torch.ops.FasterTransformer.MatMul_backward(
                grad_out, input_A, input_B, ctx.transpose_a, ctx.transpose_b, ctx.scale)
            return grad_A, grad_B, None, None, None

    def faster_matmul(input_A, input_B, transpose_a = False, transpose_b = False, scale = 1.0):
        input_B = input_B.to(input_A.dtype)
        if torch.jit.is_tracing():
            return torch.ops.FasterTransformer.MatMul_forward(input_A, input_B, transpose_a, transpose_b, scale)
        else:
            return MatMulFunction.apply(input_A, input_B, transpose_a, transpose_b, scale)

    FTMatMul = lambda: faster_matmul


    class RotaryEmbeddingFunction(autograd.Function):
        @staticmethod
        def forward(ctx, input_Q, input_K):
            output_Q, output_K = RotaryEmbedding.forward(input_Q, input_K)
            return output_Q, output_K

        @staticmethod
        def backward(ctx, grad_out_Q, grad_out_K):
            grad_Q, grad_K = RotaryEmbedding.backward(grad_out_Q, grad_out_K)
            return grad_Q, grad_K

    def faster_rotary_embedding(input_Q, input_K):
        return RotaryEmbeddingFunction.apply(input_Q, input_K)


    def faster_attention_infer(input_list, mask=None, head_num=1):
        input_len = len(input_list)
        if input_len == 3:  #input_q, input_k, input_v
            return torch.ops.FasterTransformer.FuseAttention_infer(*input_list, mask, head_num)
        elif input_len == 2:#input_q, input_kv
            return torch.ops.FasterTransformer.FuseAttention_infer_q_kv(*input_list, mask, head_num)
        elif input_len == 1:#input_qkv
            return torch.ops.FasterTransformer.FuseAttention_infer_qkv(*input_list, mask, head_num)
        else:
            print("Wrong input list")


    class GatedLinearUnitFunction(autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, weight, bias, dropout_rate = 0.0):
            bias_out = torch.Tensor(0)
            dropout_mask = torch.Tensor(0)
            output, bias_out, dropout_mask = torch.ops.FasterTransformer.GatedLinearUnit_forward(input_tensor, weight, bias, dropout_rate)
            ctx.save_for_backward(input_tensor, weight, bias_out, dropout_mask)
            ctx.dropout_rate = dropout_rate
            return output

        @staticmethod
        def backward(ctx, grad_out):
            input_tensor, weight, bias_out, dropout_mask = ctx.saved_tensors
            grad_in, grad_weight, grad_bias = torch.ops.FasterTransformer.GatedLinearUnit_backward(
                grad_out, input_tensor, weight, ctx.act_gelu, ctx.dropout_rate, bias_out, dropout_mask)
            return grad_in, grad_weight, grad_bias, None


    class FasterGatedLinearUnit(nn.Module):
        def __init__(self, in_features, out_features, dropout_rate = 0.0):
            super(FasterGatedLinearUnit, self).__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.bias   = nn.Parameter(torch.Tensor(out_features))
            torch.nn.init.normal_(self.weight, mean=0, std=1)
            torch.nn.init.normal_(self.bias,   mean=0, std=1)
            self.dropout_rate = dropout_rate

        def forward(self, input_tensor):
            if self.training:
                return GatedLinearUnitFunction.apply(input_tensor, self.weight, self.bias, self.dropout_rate)
            else:
                output, bias_out, dropout_mask = torch.ops.FasterTransformer.GatedLinearUnit_forward(input_tensor, self.weight, self.bias, self.dropout_rate)
                return output

        def extra_repr(self):
            return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

    class FuseAttentionFunction(autograd.Function):
        @staticmethod
        def forward(
            ctx, input_Q, input_K, input_V, softmax_mask, head_num, dropout_rate = 0.0):

            attention_out, softmax_out, dropout_mask, softmax_dropout_out = torch.ops.FasterTransformer.FuseAttention_forward(
                input_Q, input_K, input_V, softmax_mask, head_num, dropout_rate)

            ctx.save_for_backward(
                input_Q, input_K, input_V, softmax_out, dropout_mask, softmax_dropout_out)
            ctx.head_num = head_num
            ctx.dropout_rate = dropout_rate
            return attention_out

        @staticmethod
        def backward(ctx, grad_out):
            input_Q, input_K, input_V, softmax_out, dropout_mask, softmax_dropout_out = ctx.saved_tensors
            grad_Q, grad_K, grad_V = torch.ops.FasterTransformer.FuseAttention_backward(
                grad_out, softmax_out, input_Q, input_K, input_V, ctx.head_num, ctx.dropout_rate, dropout_mask, softmax_dropout_out)

            return grad_Q, grad_K, grad_V, None, None, None

    class FasterFuseAttention(nn.Module):
        def __init__(self, head_num, dropout_rate = 0.0):
            super(FasterFuseAttention, self).__init__()
            self.head_num = head_num
            self.dropout_rate = dropout_rate

        def forward(self, input_Q, input_K, input_V, softmax_mask):
            return FuseAttentionFunction.apply(
                input_Q, input_K, input_V, softmax_mask, self.head_num, self.dropout_rate)

    FTFusedAttention = FasterFuseAttention


    def _flash_attn_forward(q, k, v, out, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                            dropout_p, attn_mask, attn_bias, softmax_scale, causal, return_softmax, num_splits=0,
                            generator=None):
        """
        num_splits: how much to parallelize over the seqlen_q dimension. num_splits=0 means
        it will be set by an internal heuristic. We're exposing num_splits mostly for benchmarking.
        Don't change it unless you know what you're doing.
        """

        softmax_lse, *rest = torch.ops.Extern.FlashAttn_forward(
            q, k, v, out, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
            softmax_scale, False, causal, return_softmax, num_splits, generator, attn_mask, attn_bias
        )
        # if out.isnan().any() or softmax_lse.isnan().any():
        #     breakpoint()
        S_dmask = rest[0] if return_softmax else None
        return out, softmax_lse, S_dmask

    def _flash_attn_backward(dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k,
                             max_seqlen_q, max_seqlen_k, dropout_p, attn_mask, attn_bias, softmax_scale, causal, num_splits=0,
                             generator=None):
        """
        num_splits: how much to parallelize over the seqlen_q dimension. num_splits=0 means
        it will be set by an internal heuristic. Setting this too large (e.g. > 10) could make
        numerical error of dK and dV larger (scaling as sqrt(num_splits)).
        This hyperparameter can be tuned for performance, but default value (heuristic) should work fine.
        """
        softmax_d, *rest = torch.ops.Extern.FlashAttn_backward(
            dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, False, causal, num_splits, generator, attn_mask, attn_bias)
        # if dk.isnan().any() or dk.isnan().any() or dv.isnan().any() or softmax_d.isnan().any():
        #     breakpoint()
        dbias = None if attn_bias is None else rest[0]
        return dq, dk, dv, softmax_d, dbias

    class FlashAttnFunc(torch.autograd.Function):

        @staticmethod
        def forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
                    attn_mask, attn_bias, softmax_scale, causal, return_softmax):
            # Save rng_state because the backward pass will regenerate the dropout mask
            rng_state = torch.cuda.get_rng_state() if dropout_p > 0 else None
            if softmax_scale is None:
                softmax_scale = q.shape[-1] ** (-0.5)
            out, softmax_lse, S_dmask = _flash_attn_forward(
                q, k, v, torch.empty_like(q), cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                dropout_p, attn_mask, attn_bias, softmax_scale, causal=causal, return_softmax=return_softmax
            )
            ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state, attn_mask, attn_bias)
            ctx.dropout_p = dropout_p
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            return out if not return_softmax else (out, softmax_lse, S_dmask)

        @staticmethod
        def backward(ctx, dout, *args):
            q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state, attn_mask, attn_bias= ctx.saved_tensors
            if rng_state is not None:
                cur_rng_state = torch.cuda.get_rng_state()
                torch.cuda.set_rng_state(rng_state)
            dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
            _, _, _, softmax_d, dbias = _flash_attn_backward(
                dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k,
                ctx.max_seqlen_q, ctx.max_seqlen_k, ctx.dropout_p, attn_mask, attn_bias, ctx.softmax_scale, ctx.causal
            )
            if rng_state is not None:
                torch.cuda.set_rng_state(cur_rng_state)
            return dq, dk, dv, None, None, None, None, None, None, dbias, None, None, None

    def flash_attn_unpadded_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                 dropout_p, attn_mask = None, attn_bias = None, softmax_scale=None, causal=False, return_attn_probs=False):
        if torch.jit.is_tracing():
            if softmax_scale is None:
                softmax_scale = q.shape[-1] ** (-0.5)
            out, softmax_lse, S_dmask = _flash_attn_forward(
                q, k, v, torch.empty_like(q), cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                dropout_p, attn_mask, attn_bias, softmax_scale, causal=causal, return_softmax=return_attn_probs)
            return out
        else:
            return FlashAttnFunc.apply(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                    dropout_p, attn_mask, attn_bias, softmax_scale, causal, return_attn_probs)

    def get_seq_len(attention_mask):
        return torch.ops.FasterTransformer.RemovePadding_get_seq_len(attention_mask)

    def faster_flash_attention(input_list, head_num, attn_mask = None, attn_bias = None, causal = False,
        cu_seqlens_q = None, cu_seqlens_k = None, max_seqlens_q = None, max_seqlens_k = None,
        softmax_scale = None, attention_dropout = 0.0, word_idx=None, use_rmpad_attn=False):
        input_count = len(input_list)
        assert input_count == 3
        if input_count == 3:
            q, k, v = input_list

            if not use_rmpad_attn:
                batch, seqlen, _ = q.shape

            if causal == False:
                if cu_seqlens_q is None:
                    max_seqlens_q = q.shape[1]
                    max_seqlens_k = k.shape[1]
            else:
                # assert attn_mask.shape[1] == 1
                if use_rmpad_attn:
                    max_seqlens_q = torch.max(cu_seqlens_q[1:] - cu_seqlens_q[:-1])
                    max_seqlens_k = torch.max(cu_seqlens_k[1:] - cu_seqlens_k[:-1])
                else:
                    max_seqlens_q = q.shape[1]
                    max_seqlens_k = k.shape[1]
                # attn_mask = attn_mask.type(q.dtype)
            q = q.view(-1, head_num, q.shape[-1] // head_num)
            k = k.view(-1, head_num, k.shape[-1] // head_num)
            v = v.view(-1, head_num, v.shape[-1] // head_num)
            dtype = q.dtype
            if dtype == torch.float32:
                q = q.to(dtype=torch.bfloat16)
                k = k.to(dtype=torch.bfloat16)
                v = v.to(dtype=torch.bfloat16)
                if attn_mask is not None:
                    attn_mask = attn_mask.to(dtype=torch.bfloat16)
            
            if causal == False:
                output = flash_attn_unpadded_func(q, k, v,
                            cu_seqlens_q, cu_seqlens_k, max_seqlens_q, max_seqlens_k,
                            attention_dropout, attn_mask, attn_bias, softmax_scale, causal, False)
            else:
                if use_rmpad_attn:
                    output = flash_attn_unpadded_func(q, k, v,
                        cu_seqlens_q, cu_seqlens_k , max_seqlens_q, max_seqlens_k,
                        attention_dropout, None, None, None, causal, False)
                else:
                    output = flash_attn_unpadded_func(q, k, v,
                        None, None, max_seqlens_q, max_seqlens_k,
                        attention_dropout, None, None, None, causal, False)
            if dtype == torch.float32:
                output = output.to(dtype)
        if use_rmpad_attn:
            output = output.contiguous().view(output.shape[0], output.shape[1]*output.shape[2])
        else:
            output = output.view(batch, seqlen, -1)
        return output

    FTFlashAttention = lambda: faster_flash_attention

except ImportError:
    logging.warning("Unable to import xperf_training, skip FT ops...")
    FTLinear = None
    FTLinearWeightTransposed = None
    FTTranspose = None
    FTTransposeV1 = None
    FTMatMul = None
    FTLinearTranspose = None
    FTDAGather = None
    FTSoftmax = None
    FTLayerNorm = None
    FTFusedAttention = None
    FTFlashAttention = None

except OSError:
    logging.warning("Import xperf_training failed, perhaps upgrade xperf version.")
    FTLinear = None
    FTLinearWeightTransposed = None
    FTTranspose = None
    FTTransposeV1 = None
    FTMatMul = None
    FTLinearTranspose = None
    FTDAGather = None
    FTSoftmax = None
    FTLayerNorm = None
    FTFusedAttention = None
    FTFlashAttention = None
