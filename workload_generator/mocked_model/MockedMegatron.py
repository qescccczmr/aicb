"""
Copyright (c) 2021, Alibaba Group;
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import torch
import math
import re

from utils.utils import divide, CommType, CommGroup
from workload_generator.mocked_model.MockedModel import MockedModel, Linear, MockedParam
from log_analyzer.log import Workload, LogItem


# mocked version of Megatron RowParallelLinear
class MegatronRowLinear(MockedModel):
    def __init__(
        self,
        input_size,
        output_size,
        tp,
        seq_len,
        batch_size,
        layer_id,
        prefix_name,
        sequence_parallel_enabled=True,
        computation_enable=False,
        name=None,
        add_bias_linear=False,
    ):
        self.layer_id = layer_id
        self.name = prefix_name + "_row"
        self.input_size, self.output_size = input_size, output_size
        self.input_size_per_partition = divide(input_size, tp)
        self.weight = MockedParam(
            (output_size, self.input_size_per_partition), name=name
        )
        if add_bias_linear:
            self.bias = MockedParam((output_size, 1), name=self.name + "_bias")
        self.sequence_parallel_enabled = sequence_parallel_enabled
        self.computation_enable = computation_enable
        self.tensor_model_parallel_size, self.seq_len, self.batch_size = tp, seq_len, batch_size
        self.comm_size = 2 * seq_len * batch_size * output_size

    def forward(self):
        workloads = Workload()
        # output_ = torch.matmul(total_input, weight.t()): (s, b, h)
        if self.computation_enable:
            workloads.append(
                LogItem(
                    comm_type=CommType.computation,
                    msg_size=(
                        (self.seq_len, self.batch_size, self.input_size_per_partition),
                        (self.input_size_per_partition, self.output_size),
                    ),
                    stage="forward.MegatronRowLinear." + self.name,
                )
            )
        if self.tensor_model_parallel_size > 1:
            if self.sequence_parallel_enabled:
                # output_ = reduce_scatter_to_sequence_parallel_region(output_parallel): (s/tp, b, h)
                workloads.append(
                    LogItem(
                        comm_type=CommType.reduce_scatter,
                        comm_group=CommGroup.tp_group,
                        comm_group_size=self.tensor_model_parallel_size,
                        msg_size=self.comm_size,
                        stage="forward.MegatronRowLinear",
                    )
                )
            else:
                # output_ = reduce_from_tensor_model_parallel_region(output_parallel)
                workloads.append(
                    LogItem(
                        comm_type=CommType.all_reduce,
                        comm_group=CommGroup.tp_group,
                        comm_group_size=self.tensor_model_parallel_size,
                        msg_size=self.comm_size,
                        stage="forward.MegatronRowLinear",
                    )
                )
        return workloads

    def backward(self):
        workloads = Workload()
        if self.tensor_model_parallel_size > 1:
            if self.sequence_parallel_enabled:
                # output_ = reduce_scatter_to_sequence_parallel_region(output_parallel): (s/tp, b, h)
                workloads.append(
                    LogItem(
                        comm_type=CommType.all_gather,
                        comm_group=CommGroup.tp_group,
                        comm_group_size=self.tensor_model_parallel_size,
                        msg_size=self.comm_size,
                        stage="backward.MegatronRowLinear",
                    )
                )
        # grad_input = grad_output.matmul(weight): (s, b, h)*(h, h'/N)
        # grad_weight = grad_output.t().matmul(total_input): (h, s*b)*(s*b, h'/N)
        if self.computation_enable:
            workloads.append(
                LogItem(
                    comm_type=CommType.computation,
                    msg_size=(
                        (self.seq_len, self.batch_size, self.output_size),
                        self.weight.shape,
                    ),
                    stage="backward.MegatronRowLinear" + self.name,
                )
            )
            workloads.append(
                LogItem(
                    comm_type=CommType.computation,
                    msg_size=(
                        (self.output_size, self.seq_len * self.batch_size),
                        (self.seq_len * self.batch_size, self.input_size_per_partition),
                    ),
                    stage="backward.MegatronRowLinear" + self.name,
                )
            )
        return workloads


class MegatronColumnLinear(MockedModel):
    def __init__(
        self,
        input_size,
        output_size,
        tp,
        seq_len,
        batch_size,
        layer_id,
        prefix_name,
        sequence_parallel_enabled=True,
        computation_enable=False,
        name=None,
        add_bias_linear=False,
    ):
        self.layer_id = layer_id
        self.name = prefix_name + "_column"
        self.input_size, self.output_size = input_size, output_size
        self.output_size_per_partition = divide(output_size, tp)
        self.weight = MockedParam(
            (input_size , self.output_size_per_partition), name=name
        )
        if add_bias_linear:
            self.bias = MockedParam(
                (self.output_size_per_partition, 1), name=self.name + "_bias"
            )
        self.sequence_parallel_enabled = sequence_parallel_enabled
        self.computation_enable = computation_enable
        self.tensor_model_parallel_size, self.seq_len, self.batch_size = tp, seq_len, batch_size
        self.comm_size = 2 * seq_len * batch_size * input_size
        if self.tensor_model_parallel_size > 1 and self.sequence_parallel_enabled:
            self.seq_len *= self.tensor_model_parallel_size

    def forward(self):
        workloads = Workload()
        if self.tensor_model_parallel_size > 1:
            if self.sequence_parallel_enabled:
                workloads.append(
                    LogItem(
                        comm_type=CommType.all_gather,
                        comm_group=CommGroup.tp_group,
                        comm_group_size=self.tensor_model_parallel_size,
                        msg_size=self.comm_size,
                        stage="forward.MegatronColumnLinear",
                    )
                )
        # output = torch.matmul(total_input, weight.t())
        if self.computation_enable:
            workloads.append(
                LogItem(
                    comm_type=CommType.computation,
                    msg_size=(
                        (self.seq_len, self.batch_size, self.input_size),
                        self.weight.shape,
                    ),
                    stage="forward.MegatronColumnLinear." + self.name,
                )
            )
        return workloads

    def backward(self):
        workloads = Workload()
        if self.tensor_model_parallel_size > 1:
            if self.sequence_parallel_enabled:
                workloads.append(
                    LogItem(
                        comm_type=CommType.all_gather,
                        comm_group=CommGroup.tp_group,
                        comm_group_size=self.tensor_model_parallel_size,
                        msg_size=self.comm_size,
                        stage="backward.MegatronColumnLinear",
                    )
                )
        # grad_input = grad_output.matmul(weight): (s, b, h'/N)*(h'/N, h)
        # grad_weight = grad_output.t().matmul(total_input): (h, s*b)*(s*b, h'/N)
        if self.computation_enable:
            workloads.append(
                LogItem(
                    comm_type=CommType.computation,
                    msg_size=(
                        (self.seq_len, self.batch_size, self.output_size_per_partition),
                        (self.output_size_per_partition, self.input_size),
                    ),
                    stage="backward.MegatronColumnLinear." + self.name,
                )
            )
        if self.tensor_model_parallel_size > 1:
            if self.sequence_parallel_enabled:
                workloads.append(
                    LogItem(
                        comm_type=CommType.reduce_scatter,
                        comm_group=CommGroup.tp_group,
                        comm_group_size=self.tensor_model_parallel_size,
                        msg_size=self.comm_size,
                        stage="backward.MegatronColumnLinear",
                    )
                )
        if self.computation_enable:
            workloads.append(
                LogItem(
                    comm_type=CommType.computation,
                    msg_size=(
                        (
                            self.output_size_per_partition,
                            self.seq_len * self.batch_size,
                        ),
                        (self.seq_len * self.batch_size, self.input_size),
                    ),
                    stage="backward.MegatronColumnLinear." + self.name,
                )
            )
        if self.tensor_model_parallel_size > 1:
            if not self.sequence_parallel_enabled:
                workloads.append(
                    LogItem(
                        comm_type=CommType.all_reduce,
                        comm_group=CommGroup.tp_group,
                        comm_group_size=self.tensor_model_parallel_size,
                        msg_size=self.comm_size,
                        stage="backward.MegatronColumnLinear",
                    )
                )
        return workloads


class FusedLayernorm(MockedModel):
    def __init__(self, hidden_size):
        self.layer_id = 0
        self.name = "fused"
        self.weight = MockedParam((hidden_size, 1))
        self.bias = MockedParam((hidden_size, 1))


class MegatronAttention(MockedModel):
    def __init__(
        self,
        num_attention_heads,
        hidden_size,
        tp,
        seq_len,
        batch_size,
        layer_id,
        sequence_parallel_enabled,
        computation_enable,
        add_bias_linear,
    ):
        self.name = "attention_layer"
        self.layer_id = layer_id
        self.kv_channels = hidden_size // num_attention_heads
        self.kv_projection_size = self.kv_channels * num_attention_heads
        self.query_projection_size = self.kv_channels * num_attention_heads
        self.qkv = MegatronColumnLinear(
            hidden_size,
            self.query_projection_size + 2 * self.kv_projection_size,
            tp,
            seq_len,
            batch_size,
            layer_id,
            "attention",
            sequence_parallel_enabled,
            computation_enable,
            name="attention_column",
            add_bias_linear=add_bias_linear,
        )
        self.attention_dense = MegatronRowLinear(
            self.query_projection_size,
            hidden_size,
            tp,
            seq_len,
            batch_size,
            layer_id,
            "attention",
            sequence_parallel_enabled,
            computation_enable,
            name="attention_row",
            add_bias_linear=add_bias_linear,
        )

    def forward(self):
        workloads = Workload()
        workloads.extend(self.qkv.forward())
        workloads.extend(self.attention_dense.forward())
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads

    def backward(self):
        workloads = Workload()
        workloads.extend(self.qkv.backward())
        workloads.extend(self.attention_dense.backward())
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads


class MegatronMlp(MockedModel):
    def __init__(
        self,
        hidden_size,
        ffn_hidden_size,
        tp,
        seq_len,
        batch_size,
        layer_id,
        sequence_parallel_enabled,
        computation_enable,
        add_bias_linear,
    ):
        self.name = "mlp_layer"
        self.layer_id = layer_id
        self.dense_h_to_4h = MegatronColumnLinear(
            hidden_size,
            ffn_hidden_size,
            tp,
            seq_len,
            batch_size,
            layer_id,
            "mlp",
            sequence_parallel_enabled,
            computation_enable,
            name="mlp_column",
            add_bias_linear=add_bias_linear,
        )
        self.dense_4h_to_h = MegatronRowLinear(
            ffn_hidden_size,
            hidden_size,
            tp,
            seq_len,
            batch_size,
            layer_id,
            "mlp",
            sequence_parallel_enabled,
            computation_enable,
            name="mlp_row",
            add_bias_linear=add_bias_linear,
        )

    def forward(self):
        workloads = Workload()
        workloads.extend(self.dense_h_to_4h.forward())
        workloads.extend(self.dense_4h_to_h.forward())
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads

    def backward(self):
        workloads = Workload()
        workloads.extend(self.dense_h_to_4h.backward())
        workloads.extend(self.dense_4h_to_h.backward())
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads


class GroupedMLP(MockedModel):
    def __init__(
        self,
        batch_size,
        hidden_size,
        tp,
        expert_model_parallel_size,
        ffn_hidden_size,
        seq_len,
        topk,
        num_experts,
        id,
    ):
        self.name = "mlp_moelayer"
        self.layer_id = id
        num_local_experts = num_experts // expert_model_parallel_size
        fc1_output_size = ffn_hidden_size * num_local_experts
        fc1_output_size_per_parttition = divide(fc1_output_size, tp)
        fc2_input_size = ffn_hidden_size * num_local_experts
        fc2_input_size_per_parttition = divide(fc2_input_size, tp)
        self.weight1 = MockedParam((hidden_size, fc1_output_size_per_parttition))
        self.weight2 = MockedParam((fc2_input_size_per_parttition, hidden_size))
        self.tp_size = tp
        self.topk = topk
        self.seq_len = seq_len
        self.num_experts = num_experts
        self.batch_size = batch_size
        self.hidden_size = hidden_size

    def permutation(self, stage):
        workloads = Workload()
        if self.tp_size > 1:
            workloads.append(
                LogItem(
                    comm_type=CommType.all_to_all,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.tp_size,
                    msg_size=self.seq_len
                    * self.hidden_size
                    * self.batch_size
                    // self.tp_size
                    * 2,
                    stage=f"{stage}.MoE",
                )
            )
        workloads.append(
            LogItem(
                comm_type=CommType.all_to_all,
                comm_group=CommGroup.ep_group,
                msg_size=self.seq_len
                * self.hidden_size
                * self.batch_size
                // self.tp_size
                * 2,
                stage=f"{stage}.MoE",
            )
        )
        if self.tp_size > 1:
            # TODO:we assume tokens consistent split to all experts, but actually its not
            workloads.append(
                LogItem(
                    comm_type=CommType.all_gather,
                    comm_group=CommGroup.tp_group,
                    msg_size=2
                    * self.hidden_size
                    * self.topk * self.batch_size
                    * self.seq_len,
                    stage=f"{stage}.MoE.permutation",
                )
            )

        return workloads

    def unpermutation(self, stage):
        workloads = Workload()
        if self.tp_size > 1:
            # TODO:we assume tokens consistent split to all experts, but actually its not
            workloads.append(
                LogItem(
                    comm_type=CommType.reduce_scatter,
                    comm_group=CommGroup.tp_group,
                    msg_size=2
                    * self.hidden_size * self.batch_size
                    * self.topk
                    * self.seq_len,
                    stage=f"{stage}.MoE.unpermutation",
                )
            )
        workloads.append(
            LogItem(
                comm_type=CommType.all_to_all,
                comm_group=CommGroup.ep_group,
                msg_size=self.seq_len
                * self.hidden_size
                * self.batch_size
                * self.topk
                // self.tp_size
                * 2,
                stage=f"{stage}.MoE",
            )
        )

        if self.tp_size > 1:
            # TODO:we assume tokens consistent split to all experts, but actually its not
            workloads.append(
                LogItem(
                    comm_type=CommType.all_to_all,
                    comm_group=CommGroup.tp_group,
                    msg_size=2 * self.hidden_size * self.seq_len * self.batch_size // self.tp_size,
                    stage=f"{stage}.MoE",
                )
            )

        return workloads

    def forward(self):
        workloads = Workload()
        workloads.append(LogItem(
                    comm_type=CommType.all_gather,
                    comm_group=CommGroup.tp_group,
                    msg_size=2 * self.hidden_size * self.batch_size * self.seq_len,
                    stage=f"forward.MoE.preprocess",
                ))
        workloads.extend(self.permutation(stage="forward"))
        workloads.extend(self.unpermutation(stage="forward"))
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads

    def backward(self):
        workloads = Workload()
        self.permutation(stage="backward")
        self.unpermutation(stage="backward")
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads


class SequentialMLP(MockedModel):
    def __init__(self):
        print("Not implement yet!")
        pass


class MegatronTransformorLayer(MockedModel):
    def __init__(
        self,
        hidden_size,
        ffn_hidden_size,
        tp,
        seq_len,
        batch_size,
        num_attention_heads,
        layer_id,
        expert_model_parallel_size,
        moe_router_topk,
        num_experts,
        moe_grouped_gemm=True,
        sequence_parallel_enabled=True,
        computation_enable=False,
        add_bias_linear=False,
        moe_enable=False,
    ):
        self.attention = MegatronAttention(
            num_attention_heads,
            hidden_size,
            tp,
            seq_len,
            batch_size,
            layer_id,
            sequence_parallel_enabled,
            computation_enable,
            add_bias_linear,
        )
        self.pre_mlp_layernorm = FusedLayernorm(hidden_size)
        self.post_attention_layernorm_bias = MockedParam((hidden_size, 1))
        if moe_enable and moe_grouped_gemm:
            self.mlp = GroupedMLP(
                batch_size,
                hidden_size,
                tp,
                expert_model_parallel_size,
                ffn_hidden_size,
                seq_len,
                moe_router_topk,
                num_experts,
                layer_id,
            )
        else:
            self.mlp = MegatronMlp(
                hidden_size,
                ffn_hidden_size,
                tp,
                seq_len,
                batch_size,
                layer_id,
                sequence_parallel_enabled,
                computation_enable,
                add_bias_linear,
            )

    def forward(self):
        workloads = Workload()
        workloads.extend(self.attention.forward())
        workloads.extend(self.mlp.forward())
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads

    def backward(self):
        workloads = Workload()
        workloads.extend(self.attention.backward())
        workloads.extend(self.mlp.backward())
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads


class MegatronEmbedding(MockedModel):
    def __init__(self, padded_vocab_size, hidden_size, tp, seq_len, batch_size):
        self.name = "embedding_layer"
        self.layer_id = 0
        num_embedding_per_partition = divide(padded_vocab_size, tp)
        self.word_embedding = MockedParam(
            (2 * num_embedding_per_partition, hidden_size), name=self.name
        )
        self.tensor_model_parallel_size = tp
        # TODO : position embedding shape is max_sequence_length not sequence_length
        self.position_embedding = MockedParam((seq_len, hidden_size))
        self.comm_size = 2 * batch_size * seq_len * hidden_size

    def forward(self):
        workloads = Workload()
        if self.tensor_model_parallel_size > 1:
            workloads.append(
                LogItem(
                    comm_type=CommType.all_reduce,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.tensor_model_parallel_size,
                    msg_size=self.comm_size,
                    stage="forward.MegatronEmbedding",
                )
            )
        print(workloads)
        return workloads

    def backward(self):
        workloads = Workload()
        if self.tensor_model_parallel_size > 1:
            workloads.append(
                LogItem(
                    comm_type=CommType.all_reduce,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.tensor_model_parallel_size,
                    msg_size=self.comm_size,
                    stage="backward.MegatronEmbedding",
                )
            )
        return workloads


class MegatronModel(MockedModel):
    def __init__(self, config):
        self.embedding = MegatronEmbedding(
            config.padded_vocab_size,
            config.hidden_size,
            config.tensor_model_parallel_size,
            config.seq_length,
            config.micro_batch,
        )
        self.layers = [
            MegatronTransformorLayer(
                config.hidden_size,
                config.ffn_hidden_size,
                config.tensor_model_parallel_size,
                config.seq_length,
                config.micro_batch,
                config.num_attention_heads,
                i,
                config.expert_model_parallel_size,
                config.moe_router_topk,
                config.num_experts,
                config.moe_grouped_gemm,
                config.enable_sequence_parallel,
                config.computation_enable,
                config.add_bias_linear,
                config.moe_enable,
            )
            for i in range(config.num_layers)
        ]
        self.final_norm = MegatronColumnLinear(
            config.hidden_size,
            config.padded_vocab_size,
            config.tensor_model_parallel_size,
            config.seq_length,
            config.micro_batch,
            1,
            "final",
            sequence_parallel_enabled=config.enable_sequence_parallel,
            computation_enable=config.computation_enable,
            add_bias_linear=config.add_bias_linear,
        )

    def forward(self):
        workloads = Workload()
        workloads.extend(self.embedding.forward())
        for layer in self.layers:
            workloads.extend(layer.forward())
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads

    def backward(self):
        workloads = Workload()
        for layer in self.layers[::-1]:
            workloads.extend(layer.backward())
        workloads.extend(self.embedding.backward())
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads
# class Config:
#     padded_vocab_size = 51200
#     hidden_size = 1024
#     tensor_model_parallel_size = 4  # 4个设备做张量并行
#     seq_length = 512
#     micro_batch = 32
#     num_attention_heads = 16
#     ffn_hidden_size = 4096
#     num_layers = 1  # 仅1层Transformer
#     expert_model_parallel_size=4
#     moe_router_topk=4
#     num_experts=16
#     moe_grouped_gemm=True
#     enable_sequence_parallel=True
#     computation_enable=False
#     add_bias_linear=False
#     moe_enable=False
#     enable_sequence_parallel=True
#     computation_enable=False
#     add_bias_linear=False
# config.py
# 该配置文件定义了 Megatron 模型及分布式训练相关的所有参数，
# 参数值均基于你提供的信息。

class Config:
    # 通信与框架相关
    frame = 'Megatron'
    world_size = 4096
    tensor_model_parallel_size = 4
    pipeline_model_parallel = 1
    context_parallel_size = 1
    pp_rank = -1
    virtual_pipeline_model_parallel=None
    untie_embeddings_and_output_weights=True
    data_parallel_size=1
    # 全局与微批次设置
    global_batch = 8192
    micro_batch = 1
    epoch_num = 1
    # 开启计算相关
    computation_enable = True
    dtype = 'bfloat16'  # 数据类型，支持 'bfloat16', 'float16', 'float32'

    # 前馈网络与 Transformer 层参数
    hidden_size = 4096
    num_layers = 40
    seq_length = 4096
    num_attention_heads = 32
    ffn_hidden_size = 10880

    # 词表与位置编码
    vocab_size = 50257
    max_position_embeddings = 4096
    padded_vocab_size = 50688

    # 模型名称（如 'gpt_7B'）
    model_name = 'gpt_7B'

    # 其他模型相关设置
    add_bias_linear = False
    use_flash_attn = False
    swiglu = True
    stage = 3
    amp_enabled = False

    # 通信相关（例如梯度通信的 bucket 大小等）
    reduce_bucket_size = 500000000
    allgather_bucket_size = 500000000
    contiguous_gradients = False

    # 参数/模型持久化和内存阈值设置
    param_persistence_threshold = 100000
    model_persistence_threshold = 9223372036854775807
    max_live_parameters = 1000000000
    prefetch_bucket_size = 1000000000

    # 是否启用序列并行和分布式优化器
    enable_sequence_parallel = False
    use_distributed_optimizer = True
    make_vocab_size_divisible_by = 128
    overlap_grad_reduce = False
    group_query_attention=False
    # 通信测试相关（微测试起始和结束大小、通信类型、迭代次数等）
    begin_size = 1048576
    end_size = 1048576
    test_comm = 'all_reduce'
    iter_num = 500
    multi_all_reduce_enable = 0

    # MoE（混合专家）相关设置（此处未启用 MoE）
    moe_enable = False
    expert_model_parallel_size = 1
    num_experts = 1
    moe_router_topk = 1
    moe_grouped_gemm = False

    # 其他选项
    activation_func = None
    overlap_version = False
    aiob_enable = True
    comp_filepath = None

    # 激活与前馈网络融合相关
    gated_linear_unit = True
    bias_gelu_fusion = False
    openai_gelu = False
    onnx_safe = False
    squared_relu = False
    recompute_activations = False

    # 数据并行（DP）相关
    dp_num = 1024
    num_microbatches = 8

    # 模型参数数量（一般由模型初始化后计算得到）
    model_param = 1579073536   
        

config = Config()
model = MegatronModel(config)
workload = model.forward()
workload.dump("/cpfs04/user/chengqinxiu/SimAI/aicb/workload_generator/mocked_model/example_workload")  # 导出为CSV
from workload_generator.mocked_model.AiobMegatron import MegatronModel as AiobMegatronModel
# class Config1:
#     # 通信框架（一般为 Megatron）
#     frame = "Megatron"
    
#     # 全局训练相关
#     world_size = 32
#     global_batch = 1024
#     micro_batch = 1
#     epoch_num = 1

#     # 模型并行相关
#     tensor_model_parallel_size = 8
#     pipeline_model_parallel = 1  # 不启用流水线并行

#     # Transformer 模型参数（以 gpt_13B 为例）
#     num_layers = 40
#     seq_length = 4096
#     hidden_size = 5120
#     num_attention_heads = 40

#     # 前馈网络：通常设置为 4 倍 hidden_size
#     ffn_hidden_size = 4 * hidden_size  # 20480

#     # 位置编码和词表参数
#     max_position_embeddings = 4096
#     vocab_size = 50257  # 原始 GPT 词表大小

#     # 模型规模选择（13 表示 gpt_13B）
#     model_size = 13
#     model_name = "gpt_13B"

#     # 是否启用 Flash Attention 加速注意力计算（默认 False）
#     use_flash_attn = False

#     # 是否使用 swiglu 激活（默认 False）
#     swiglu = False

#     # 是否启用序列并行，默认可设为 False；若启用，则模型内部部分计算会仅在局部处理后再重复扩展
#     enable_sequence_parallel = False

#     # 计算时间统计文件（默认未指定）
#     comp_filepath = ""

#     # MoE（混合专家）相关参数，若不使用则保留默认值
#     num_experts = 1
#     moe_enable = False
#     # 如果启用 MoE，还可能需要设置 moe_router_topk、expert_model_parallel_size、moe_grouped_gemm 等，
#     # 此处默认不启用 MoE

#     # 是否开启激活重计算以降低显存占用（默认 False）
#     recompute_activations = False

#     # 数据类型（可选："float32", "float16", "bfloat16"），这里推荐使用 float16
#     dtype = "float16"
#     padded_vocab_size=16
#     expert_model_parallel_size=8
#     moe_router_topk=4
#     moe_grouped_gemm=True
#     enable_sequence_parallel=True
#     computation_enable=False
#     add_bias_linear=False
#     moe_enable=False
#     gated_linear_unit=True
#     openai_gelu=True
#     squared_relu=True
#     bias_gelu_fusion=True
#     ffn_hidden_size=20480
#     tp=1
#     gated_linear_unit=True
#     model_param=1300
#     openai_gelu=True
#     dp_num=1
#     # 其他参数，如是否使用分布式优化器、路由 topk 值、专家并行参数等，
#     # 根据实际需要添加。如果需要，可继续扩展该配置类。
    

#     # 其他配置参数请根据实际需求补充...
# config = Config1()
# measure_model = AiobMegatronModel(config)
# measure_model.train()
# batch_size = config.micro_batch
# vocab_size=config.padded_vocab_size
# tp=config.tensor_model_parallel_size
# seq_len = config.seq_length
# device = "cuda"
# dtype = torch.float32

# # # total_input_1 = torch.rand(args.seq_len,
# # #                                       args.batch_size,
# # #                                       args.hidden_size,
# # #                                       device=device).to(dtype)
# masked_input = torch.randint(
#     0,
#     math.ceil(vocab_size / tp),
#     (batch_size, seq_len),
#     device=device,
#     dtype=torch.int64,
# )
# filepath = measure_model(masked_input)
# attention_avg_sum = 0.0
# mlp_avg_sum = 0.0
# other_avgs = {}
# grad_forward = 0.0
# grad_backward = 0.0

# section_header_re = re.compile(r"^(\w+):")
# time_gpu_avg_re = re.compile(r"time_gpu_avg:\s+(\d+(\.\d+)?)")
# time_gpu_min_re = re.compile(r"time_gpu_min:\s+(\d+(\.\d+)?)")

# with open(filepath, "r") as file:
#     current_section = None

#     for line in file:
#         header_match = section_header_re.match(line)
#         if header_match:
#             current_section = header_match.group(1).strip()

#         avg_match = time_gpu_avg_re.search(line)
#         min_match = time_gpu_min_re.search(line)
#         if current_section == "param_time":
#             if min_match:
#                 grad_forward = float(min_match.group(1)) * 1000 #us
#             if avg_match:
#                 grad_backward = float(avg_match.group(1)) * 1000
#         elif avg_match and current_section:
#             avg_value = float(avg_match.group(1)) * 1000
#             if "atten" in current_section or current_section == "layernorm":
#                 attention_avg_sum += avg_value
#             elif "mlp" in current_section or current_section == "layernorm2":
#                 mlp_avg_sum += avg_value
#             else:
#                 other_avgs[current_section] = avg_value

# # 四舍五入并转换为整数
# attention_forward = round(attention_avg_sum)
# attention_backward = attention_forward
# mlp_forward = round(mlp_avg_sum)
# mlp_backward = mlp_forward
# grad_backward = round(grad_backward)
# grad_forward = round(grad_forward)
# other_avgs_int = {k: round(v) for k, v in other_avgs.items() if k != "param_time"}

# a100_compute_cache = {
#     "attention_forward": attention_forward,
#     "attention_backward": attention_backward,
#     "mlp_forward": mlp_forward,
#     "mlp_backward": mlp_backward,
#     "grad_forward": grad_forward,
#     "grad_backward": grad_backward,
# }
# a100_compute_cache.update(other_avgs_int)
# for key, value in a100_compute_cache.items():
#             print(f"    '{key}' : {value},")
# print("}")