# 多GPU支持设计

## 🎯 设计目标

多GPU支持是Mirage Persistent Kernel实现大规模LLM推理的关键技术，主要目标包括：
- **可扩展性**：支持从2个到数百个GPU的灵活扩展
- **高效通信**：最小化GPU间通信开销
- **负载均衡**：在多个GPU间均匀分配计算负载
- **容错性**：处理GPU故障和网络异常

## 🏗️ 多GPU架构

```
┌─────────────────────────────────────────────────────────────┐
│                   Multi-GPU Architecture                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │DistributedCoord │  │   LoadBalancer   │  │CommOptimizer│ │
│  │   (分布式协调)   │  │   (负载均衡器)   │  │ (通信优化)   │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │   NVSHMEM       │  │      MPI         │  │   NCCL      │ │
│  │   (GPU直连)     │  │   (进程通信)     │  │ (集合通信)   │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │ TensorParallel  │  │ PipelineParallel │  │DataParallel │ │
│  │  (张量并行)     │  │  (流水线并行)    │  │ (数据并行)   │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │  FaultTolerance │  │  MemoryManager   │  │ Synchronizer│ │
│  │   (容错处理)     │  │  (内存管理)      │  │ (同步控制)   │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🌐 分布式协调器

### 核心协调机制
```cpp
namespace mirage::distributed {

class DistributedCoordinator {
private:
    struct GPUInfo {
        int gpu_id;
        int node_id;
        std::string hostname;
        size_t memory_capacity;
        float compute_capability;
        bool is_active;
        std::chrono::steady_clock::time_point last_heartbeat;
    };
    
    struct TaskPartition {
        int partition_id;
        std::vector<int> assigned_gpus;
        size_t data_size;
        ComputeType compute_type;
        std::vector<TaskId> tasks;
    };
    
    std::vector<GPUInfo> gpu_cluster;
    std::vector<TaskPartition> task_partitions;
    int world_size;
    int my_rank;
    
public:
    DistributedCoordinator(int world_size, int rank) 
        : world_size(world_size), my_rank(rank) {
        initialize_cluster();
    }
    
    bool initialize_cluster() {
        // 1. MPI初始化
        MPI_Init(nullptr, nullptr);
        int mpi_size, mpi_rank;
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        
        if (mpi_size != world_size || mpi_rank != my_rank) {
            LOG_ERROR("MPI configuration mismatch");
            return false;
        }
        
        // 2. NVSHMEM初始化
        nvshmemx_init_attr_t attr;
        attr.mpi_comm = &MPI_COMM_WORLD;
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
        
        // 3. 收集集群信息
        collect_cluster_info();
        
        // 4. 建立心跳机制
        start_heartbeat_monitor();
        
        return true;
    }
    
    void coordinate_distributed_execution(const LLMInferenceTask& task) {
        // 1. 分析任务并创建执行计划
        auto execution_plan = create_execution_plan(task);
        
        // 2. 分发任务到各个GPU
        distribute_tasks(execution_plan);
        
        // 3. 同步启动执行
        synchronized_execution_start();
        
        // 4. 监控执行进度
        monitor_execution_progress();
        
        // 5. 收集结果
        collect_execution_results();
    }
    
private:
    void collect_cluster_info() {
        GPUInfo local_gpu_info;
        local_gpu_info.gpu_id = my_rank;
        local_gpu_info.node_id = get_node_id();
        local_gpu_info.hostname = get_hostname();
        local_gpu_info.memory_capacity = get_gpu_memory_capacity();
        local_gpu_info.compute_capability = get_compute_capability();
        local_gpu_info.is_active = true;
        local_gpu_info.last_heartbeat = std::chrono::steady_clock::now();
        
        // 使用MPI_Allgather收集所有GPU信息
        std::vector<GPUInfo> all_gpu_info(world_size);
        MPI_Allgather(&local_gpu_info, sizeof(GPUInfo), MPI_BYTE,
                     all_gpu_info.data(), sizeof(GPUInfo), MPI_BYTE,
                     MPI_COMM_WORLD);
        
        gpu_cluster = std::move(all_gpu_info);
    }
    
    ExecutionPlan create_execution_plan(const LLMInferenceTask& task) {
        ExecutionPlan plan;
        
        // 根据模型大小和GPU配置选择并行策略
        if (task.model_size > LARGE_MODEL_THRESHOLD) {
            plan.parallelism_type = ParallelismType::TENSOR_PARALLEL;
            plan.tensor_parallel_size = calculate_optimal_tp_size(task.model_size);
        } else {
            plan.parallelism_type = ParallelismType::DATA_PARALLEL;
            plan.data_parallel_size = world_size;
        }
        
        // 创建任务分区
        plan.partitions = create_task_partitions(task, plan.parallelism_type);
        
        return plan;
    }
    
    std::vector<TaskPartition> create_task_partitions(
        const LLMInferenceTask& task, ParallelismType parallelism_type) {
        
        std::vector<TaskPartition> partitions;
        
        switch (parallelism_type) {
            case ParallelismType::TENSOR_PARALLEL:
                partitions = create_tensor_parallel_partitions(task);
                break;
            case ParallelismType::PIPELINE_PARALLEL:
                partitions = create_pipeline_parallel_partitions(task);
                break;
            case ParallelismType::DATA_PARALLEL:
                partitions = create_data_parallel_partitions(task);
                break;
            case ParallelismType::HYBRID:
                partitions = create_hybrid_parallel_partitions(task);
                break;
        }
        
        return partitions;
    }
};

}
```

### 动态负载均衡
```cpp
class DynamicLoadBalancer {
private:
    struct GPUWorkload {
        int gpu_id;
        float current_load;           // 当前负载 (0.0-1.0)
        float memory_usage;           // 内存使用率
        float communication_cost;     // 通信成本
        int pending_tasks;            // 待处理任务数
        std::chrono::steady_clock::time_point last_update;
    };
    
    std::vector<GPUWorkload> gpu_workloads;
    std::mutex load_balance_mutex;
    
public:
    void update_gpu_workload(int gpu_id, const WorkloadMetrics& metrics) {
        std::lock_guard<std::mutex> lock(load_balance_mutex);
        
        auto& workload = gpu_workloads[gpu_id];
        workload.current_load = metrics.compute_utilization;
        workload.memory_usage = metrics.memory_utilization;
        workload.pending_tasks = metrics.queue_length;
        workload.last_update = std::chrono::steady_clock::now();
        
        // 计算通信成本（基于与其他GPU的距离）
        workload.communication_cost = calculate_communication_cost(gpu_id);
    }
    
    std::vector<TaskReassignment> rebalance_workload() {
        std::lock_guard<std::mutex> lock(load_balance_mutex);
        
        std::vector<TaskReassignment> reassignments;
        
        // 找到负载最高和最低的GPU
        auto max_load_gpu = std::max_element(gpu_workloads.begin(), gpu_workloads.end(),
            [](const GPUWorkload& a, const GPUWorkload& b) {
                return calculate_total_load(a) < calculate_total_load(b);
            });
        
        auto min_load_gpu = std::min_element(gpu_workloads.begin(), gpu_workloads.end(),
            [](const GPUWorkload& a, const GPUWorkload& b) {
                return calculate_total_load(a) < calculate_total_load(b);
            });
        
        float load_imbalance = calculate_total_load(*max_load_gpu) - 
                              calculate_total_load(*min_load_gpu);
        
        // 如果负载不均衡超过阈值，进行重分配
        if (load_imbalance > LOAD_IMBALANCE_THRESHOLD) {
            TaskReassignment reassignment;
            reassignment.from_gpu = max_load_gpu->gpu_id;
            reassignment.to_gpu = min_load_gpu->gpu_id;
            reassignment.num_tasks = calculate_optimal_task_migration(
                *max_load_gpu, *min_load_gpu);
            
            reassignments.push_back(reassignment);
        }
        
        return reassignments;
    }
    
private:
    float calculate_total_load(const GPUWorkload& workload) {
        // 综合考虑计算负载、内存使用和通信成本
        float compute_weight = 0.5f;
        float memory_weight = 0.3f;
        float comm_weight = 0.2f;
        
        return compute_weight * workload.current_load +
               memory_weight * workload.memory_usage +
               comm_weight * workload.communication_cost;
    }
    
    float calculate_communication_cost(int gpu_id) {
        float total_cost = 0.0f;
        
        for (const auto& other_workload : gpu_workloads) {
            if (other_workload.gpu_id != gpu_id) {
                // 计算与其他GPU的通信成本
                float distance = get_gpu_distance(gpu_id, other_workload.gpu_id);
                float bandwidth = get_interconnect_bandwidth(gpu_id, other_workload.gpu_id);
                total_cost += distance / bandwidth;
            }
        }
        
        return total_cost / (gpu_workloads.size() - 1);
    }
};
```

## ⚡ 并行策略实现

### 张量并行
```cpp
class TensorParallelManager {
private:
    int tensor_parallel_size;
    int tensor_parallel_rank;
    std::vector<int> tp_group_ranks;
    
public:
    TensorParallelManager(int tp_size, int tp_rank) 
        : tensor_parallel_size(tp_size), tensor_parallel_rank(tp_rank) {
        initialize_tp_group();
    }
    
    void split_linear_layer(const LinearLayerDesc& layer,
                           std::vector<LinearLayerDesc>& split_layers) {
        split_layers.resize(tensor_parallel_size);
        
        // 按列分割权重矩阵
        int output_size_per_gpu = layer.output_size / tensor_parallel_size;
        
        for (int i = 0; i < tensor_parallel_size; ++i) {
            split_layers[i].input_size = layer.input_size;
            split_layers[i].output_size = output_size_per_gpu;
            split_layers[i].weight_offset = i * output_size_per_gpu;
            
            // 分配权重数据
            size_t weight_size = layer.input_size * output_size_per_gpu * sizeof(float);
            split_layers[i].weight_data = allocate_gpu_memory(weight_size);
            
            // 拷贝对应的权重分片
            copy_weight_slice(layer.weight_data, split_layers[i].weight_data,
                             layer.input_size, output_size_per_gpu, i);
        }
    }
    
    void execute_tensor_parallel_linear(const DTensor& input,
                                       const LinearLayerDesc& local_layer,
                                       DTensor& output) {
        // 1. 每个GPU计算本地的矩阵乘法
        DTensor local_output = matmul(input, local_layer.weight_data);
        
        // 2. AllGather收集所有GPU的结果
        std::vector<DTensor> all_outputs(tensor_parallel_size);
        allgather_tensor(local_output, all_outputs, tp_group_ranks);
        
        // 3. 拼接结果
        concatenate_tensors(all_outputs, output, /*dim=*/1);
    }
    
    void execute_tensor_parallel_attention(const DTensor& input,
                                          const AttentionLayerDesc& local_layer,
                                          DTensor& output) {
        // 1. QKV投影（每个GPU处理部分head）
        int heads_per_gpu = local_layer.num_heads / tensor_parallel_size;
        DTensor local_qkv = compute_qkv_projection(input, local_layer, heads_per_gpu);
        
        // 2. 本地注意力计算
        DTensor local_attn_output = compute_local_attention(local_qkv, heads_per_gpu);
        
        // 3. 输出投影
        DTensor local_output = compute_output_projection(local_attn_output, local_layer);
        
        // 4. AllReduce汇总结果
        allreduce_tensor(local_output, output, tp_group_ranks);
    }
    
private:
    void initialize_tp_group() {
        // 创建张量并行通信组
        tp_group_ranks.resize(tensor_parallel_size);
        for (int i = 0; i < tensor_parallel_size; ++i) {
            tp_group_ranks[i] = i;
        }
        
        // 初始化NCCL通信器
        ncclUniqueId nccl_id;
        if (tensor_parallel_rank == 0) {
            ncclGetUniqueId(&nccl_id);
        }
        MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
        
        ncclComm_t nccl_comm;
        ncclCommInitRank(&nccl_comm, tensor_parallel_size, nccl_id, tensor_parallel_rank);
        
        this->nccl_comm = nccl_comm;
    }
    
    void allgather_tensor(const DTensor& local_tensor,
                         std::vector<DTensor>& all_tensors,
                         const std::vector<int>& group_ranks) {
        size_t tensor_size = local_tensor.size_in_bytes();
        
        // 为所有张量分配内存
        for (auto& tensor : all_tensors) {
            tensor.allocate(local_tensor.dims, local_tensor.data_type);
        }
        
        // 使用NCCL AllGather
        ncclAllGather(local_tensor.data_ptr(), all_tensors[0].data_ptr(),
                     tensor_size, ncclFloat, nccl_comm, cuda_stream);
        
        cudaStreamSynchronize(cuda_stream);
    }
};
```

### 流水线并行
```cpp
class PipelineParallelManager {
private:
    struct PipelineStage {
        int stage_id;
        int start_layer;
        int end_layer;
        std::vector<int> gpu_ranks;
        size_t memory_requirement;
    };
    
    std::vector<PipelineStage> pipeline_stages;
    int pipeline_parallel_size;
    int pipeline_parallel_rank;
    int micro_batch_size;
    
public:
    PipelineParallelManager(int pp_size, int pp_rank, int micro_batch_size)
        : pipeline_parallel_size(pp_size), pipeline_parallel_rank(pp_rank),
          micro_batch_size(micro_batch_size) {
        create_pipeline_stages();
    }
    
    void execute_pipeline_parallel_inference(const DTensor& input, DTensor& output) {
        // 将输入分割为微批次
        std::vector<DTensor> micro_batches = split_into_micro_batches(input);
        
        // GPipe风格的流水线执行
        std::vector<DTensor> stage_outputs;
        
        for (int micro_batch_id = 0; micro_batch_id < micro_batches.size(); ++micro_batch_id) {
            // 前向传播
            DTensor stage_input = (pipeline_parallel_rank == 0) ? 
                                 micro_batches[micro_batch_id] : 
                                 receive_from_previous_stage();
            
            DTensor stage_output = execute_local_stage(stage_input);
            
            if (pipeline_parallel_rank < pipeline_parallel_size - 1) {
                // 发送到下一个stage
                send_to_next_stage(stage_output);
            } else {
                // 最后一个stage，保存输出
                stage_outputs.push_back(stage_output);
            }
            
            // 流水线同步
            pipeline_synchronize();
        }
        
        // 合并微批次输出
        if (pipeline_parallel_rank == pipeline_parallel_size - 1) {
            concatenate_micro_batches(stage_outputs, output);
        }
    }
    
private:
    void create_pipeline_stages() {
        int layers_per_stage = total_layers / pipeline_parallel_size;
        
        for (int stage = 0; stage < pipeline_parallel_size; ++stage) {
            PipelineStage pipeline_stage;
            pipeline_stage.stage_id = stage;
            pipeline_stage.start_layer = stage * layers_per_stage;
            pipeline_stage.end_layer = (stage + 1) * layers_per_stage - 1;
            
            // 最后一个stage处理剩余的层
            if (stage == pipeline_parallel_size - 1) {
                pipeline_stage.end_layer = total_layers - 1;
            }
            
            pipeline_stage.gpu_ranks = {stage};  // 简单映射，一个stage一个GPU
            pipeline_stage.memory_requirement = estimate_stage_memory(pipeline_stage);
            
            pipeline_stages.push_back(pipeline_stage);
        }
    }
    
    DTensor execute_local_stage(const DTensor& input) {
        DTensor current_input = input;
        
        const auto& my_stage = pipeline_stages[pipeline_parallel_rank];
        
        // 执行分配给此stage的所有层
        for (int layer_id = my_stage.start_layer; layer_id <= my_stage.end_layer; ++layer_id) {
            current_input = execute_transformer_layer(current_input, layer_id);
        }
        
        return current_input;
    }
    
    void send_to_next_stage(const DTensor& tensor) {
        int next_rank = pipeline_parallel_rank + 1;
        
        // 使用NVSHMEM进行点对点通信
        nvshmem_float_put_nbi(remote_buffer_ptr, tensor.data_ptr(), 
                             tensor.num_elements(), next_rank);
        nvshmem_quiet();
        
        // 发送完成信号
        send_completion_signal(next_rank);
    }
    
    DTensor receive_from_previous_stage() {
        int prev_rank = pipeline_parallel_rank - 1;
        
        // 等待数据到达
        wait_for_data_signal(prev_rank);
        
        // 从共享内存读取数据
        DTensor received_tensor;
        received_tensor.allocate_from_nvshmem_buffer(local_buffer_ptr, expected_tensor_dims);
        
        return received_tensor;
    }
};
```

## 🌐 通信优化

### 高效集合通信
```cpp
class OptimizedCollectiveCommunication {
private:
    ncclComm_t nccl_comm;
    cudaStream_t comm_stream;
    
public:
    void optimized_allreduce(void* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, ncclRedOp_t op) {
        
        // 根据数据大小选择最优算法
        if (count * get_datatype_size(datatype) < SMALL_MSG_THRESHOLD) {
            // 小消息：使用递归加倍算法
            recursive_doubling_allreduce(sendbuff, recvbuff, count, datatype, op);
        } else {
            // 大消息：使用环形算法
            ring_allreduce(sendbuff, recvbuff, count, datatype, op);
        }
    }
    
    void hierarchical_allreduce(void* sendbuff, void* recvbuff, size_t count,
                               ncclDataType_t datatype, ncclRedOp_t op) {
        // 分层AllReduce：先节点内，再节点间
        
        // 第一阶段：节点内reduce-scatter
        void* temp_buffer = allocate_temp_buffer(count * get_datatype_size(datatype));
        
        intra_node_reduce_scatter(sendbuff, temp_buffer, count, datatype, op);
        
        // 第二阶段：节点间allreduce
        if (is_inter_node_root()) {
            inter_node_allreduce(temp_buffer, temp_buffer, count, datatype, op);
        }
        
        // 第三阶段：节点内allgather
        intra_node_allgather(temp_buffer, recvbuff, count, datatype);
        
        free_temp_buffer(temp_buffer);
    }
    
    void overlapped_allreduce(void* sendbuff, void* recvbuff, size_t count,
                             ncclDataType_t datatype, ncclRedOp_t op,
                             cudaStream_t compute_stream) {
        
        // 将数据分块以实现计算通信重叠
        const int num_chunks = 4;
        size_t chunk_size = count / num_chunks;
        
        for (int chunk = 0; chunk < num_chunks; ++chunk) {
            size_t offset = chunk * chunk_size;
            size_t current_chunk_size = (chunk == num_chunks - 1) ? 
                                       (count - offset) : chunk_size;
            
            // 异步启动AllReduce
            ncclAllReduce(static_cast<char*>(sendbuff) + offset * get_datatype_size(datatype),
                         static_cast<char*>(recvbuff) + offset * get_datatype_size(datatype),
                         current_chunk_size, datatype, op, nccl_comm, comm_stream);
            
            // 与计算流同步
            if (chunk > 0) {
                // 前一个chunk的通信完成后可以开始使用数据进行计算
                cudaStreamWaitEvent(compute_stream, chunk_ready_events[chunk - 1], 0);
            }
            
            // 记录当前chunk完成事件
            cudaEventRecord(chunk_ready_events[chunk], comm_stream);
        }
    }
    
private:
    void ring_allreduce(void* sendbuff, void* recvbuff, size_t count,
                       ncclDataType_t datatype, ncclRedOp_t op) {
        
        int world_size = get_world_size();
        int rank = get_rank();
        size_t chunk_size = count / world_size;
        
        // Reduce-scatter阶段
        for (int step = 0; step < world_size - 1; ++step) {
            int send_chunk = (rank - step + world_size) % world_size;
            int recv_chunk = (rank - step - 1 + world_size) % world_size;
            
            int send_rank = (rank + 1) % world_size;
            int recv_rank = (rank - 1 + world_size) % world_size;
            
            // 异步发送和接收
            async_send_recv(sendbuff, recvbuff, chunk_size, send_chunk, recv_chunk,
                           send_rank, recv_rank, datatype, op);
        }
        
        // Allgather阶段
        for (int step = 0; step < world_size - 1; ++step) {
            int send_chunk = (rank - step + world_size) % world_size;
            int send_rank = (rank + 1) % world_size;
            int recv_rank = (rank - 1 + world_size) % world_size;
            
            async_send_recv_gather(recvbuff, chunk_size, send_chunk,
                                  send_rank, recv_rank, datatype);
        }
    }
    
    void compression_aware_allreduce(void* sendbuff, void* recvbuff, size_t count,
                                    ncclDataType_t datatype, ncclRedOp_t op) {
        
        // 检查是否值得压缩
        if (should_compress(count, datatype)) {
            // 压缩数据
            void* compressed_buffer;
            size_t compressed_size;
            compress_tensor(sendbuff, count, datatype, &compressed_buffer, &compressed_size);
            
            // 对压缩数据执行AllReduce
            void* compressed_result = allocate_temp_buffer(compressed_size);
            ncclAllReduce(compressed_buffer, compressed_result, compressed_size,
                         ncclInt8, ncclSum, nccl_comm, comm_stream);
            
            // 解压缩结果
            decompress_tensor(compressed_result, recvbuff, count, datatype);
            
            free_temp_buffer(compressed_buffer);
            free_temp_buffer(compressed_result);
        } else {
            // 直接执行标准AllReduce
            ncclAllReduce(sendbuff, recvbuff, count, datatype, op, nccl_comm, comm_stream);
        }
    }
};
```

### 智能拓扑感知
```cpp
class TopologyAwareOptimizer {
private:
    struct GPUTopology {
        int gpu_id;
        int node_id;
        std::vector<int> nvlink_peers;     // NVLink直连的GPU
        std::vector<int> pcie_peers;       // PCIe连接的GPU
        std::vector<int> network_peers;    // 网络连接的GPU
        float nvlink_bandwidth;
        float pcie_bandwidth;
        float network_bandwidth;
    };
    
    std::vector<GPUTopology> topology;
    
public:
    void discover_topology() {
        int num_gpus = get_num_gpus();
        topology.resize(num_gpus);
        
        for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
            discover_gpu_connections(gpu_id);
        }
        
        // 构建通信成本矩阵
        build_communication_cost_matrix();
    }
    
    std::vector<int> optimize_allreduce_tree(int root_gpu) {
        // 使用最小生成树算法构建最优通信树
        std::vector<std::vector<float>> cost_matrix = get_communication_costs();
        
        // Prim算法构建MST
        std::vector<bool> in_tree(topology.size(), false);
        std::vector<int> parent(topology.size(), -1);
        std::vector<float> key(topology.size(), INFINITY);
        
        key[root_gpu] = 0.0f;
        
        for (int count = 0; count < topology.size() - 1; ++count) {
            int u = find_min_key_vertex(key, in_tree);
            in_tree[u] = true;
            
            for (int v = 0; v < topology.size(); ++v) {
                if (!in_tree[v] && cost_matrix[u][v] < key[v]) {
                    parent[v] = u;
                    key[v] = cost_matrix[u][v];
                }
            }
        }
        
        return parent;
    }
    
    void optimize_data_placement(std::vector<DTensor>& tensors) {
        // 根据拓扑结构优化数据放置
        for (auto& tensor : tensors) {
            int optimal_gpu = find_optimal_gpu_for_tensor(tensor);
            if (optimal_gpu != tensor.current_gpu_id) {
                migrate_tensor(tensor, optimal_gpu);
            }
        }
    }
    
private:
    void discover_gpu_connections(int gpu_id) {
        auto& gpu_topo = topology[gpu_id];
        gpu_topo.gpu_id = gpu_id;
        gpu_topo.node_id = get_node_id_for_gpu(gpu_id);
        
        // 发现NVLink连接
        for (int peer_gpu = 0; peer_gpu < get_num_gpus(); ++peer_gpu) {
            if (peer_gpu != gpu_id) {
                int can_access_peer;
                cudaDeviceCanAccessPeer(&can_access_peer, gpu_id, peer_gpu);
                
                if (can_access_peer) {
                    // 检查是否是NVLink连接
                    if (is_nvlink_connection(gpu_id, peer_gpu)) {
                        gpu_topo.nvlink_peers.push_back(peer_gpu);
                    } else {
                        gpu_topo.pcie_peers.push_back(peer_gpu);
                    }
                } else {
                    // 网络连接
                    gpu_topo.network_peers.push_back(peer_gpu);
                }
            }
        }
        
        // 测量带宽
        gpu_topo.nvlink_bandwidth = measure_nvlink_bandwidth(gpu_id);
        gpu_topo.pcie_bandwidth = measure_pcie_bandwidth(gpu_id);
        gpu_topo.network_bandwidth = measure_network_bandwidth(gpu_id);
    }
    
    int find_optimal_gpu_for_tensor(const DTensor& tensor) {
        // 分析张量的访问模式
        auto access_pattern = analyze_tensor_access_pattern(tensor);
        
        float min_cost = INFINITY;
        int optimal_gpu = tensor.current_gpu_id;
        
        for (int gpu_id = 0; gpu_id < topology.size(); ++gpu_id) {
            float placement_cost = calculate_placement_cost(tensor, gpu_id, access_pattern);
            
            if (placement_cost < min_cost) {
                min_cost = placement_cost;
                optimal_gpu = gpu_id;
            }
        }
        
        return optimal_gpu;
    }
    
    float calculate_placement_cost(const DTensor& tensor, int target_gpu,
                                  const TensorAccessPattern& access_pattern) {
        float total_cost = 0.0f;
        
        // 计算与访问此张量的GPU的通信成本
        for (const auto& access : access_pattern.gpu_accesses) {
            int accessing_gpu = access.gpu_id;
            float access_frequency = access.frequency;
            
            if (accessing_gpu != target_gpu) {
                float comm_cost = get_communication_cost(target_gpu, accessing_gpu);
                total_cost += comm_cost * access_frequency * tensor.size_in_bytes();
            }
        }
        
        return total_cost;
    }
};
```

## 🛡️ 容错机制

### GPU故障检测和恢复
```cpp
class FaultToleranceManager {
private:
    struct GPUHealthStatus {
        int gpu_id;
        bool is_healthy;
        std::chrono::steady_clock::time_point last_heartbeat;
        int consecutive_failures;
        float error_rate;
    };
    
    std::vector<GPUHealthStatus> gpu_health;
    std::mutex health_mutex;
    
public:
    void monitor_gpu_health() {
        while (monitoring_active) {
            for (auto& health : gpu_health) {
                bool current_status = check_gpu_health(health.gpu_id);
                
                std::lock_guard<std::mutex> lock(health_mutex);
                
                if (!current_status) {
                    health.consecutive_failures++;
                    health.error_rate = calculate_error_rate(health.gpu_id);
                    
                    if (health.consecutive_failures > FAILURE_THRESHOLD) {
                        handle_gpu_failure(health.gpu_id);
                    }
                } else {
                    health.consecutive_failures = 0;
                    health.last_heartbeat = std::chrono::steady_clock::now();
                }
                
                health.is_healthy = current_status;
            }
            
            std::this_thread::sleep_for(std::chrono::seconds(HEALTH_CHECK_INTERVAL));
        }
    }
    
    void handle_gpu_failure(int failed_gpu_id) {
        LOG_ERROR("GPU " + std::to_string(failed_gpu_id) + " failed");
        
        // 1. 标记GPU为不可用
        mark_gpu_unavailable(failed_gpu_id);
        
        // 2. 重新分配任务
        redistribute_tasks_from_failed_gpu(failed_gpu_id);
        
        // 3. 更新通信拓扑
        update_communication_topology_after_failure(failed_gpu_id);
        
        // 4. 如果可能，尝试恢复GPU
        attempt_gpu_recovery(failed_gpu_id);
    }
    
private:
    bool check_gpu_health(int gpu_id) {
        cudaSetDevice(gpu_id);
        
        // 1. 检查CUDA错误
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            LOG_WARNING("CUDA error on GPU " + std::to_string(gpu_id) + 
                       ": " + cudaGetErrorString(error));
            return false;
        }
        
        // 2. 执行简单的内存测试
        void* test_ptr;
        error = cudaMalloc(&test_ptr, 1024);
        if (error != cudaSuccess) {
            return false;
        }
        
        error = cudaMemset(test_ptr, 0, 1024);
        cudaFree(test_ptr);
        
        if (error != cudaSuccess) {
            return false;
        }
        
        // 3. 检查GPU温度和功耗（如果支持）
        if (is_gpu_overheating(gpu_id) || is_gpu_power_throttling(gpu_id)) {
            return false;
        }
        
        return true;
    }
    
    void redistribute_tasks_from_failed_gpu(int failed_gpu_id) {
        // 获取失败GPU上的所有任务
        auto failed_tasks = get_tasks_on_gpu(failed_gpu_id);
        
        // 找到可用的GPU
        std::vector<int> available_gpus = get_available_gpus();
        
        if (available_gpus.empty()) {
            LOG_CRITICAL("No available GPUs for task redistribution");
            throw std::runtime_error("System failure: no available GPUs");
        }
        
        // 使用负载均衡算法重新分配任务
        LoadBalancer load_balancer;
        auto redistribution_plan = load_balancer.create_redistribution_plan(
            failed_tasks, available_gpus);
        
        // 执行重分配
        for (const auto& assignment : redistribution_plan) {
            migrate_task(assignment.task_id, assignment.target_gpu);
        }
    }
    
    void attempt_gpu_recovery(int gpu_id) {
        LOG_INFO("Attempting to recover GPU " + std::to_string(gpu_id));
        
        // 1. 重置GPU上下文
        cudaSetDevice(gpu_id);
        cudaDeviceReset();
        
        // 2. 重新初始化CUDA上下文
        cudaFree(0);  // 触发上下文初始化
        
        // 3. 重新初始化NVSHMEM（如果使用）
        if (using_nvshmem) {
            reinitialize_nvshmem_for_gpu(gpu_id);
        }
        
        // 4. 运行健康检查
        if (check_gpu_health(gpu_id)) {
            LOG_INFO("GPU " + std::to_string(gpu_id) + " recovered successfully");
            mark_gpu_available(gpu_id);
        } else {
            LOG_ERROR("Failed to recover GPU " + std::to_string(gpu_id));
        }
    }
};
```

### 检查点和恢复
```cpp
class CheckpointManager {
private:
    struct Checkpoint {
        std::string checkpoint_id;
        std::chrono::steady_clock::time_point timestamp;
        std::vector<DTensor> tensor_snapshots;
        ExecutionState execution_state;
        size_t checkpoint_size;
    };
    
    std::vector<Checkpoint> checkpoints;
    std::string checkpoint_dir;
    
public:
    void create_checkpoint(const std::string& checkpoint_id) {
        Checkpoint checkpoint;
        checkpoint.checkpoint_id = checkpoint_id;
        checkpoint.timestamp = std::chrono::steady_clock::now();
        
        // 1. 保存所有张量状态
        checkpoint.tensor_snapshots = create_tensor_snapshots();
        
        // 2. 保存执行状态
        checkpoint.execution_state = capture_execution_state();
        
        // 3. 计算检查点大小
        checkpoint.checkpoint_size = calculate_checkpoint_size(checkpoint);
        
        // 4. 异步写入存储
        async_write_checkpoint(checkpoint);
        
        checkpoints.push_back(checkpoint);
        
        // 5. 清理旧检查点
        cleanup_old_checkpoints();
    }
    
    bool restore_from_checkpoint(const std::string& checkpoint_id) {
        auto checkpoint_it = std::find_if(checkpoints.begin(), checkpoints.end(),
            [&checkpoint_id](const Checkpoint& cp) {
                return cp.checkpoint_id == checkpoint_id;
            });
        
        if (checkpoint_it == checkpoints.end()) {
            LOG_ERROR("Checkpoint not found: " + checkpoint_id);
            return false;
        }
        
        const auto& checkpoint = *checkpoint_it;
        
        try {
            // 1. 恢复张量状态
            restore_tensor_snapshots(checkpoint.tensor_snapshots);
            
            // 2. 恢复执行状态
            restore_execution_state(checkpoint.execution_state);
            
            // 3. 重新初始化通信
            reinitialize_communication();
            
            LOG_INFO("Successfully restored from checkpoint: " + checkpoint_id);
            return true;
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to restore checkpoint: " + std::string(e.what()));
            return false;
        }
    }
    
private:
    std::vector<DTensor> create_tensor_snapshots() {
        std::vector<DTensor> snapshots;
        
        // 遍历所有活跃张量
        for (const auto& tensor : get_all_active_tensors()) {
            DTensor snapshot;
            snapshot.allocate(tensor.dims, tensor.data_type);
            
            // 拷贝张量数据
            cudaMemcpy(snapshot.data_ptr(), tensor.data_ptr(),
                      tensor.size_in_bytes(), cudaMemcpyDeviceToDevice);
            
            snapshots.push_back(std::move(snapshot));
        }
        
        return snapshots;
    }
    
    void async_write_checkpoint(const Checkpoint& checkpoint) {
        // 使用后台线程异步写入
        std::thread write_thread([this, checkpoint]() {
            std::string checkpoint_path = checkpoint_dir + "/" + checkpoint.checkpoint_id;
            
            // 创建检查点文件
            std::ofstream checkpoint_file(checkpoint_path, std::ios::binary);
            if (!checkpoint_file) {
                LOG_ERROR("Failed to create checkpoint file: " + checkpoint_path);
                return;
            }
            
            // 写入元数据
            write_checkpoint_metadata(checkpoint_file, checkpoint);
            
            // 写入张量数据
            for (const auto& tensor : checkpoint.tensor_snapshots) {
                write_tensor_to_file(checkpoint_file, tensor);
            }
            
            // 写入执行状态
            write_execution_state(checkpoint_file, checkpoint.execution_state);
            
            checkpoint_file.close();
            LOG_INFO("Checkpoint written: " + checkpoint_path);
        });
        
        write_thread.detach();
    }
};
```

多GPU支持通过分布式协调、智能负载均衡、高效通信优化和完善的容错机制，为Mirage系统提供了强大的可扩展性和可靠性，确保了大规模LLM推理的高效执行。