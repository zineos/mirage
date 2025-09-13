# 性能优化机制

## 🎯 优化目标

Mirage Persistent Kernel的性能优化贯穿整个系统的各个层次，主要目标是：
- **延迟最小化**：将LLM推理延迟降低1.2×到6.7×
- **吞吐量最大化**：提高GPU计算资源利用率
- **内存效率**：优化内存访问模式和数据布局
- **能耗优化**：在保证性能的同时降低能耗

## 🏗️ 多层次优化架构

```
┌─────────────────────────────────────────────────────────────┐
│                Performance Optimization                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │  Compile-Time   │  │   Runtime        │  │ Hardware    │ │
│  │   Optimization  │  │  Optimization    │  │Optimization │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │  Graph-Level    │  │  Kernel-Level    │  │Memory-Level │ │
│  │  Optimization   │  │  Optimization    │  │Optimization │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │Communication    │  │  Synchronization │  │   Energy    │ │
│  │  Optimization   │  │   Optimization   │  │Optimization │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## ⚡ 编译时优化

### 图级优化
```cpp
namespace mirage::optimization {

class GraphOptimizer {
private:
    struct OptimizationPass {
        std::string name;
        std::function<bool(kernel::Graph&)> optimize;
        int priority;  // 优化优先级
        bool is_mandatory;
    };
    
    std::vector<OptimizationPass> optimization_passes;
    
public:
    GraphOptimizer() {
        register_optimization_passes();
    }
    
    void optimize_graph(kernel::Graph& graph) {
        // 按优先级排序优化pass
        std::sort(optimization_passes.begin(), optimization_passes.end(),
                 [](const OptimizationPass& a, const OptimizationPass& b) {
                     return a.priority > b.priority;
                 });
        
        bool graph_changed = true;
        int max_iterations = 10;  // 防止无限循环
        int iteration = 0;
        
        while (graph_changed && iteration < max_iterations) {
            graph_changed = false;
            iteration++;
            
            for (auto& pass : optimization_passes) {
                if (pass.optimize(graph)) {
                    graph_changed = true;
                    LOG_INFO("Applied optimization pass: " + pass.name);
                }
            }
        }
    }
    
private:
    void register_optimization_passes() {
        // 操作融合优化
        optimization_passes.push_back({
            "VerticalFusion",
            [this](kernel::Graph& g) { return apply_vertical_fusion(g); },
            100, false
        });
        
        // 内存布局优化
        optimization_passes.push_back({
            "LayoutOptimization", 
            [this](kernel::Graph& g) { return optimize_tensor_layouts(g); },
            90, false
        });
        
        // 冗余消除
        optimization_passes.push_back({
            "RedundancyElimination",
            [this](kernel::Graph& g) { return eliminate_redundancy(g); },
            80, false
        });
        
        // 常数折叠
        optimization_passes.push_back({
            "ConstantFolding",
            [this](kernel::Graph& g) { return fold_constants(g); },
            70, false
        });
    }
    
    bool apply_vertical_fusion(kernel::Graph& graph) {
        bool changed = false;
        auto operators = graph.get_operators();
        
        for (size_t i = 0; i < operators.size() - 1; ++i) {
            auto* current_op = operators[i];
            auto* next_op = operators[i + 1];
            
            if (can_fuse_vertically(current_op, next_op)) {
                auto fused_op = create_fused_operator(current_op, next_op);
                graph.replace_operators({current_op, next_op}, fused_op);
                changed = true;
                break;  // 重新开始，因为图结构已改变
            }
        }
        
        return changed;
    }
    
    bool can_fuse_vertically(kernel::KNOperator* op1, kernel::KNOperator* op2) {
        // 检查数据依赖
        for (auto& output : op1->output_tensors) {
            for (auto& input : op2->input_tensors) {
                if (output.guid == input.guid) {
                    // 有直接依赖，检查融合约束
                    return check_fusion_constraints(op1, op2);
                }
            }
        }
        return false;
    }
    
    bool check_fusion_constraints(kernel::KNOperator* op1, kernel::KNOperator* op2) {
        // 1. 检查内存约束
        if (estimate_memory_usage(op1) + estimate_memory_usage(op2) > MAX_SHARED_MEMORY) {
            return false;
        }
        
        // 2. 检查计算强度匹配
        float compute_intensity1 = estimate_compute_intensity(op1);
        float compute_intensity2 = estimate_compute_intensity(op2);
        if (std::abs(compute_intensity1 - compute_intensity2) > 0.5f) {
            return false;  // 计算强度差异过大
        }
        
        // 3. 检查数据重用机会
        if (!has_data_reuse_opportunity(op1, op2)) {
            return false;
        }
        
        return true;
    }
};

}
```

### 内存布局优化
```cpp
class MemoryLayoutOptimizer {
private:
    struct LayoutCost {
        layout::DmemLayout layout;
        float access_cost;        // 访问成本
        float storage_cost;       // 存储成本
        float conversion_cost;    // 转换成本
    };
    
public:
    void optimize_tensor_layouts(kernel::Graph& graph) {
        auto tensors = graph.get_all_tensors();
        
        // 构建张量访问图
        auto access_graph = build_tensor_access_graph(tensors);
        
        // 为每个张量选择最优布局
        for (auto& tensor : tensors) {
            auto optimal_layout = find_optimal_layout(tensor, access_graph);
            if (optimal_layout != tensor.layout) {
                update_tensor_layout(tensor, optimal_layout, graph);
            }
        }
    }
    
private:
    layout::DmemLayout find_optimal_layout(
        const kernel::DTensor& tensor,
        const TensorAccessGraph& access_graph) {
        
        std::vector<LayoutCost> layout_costs;
        
        // 评估不同布局的成本
        for (auto layout : {layout::DmemLayout::ROW_MAJOR, 
                           layout::DmemLayout::COLUMN_MAJOR}) {
            LayoutCost cost;
            cost.layout = layout;
            cost.access_cost = calculate_access_cost(tensor, layout, access_graph);
            cost.storage_cost = calculate_storage_cost(tensor, layout);
            cost.conversion_cost = calculate_conversion_cost(tensor, layout, access_graph);
            
            layout_costs.push_back(cost);
        }
        
        // 选择总成本最低的布局
        auto best_layout = std::min_element(layout_costs.begin(), layout_costs.end(),
            [](const LayoutCost& a, const LayoutCost& b) {
                float total_cost_a = a.access_cost + a.storage_cost + a.conversion_cost;
                float total_cost_b = b.access_cost + b.storage_cost + b.conversion_cost;
                return total_cost_a < total_cost_b;
            });
        
        return best_layout->layout;
    }
    
    float calculate_access_cost(const kernel::DTensor& tensor,
                               layout::DmemLayout layout,
                               const TensorAccessGraph& access_graph) {
        float total_cost = 0.0f;
        
        for (auto& access : access_graph.get_accesses(tensor.guid)) {
            // 根据访问模式计算成本
            switch (access.pattern) {
                case AccessPattern::SEQUENTIAL:
                    total_cost += (layout == layout::DmemLayout::ROW_MAJOR) ? 1.0f : 2.0f;
                    break;
                case AccessPattern::STRIDED:
                    total_cost += 3.0f;  // 跨步访问成本较高
                    break;
                case AccessPattern::RANDOM:
                    total_cost += 5.0f;  // 随机访问成本最高
                    break;
            }
            
            total_cost *= access.frequency;  // 乘以访问频率
        }
        
        return total_cost;
    }
};
```

## 🚀 运行时优化

### 动态负载均衡
```cpp
class DynamicLoadBalancer {
private:
    struct WorkerStats {
        int worker_id;
        int completed_tasks;
        float average_task_time;
        float current_load;
        std::chrono::steady_clock::time_point last_update;
    };
    
    std::vector<WorkerStats> worker_stats;
    std::mutex stats_mutex;
    
public:
    void update_worker_stats(int worker_id, float task_execution_time) {
        std::lock_guard<std::mutex> lock(stats_mutex);
        
        auto& stats = worker_stats[worker_id];
        stats.completed_tasks++;
        
        // 使用指数移动平均更新平均任务时间
        float alpha = 0.1f;  // 平滑因子
        stats.average_task_time = alpha * task_execution_time + 
                                 (1.0f - alpha) * stats.average_task_time;
        
        // 更新当前负载
        auto now = std::chrono::steady_clock::now();
        float time_since_last = std::chrono::duration<float>(now - stats.last_update).count();
        stats.current_load = std::max(0.0f, stats.current_load - time_since_last);
        stats.current_load += task_execution_time;
        stats.last_update = now;
    }
    
    int select_optimal_worker(const TaskDesc& task) {
        std::lock_guard<std::mutex> lock(stats_mutex);
        
        float estimated_task_time = estimate_task_execution_time(task);
        
        int best_worker = 0;
        float best_completion_time = INFINITY;
        
        for (size_t i = 0; i < worker_stats.size(); ++i) {
            float completion_time = worker_stats[i].current_load + estimated_task_time;
            
            if (completion_time < best_completion_time) {
                best_completion_time = completion_time;
                best_worker = static_cast<int>(i);
            }
        }
        
        return best_worker;
    }
    
    void rebalance_workload() {
        std::lock_guard<std::mutex> lock(stats_mutex);
        
        // 计算负载方差
        float mean_load = 0.0f;
        for (auto& stats : worker_stats) {
            mean_load += stats.current_load;
        }
        mean_load /= worker_stats.size();
        
        float load_variance = 0.0f;
        for (auto& stats : worker_stats) {
            float diff = stats.current_load - mean_load;
            load_variance += diff * diff;
        }
        load_variance /= worker_stats.size();
        
        // 如果负载不均衡，触发工作窃取
        if (load_variance > LOAD_IMBALANCE_THRESHOLD) {
            trigger_work_stealing();
        }
    }
    
private:
    void trigger_work_stealing() {
        // 找到负载最高和最低的worker
        auto max_load_worker = std::max_element(worker_stats.begin(), worker_stats.end(),
            [](const WorkerStats& a, const WorkerStats& b) {
                return a.current_load < b.current_load;
            });
        
        auto min_load_worker = std::min_element(worker_stats.begin(), worker_stats.end(),
            [](const WorkerStats& a, const WorkerStats& b) {
                return a.current_load < b.current_load;
            });
        
        // 如果负载差异足够大，执行工作窃取
        if (max_load_worker->current_load - min_load_worker->current_load > STEAL_THRESHOLD) {
            steal_work(max_load_worker->worker_id, min_load_worker->worker_id);
        }
    }
};
```

### 内存预取优化
```cpp
class MemoryPrefetcher {
private:
    struct PrefetchPattern {
        void* base_address;
        size_t stride;
        int prefetch_distance;
        AccessPattern pattern_type;
    };
    
    std::unordered_map<void*, PrefetchPattern> learned_patterns;
    
public:
    void learn_access_pattern(void* address, size_t size, AccessPattern pattern) {
        PrefetchPattern prefetch_pattern;
        prefetch_pattern.base_address = address;
        prefetch_pattern.stride = detect_stride(address, pattern);
        prefetch_pattern.prefetch_distance = calculate_optimal_distance(size, pattern);
        prefetch_pattern.pattern_type = pattern;
        
        learned_patterns[address] = prefetch_pattern;
    }
    
    void prefetch_next_access(void* current_address) {
        auto it = learned_patterns.find(current_address);
        if (it != learned_patterns.end()) {
            const auto& pattern = it->second;
            
            // 计算下一个访问地址
            void* next_address = static_cast<char*>(current_address) + 
                                pattern.stride * pattern.prefetch_distance;
            
            // 发起预取
            __builtin_prefetch(next_address, 0, 3);  // 预取到L1缓存
        }
    }
    
private:
    size_t detect_stride(void* address, AccessPattern pattern) {
        switch (pattern) {
            case AccessPattern::SEQUENTIAL:
                return sizeof(float);  // 假设float类型
            case AccessPattern::STRIDED:
                return estimate_stride_from_history(address);
            default:
                return 0;
        }
    }
    
    int calculate_optimal_distance(size_t data_size, AccessPattern pattern) {
        // 基于内存延迟和带宽计算最优预取距离
        float memory_latency_cycles = 300.0f;  // 典型内存延迟
        float compute_cycles_per_element = 10.0f;  // 每个元素的计算周期
        
        int optimal_distance = static_cast<int>(
            memory_latency_cycles / compute_cycles_per_element);
        
        // 限制在合理范围内
        return std::clamp(optimal_distance, 1, 16);
    }
};
```

## 🧠 内存优化

### 缓存友好的数据布局
```cpp
class CacheOptimizer {
private:
    static constexpr int L1_CACHE_SIZE = 32 * 1024;      // 32KB L1缓存
    static constexpr int L2_CACHE_SIZE = 1024 * 1024;    // 1MB L2缓存
    static constexpr int CACHE_LINE_SIZE = 128;           // 128字节缓存行
    
public:
    void optimize_data_layout(std::vector<kernel::DTensor>& tensors) {
        // 1. 分析张量访问模式
        auto access_patterns = analyze_access_patterns(tensors);
        
        // 2. 对张量进行分组
        auto tensor_groups = group_tensors_by_locality(tensors, access_patterns);
        
        // 3. 为每组选择最优布局
        for (auto& group : tensor_groups) {
            optimize_group_layout(group);
        }
    }
    
private:
    void optimize_group_layout(std::vector<kernel::DTensor*>& tensor_group) {
        // 计算组的总大小
        size_t total_size = 0;
        for (auto* tensor : tensor_group) {
            total_size += tensor->size_in_bytes();
        }
        
        if (total_size <= L1_CACHE_SIZE) {
            // 适合L1缓存，使用紧凑布局
            apply_compact_layout(tensor_group);
        } else if (total_size <= L2_CACHE_SIZE) {
            // 适合L2缓存，使用分块布局
            apply_tiled_layout(tensor_group);
        } else {
            // 超出L2缓存，使用流式布局
            apply_streaming_layout(tensor_group);
        }
    }
    
    void apply_compact_layout(std::vector<kernel::DTensor*>& tensors) {
        // 将相关张量在内存中紧密排列
        for (size_t i = 0; i < tensors.size(); ++i) {
            auto* tensor = tensors[i];
            
            // 确保对齐到缓存行边界
            size_t alignment = CACHE_LINE_SIZE;
            tensor->strides = calculate_aligned_strides(tensor->dims, alignment);
        }
    }
    
    void apply_tiled_layout(std::vector<kernel::DTensor*>& tensors) {
        // 使用分块布局优化缓存利用率
        constexpr int TILE_SIZE = 64;  // 64x64分块
        
        for (auto* tensor : tensors) {
            if (tensor->dims.size() >= 2) {
                // 对2D及以上张量应用分块
                tensor->layout = layout::DmemLayout::TILED;
                // 设置分块参数
                tensor->tile_size = {TILE_SIZE, TILE_SIZE};
            }
        }
    }
    
    std::vector<size_t> calculate_aligned_strides(
        const std::vector<int>& dims, size_t alignment) {
        
        std::vector<size_t> strides(dims.size());
        
        // 从最内层维度开始计算
        strides.back() = 1;
        for (int i = dims.size() - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * dims[i + 1];
            
            // 确保stride对齐
            strides[i] = (strides[i] + alignment - 1) / alignment * alignment;
        }
        
        return strides;
    }
};
```

### 内存池优化
```cpp
class OptimizedMemoryPool {
private:
    struct MemoryChunk {
        void* ptr;
        size_t size;
        bool is_free;
        std::chrono::steady_clock::time_point last_used;
        int access_frequency;
    };
    
    struct SizeClass {
        size_t size;
        std::vector<MemoryChunk> chunks;
        std::mutex mutex;
    };
    
    std::vector<SizeClass> size_classes;
    std::unordered_map<void*, size_t> ptr_to_size_class;
    
public:
    void* allocate(size_t size) {
        size_t size_class_idx = find_size_class(size);
        auto& size_class = size_classes[size_class_idx];
        
        std::lock_guard<std::mutex> lock(size_class.mutex);
        
        // 寻找空闲块
        for (auto& chunk : size_class.chunks) {
            if (chunk.is_free && chunk.size >= size) {
                chunk.is_free = false;
                chunk.last_used = std::chrono::steady_clock::now();
                chunk.access_frequency++;
                
                ptr_to_size_class[chunk.ptr] = size_class_idx;
                return chunk.ptr;
            }
        }
        
        // 没有合适的空闲块，分配新块
        void* new_ptr = cuda_malloc_with_optimization(size_class.size);
        if (new_ptr != nullptr) {
            MemoryChunk new_chunk;
            new_chunk.ptr = new_ptr;
            new_chunk.size = size_class.size;
            new_chunk.is_free = false;
            new_chunk.last_used = std::chrono::steady_clock::now();
            new_chunk.access_frequency = 1;
            
            size_class.chunks.push_back(new_chunk);
            ptr_to_size_class[new_ptr] = size_class_idx;
        }
        
        return new_ptr;
    }
    
    void deallocate(void* ptr) {
        auto it = ptr_to_size_class.find(ptr);
        if (it == ptr_to_size_class.end()) return;
        
        size_t size_class_idx = it->second;
        auto& size_class = size_classes[size_class_idx];
        
        std::lock_guard<std::mutex> lock(size_class.mutex);
        
        // 找到对应的chunk并标记为空闲
        for (auto& chunk : size_class.chunks) {
            if (chunk.ptr == ptr) {
                chunk.is_free = true;
                break;
            }
        }
        
        ptr_to_size_class.erase(it);
    }
    
    void optimize_memory_layout() {
        // 定期优化内存布局
        for (auto& size_class : size_classes) {
            std::lock_guard<std::mutex> lock(size_class.mutex);
            
            // 按访问频率排序
            std::sort(size_class.chunks.begin(), size_class.chunks.end(),
                     [](const MemoryChunk& a, const MemoryChunk& b) {
                         return a.access_frequency > b.access_frequency;
                     });
            
            // 回收长时间未使用的内存
            auto now = std::chrono::steady_clock::now();
            for (auto it = size_class.chunks.begin(); it != size_class.chunks.end();) {
                if (it->is_free) {
                    auto time_unused = std::chrono::duration<float>(now - it->last_used).count();
                    if (time_unused > MEMORY_RECLAIM_THRESHOLD) {
                        cudaFree(it->ptr);
                        it = size_class.chunks.erase(it);
                        continue;
                    }
                }
                ++it;
            }
        }
    }
    
private:
    void* cuda_malloc_with_optimization(size_t size) {
        void* ptr;
        
        // 尝试使用内存池分配
        cudaError_t err = cudaMallocFromPoolAsync(&ptr, size, memory_pool, stream);
        if (err == cudaSuccess) {
            return ptr;
        }
        
        // 回退到普通分配
        err = cudaMalloc(&ptr, size);
        return (err == cudaSuccess) ? ptr : nullptr;
    }
};
```

## 🌐 通信优化

### 高效AllReduce实现
```cpp
class OptimizedAllReduce {
private:
    enum class AllReduceAlgorithm {
        RING,
        TREE,
        BUTTERFLY,
        HIERARCHICAL
    };
    
public:
    void allreduce(void* data, size_t count, int num_gpus, int gpu_rank) {
        // 根据数据大小和GPU数量选择最优算法
        AllReduceAlgorithm algorithm = select_optimal_algorithm(count, num_gpus);
        
        switch (algorithm) {
            case AllReduceAlgorithm::RING:
                ring_allreduce(data, count, num_gpus, gpu_rank);
                break;
            case AllReduceAlgorithm::TREE:
                tree_allreduce(data, count, num_gpus, gpu_rank);
                break;
            case AllReduceAlgorithm::BUTTERFLY:
                butterfly_allreduce(data, count, num_gpus, gpu_rank);
                break;
            case AllReduceAlgorithm::HIERARCHICAL:
                hierarchical_allreduce(data, count, num_gpus, gpu_rank);
                break;
        }
    }
    
private:
    AllReduceAlgorithm select_optimal_algorithm(size_t count, int num_gpus) {
        size_t data_size = count * sizeof(float);
        
        // 小数据量：使用树形算法（延迟优化）
        if (data_size < 1024 * 1024) {  // 1MB
            return AllReduceAlgorithm::TREE;
        }
        
        // 中等数据量：使用环形算法（带宽优化）
        if (data_size < 64 * 1024 * 1024) {  // 64MB
            return AllReduceAlgorithm::RING;
        }
        
        // 大数据量：使用分层算法
        if (num_gpus > 8) {
            return AllReduceAlgorithm::HIERARCHICAL;
        }
        
        // 默认使用环形算法
        return AllReduceAlgorithm::RING;
    }
    
    void hierarchical_allreduce(void* data, size_t count, int num_gpus, int gpu_rank) {
        // 分层AllReduce：先节点内，再节点间
        int gpus_per_node = 8;  // 假设每个节点8个GPU
        int node_id = gpu_rank / gpus_per_node;
        int local_rank = gpu_rank % gpus_per_node;
        int num_nodes = (num_gpus + gpus_per_node - 1) / gpus_per_node;
        
        // 第一阶段：节点内AllReduce
        ring_allreduce_local(data, count, gpus_per_node, local_rank);
        
        // 第二阶段：节点间AllReduce (只有每个节点的rank 0参与)
        if (local_rank == 0) {
            ring_allreduce_inter_node(data, count, num_nodes, node_id);
        }
        
        // 第三阶段：节点内广播
        broadcast_within_node(data, count, gpus_per_node, local_rank);
    }
    
    void ring_allreduce_with_compression(void* data, size_t count, 
                                        int num_gpus, int gpu_rank) {
        // 使用压缩减少通信量
        size_t compressed_size;
        void* compressed_data = compress_gradient(data, count, &compressed_size);
        
        // 执行压缩数据的AllReduce
        ring_allreduce(compressed_data, compressed_size / sizeof(float), num_gpus, gpu_rank);
        
        // 解压缩
        decompress_gradient(compressed_data, data, count);
        
        free(compressed_data);
    }
};
```

## ⚙️ 硬件特定优化

### Tensor Core优化
```cpp
class TensorCoreOptimizer {
public:
    static bool can_use_tensor_core(const MatmulDesc& desc) {
        // 检查维度对齐
        if (desc.m % 16 != 0 || desc.n % 16 != 0 || desc.k % 16 != 0) {
            return false;
        }
        
        // 检查数据类型
        if (desc.input_type != DataType::FLOAT16 && 
            desc.input_type != DataType::BFLOAT16) {
            return false;
        }
        
        return true;
    }
    
    static void optimize_for_tensor_core(MatmulDesc& desc) {
        // 调整维度以适配Tensor Core
        desc.m = round_up_to_multiple(desc.m, 16);
        desc.n = round_up_to_multiple(desc.n, 16);
        desc.k = round_up_to_multiple(desc.k, 16);
        
        // 选择最优的Tensor Core配置
        if (desc.m >= 64 && desc.n >= 64) {
            desc.tile_size = {64, 64, 16};  // 大矩阵使用大分块
        } else {
            desc.tile_size = {32, 32, 16};  // 小矩阵使用小分块
        }
    }
    
private:
    static int round_up_to_multiple(int value, int multiple) {
        return (value + multiple - 1) / multiple * multiple;
    }
};
```

### 异步执行优化
```cpp
class AsyncExecutionOptimizer {
private:
    std::vector<cudaStream_t> compute_streams;
    std::vector<cudaStream_t> memory_streams;
    
public:
    void setup_async_execution(int num_streams = 4) {
        compute_streams.resize(num_streams);
        memory_streams.resize(num_streams);
        
        for (int i = 0; i < num_streams; ++i) {
            cudaStreamCreateWithPriority(&compute_streams[i], cudaStreamNonBlocking, 0);
            cudaStreamCreateWithPriority(&memory_streams[i], cudaStreamNonBlocking, -1);  // 高优先级
        }
    }
    
    void execute_with_overlap(const std::vector<KernelLaunch>& kernels) {
        int stream_idx = 0;
        
        for (const auto& kernel : kernels) {
            cudaStream_t compute_stream = compute_streams[stream_idx % compute_streams.size()];
            cudaStream_t memory_stream = memory_streams[stream_idx % memory_streams.size()];
            
            // 异步内存传输
            if (kernel.has_input_transfer) {
                cudaMemcpyAsync(kernel.device_input, kernel.host_input, 
                               kernel.input_size, cudaMemcpyHostToDevice, memory_stream);
            }
            
            // 等待内存传输完成
            cudaStreamWaitEvent(compute_stream, kernel.input_ready_event, 0);
            
            // 启动内核
            launch_kernel(kernel, compute_stream);
            
            // 异步输出传输
            if (kernel.has_output_transfer) {
                cudaEventRecord(kernel.compute_done_event, compute_stream);
                cudaStreamWaitEvent(memory_stream, kernel.compute_done_event, 0);
                cudaMemcpyAsync(kernel.host_output, kernel.device_output,
                               kernel.output_size, cudaMemcpyDeviceToHost, memory_stream);
            }
            
            stream_idx++;
        }
    }
};
```

## 📊 性能监控和调优

### 实时性能监控
```cpp
class PerformanceMonitor {
private:
    struct PerformanceMetrics {
        float kernel_utilization;     // 内核利用率
        float memory_bandwidth;       // 内存带宽利用率
        float tensor_core_efficiency; // Tensor Core效率
        float communication_overhead; // 通信开销
        float energy_efficiency;     // 能效比
    };
    
    PerformanceMetrics current_metrics;
    std::vector<PerformanceMetrics> metrics_history;
    
public:
    void collect_metrics() {
        current_metrics.kernel_utilization = measure_kernel_utilization();
        current_metrics.memory_bandwidth = measure_memory_bandwidth();
        current_metrics.tensor_core_efficiency = measure_tensor_core_efficiency();
        current_metrics.communication_overhead = measure_communication_overhead();
        current_metrics.energy_efficiency = measure_energy_efficiency();
        
        metrics_history.push_back(current_metrics);
        
        // 保持历史记录在合理大小
        if (metrics_history.size() > MAX_HISTORY_SIZE) {
            metrics_history.erase(metrics_history.begin());
        }
    }
    
    std::vector<OptimizationSuggestion> analyze_performance() {
        std::vector<OptimizationSuggestion> suggestions;
        
        // 分析内核利用率
        if (current_metrics.kernel_utilization < 0.8f) {
            suggestions.push_back({
                "Low kernel utilization",
                "Consider increasing batch size or using more aggressive fusion",
                OptimizationPriority::HIGH
            });
        }
        
        // 分析内存带宽
        if (current_metrics.memory_bandwidth < 0.6f) {
            suggestions.push_back({
                "Low memory bandwidth utilization", 
                "Optimize memory access patterns or increase data reuse",
                OptimizationPriority::MEDIUM
            });
        }
        
        // 分析Tensor Core效率
        if (current_metrics.tensor_core_efficiency < 0.7f) {
            suggestions.push_back({
                "Suboptimal Tensor Core usage",
                "Ensure matrix dimensions are multiples of 16 and use appropriate data types",
                OptimizationPriority::HIGH
            });
        }
        
        return suggestions;
    }
    
private:
    float measure_kernel_utilization() {
        // 使用CUDA profiling API测量内核利用率
        // 这里是简化实现
        return 0.85f;  // 示例值
    }
    
    float measure_tensor_core_efficiency() {
        // 测量Tensor Core的实际利用率
        return 0.75f;  // 示例值
    }
};
```

通过这些多层次的性能优化机制，Mirage系统能够在各个层面实现最优性能，确保LLM推理的高效执行。