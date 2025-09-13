# 运行时系统设计

## 🎯 设计目标

运行时系统是Mirage Persistent Kernel的执行引擎，负责：
- **持久化内核管理**：管理长期运行的GPU内核生命周期
- **动态任务调度**：在GPU内部实现高效的任务调度
- **内存管理**：智能的内存分配、复用和回收
- **多GPU协调**：处理分布式推理中的GPU间协作

## 🏗️ 运行时架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Runtime System                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │PersistentKernel │  │  TaskScheduler   │  │MemoryMgr   │ │
│  │   (持久内核)     │  │  (任务调度器)    │  │ (内存管理)   │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │   WorkerPool    │  │   EventSystem    │  │ProfilerSys  │ │
│  │   (工作池)       │  │   (事件系统)     │  │ (性能分析)   │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │   Synchronizer  │  │   Communication  │  │ErrorHandler │ │
│  │   (同步管理)     │  │   (通信管理)     │  │ (错误处理)   │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 持久化内核系统

### 核心设计理念
持久化内核打破了传统GPU编程中"启动内核→执行→结束"的模式，改为"启动一次→持续执行任务→按需结束"的模式。

### 持久化内核结构
```cpp
namespace mirage::runtime {

// 运行时配置
struct RuntimeConfig {
    // 基本配置
    int my_mpi_rank;                    // MPI rank
    int num_workers;                    // worker线程数
    int num_local_schedulers;           // 本地调度器数
    int num_remote_schedulers;          // 远程调度器数
    int max_seq_length;                 // 最大序列长度
    long long eos_token_id;             // 结束token ID
    
    // 任务队列
    TaskId* worker_queue_last_ready_task_id; // 工作队列最后就绪任务ID
    TaskId** worker_queues;             // 工作队列数组
    
    // 事件系统
    EventId* event_queue_last_ready_event_id; // 事件队列最后就绪事件ID
    EventId** event_queues;             // 事件队列数组
    
    // 元数据
    int* step;                          // 当前步数
    long long* tokens;                  // token数组
    int* new_token_nums;                // 新token数量
    
    // 性能分析
    void* profiler_buffer;              // 性能分析缓冲区
};

// 任务描述符
struct TaskDesc {
    TaskType task_type;                 // 任务类型
    int num_inputs;                     // 输入数量
    int num_outputs;                    // 输出数量
    TensorDesc inputs[MAX_INPUTS_PER_TASK];   // 输入张量
    TensorDesc outputs[MAX_OUTPUTS_PER_TASK]; // 输出张量
    void* additional_params;            // 额外参数
    
    // 调度信息
    dim3 grid_dim;                      // 网格维度
    dim3 block_dim;                     // 线程块维度
    int shared_memory_size;             // 共享内存大小
};

// 事件描述符
struct EventDesc {
    EventType event_type;               // 事件类型
    int num_triggers;                   // 触发器数量
    TaskId first_task_id;               // 第一个任务ID
    TaskId last_task_id;                // 最后一个任务ID
    
    EventDesc() : event_type(EVENT_INVALID), num_triggers(0),
                 first_task_id(TASK_INVALID_ID), last_task_id(TASK_INVALID_ID) {}
};

}
```

### 持久化内核主循环
```cpp
__global__ void persistent_kernel(RuntimeConfig config) {
    // 线程角色分配
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = thread_id / 32;
    int lane_id = thread_id % 32;
    
    // 角色判断
    bool is_worker = (warp_id < config.num_workers);
    bool is_local_scheduler = (warp_id >= config.num_workers && 
                              warp_id < config.num_workers + config.num_local_schedulers);
    bool is_remote_scheduler = (warp_id >= config.num_workers + config.num_local_schedulers);
    
    if (is_worker) {
        worker_main_loop(config, warp_id);
    } else if (is_local_scheduler) {
        local_scheduler_main_loop(config, warp_id - config.num_workers);
    } else if (is_remote_scheduler) {
        remote_scheduler_main_loop(config, warp_id - config.num_workers - config.num_local_schedulers);
    }
}

// Worker主循环
__device__ void worker_main_loop(RuntimeConfig const& config, int worker_id) {
    while (true) {
        // 1. 获取下一个任务
        TaskId task_id = get_next_task(config, worker_id);
        
        if (task_id == TASK_INVALID_ID) {
            // 没有任务，检查是否应该终止
            if (should_terminate(config)) {
                break;
            }
            continue;
        }
        
        // 2. 获取任务描述符
        TaskDesc task_desc = get_task_descriptor(task_id);
        
        // 3. 执行任务
        execute_task(task_desc, config);
        
        // 4. 更新依赖关系
        update_task_dependencies(task_id, config);
    }
}
```

## ⚡ 动态任务调度系统

### 调度架构
```cpp
class TaskScheduler {
private:
    // 调度队列
    struct SchedulingQueue {
        std::atomic<int> head;
        std::atomic<int> tail;
        TaskId* tasks;
        int capacity;
        
        bool enqueue(TaskId task_id) {
            int current_tail = tail.load();
            int next_tail = (current_tail + 1) % capacity;
            if (next_tail == head.load()) {
                return false;  // 队列满
            }
            tasks[current_tail] = task_id;
            tail.store(next_tail);
            return true;
        }
        
        TaskId dequeue() {
            int current_head = head.load();
            if (current_head == tail.load()) {
                return TASK_INVALID_ID;  // 队列空
            }
            TaskId task = tasks[current_head];
            head.store((current_head + 1) % capacity);
            return task;
        }
    };
    
    std::vector<SchedulingQueue> worker_queues;
    std::vector<SchedulingQueue> priority_queues;
    
public:
    // 任务调度策略
    enum class SchedulingPolicy {
        ROUND_ROBIN,        // 轮询调度
        WORK_STEALING,      // 工作窃取
        PRIORITY_BASED,     // 优先级调度
        LOCALITY_AWARE      // 局部性感知
    };
    
    TaskId schedule_next_task(int worker_id, SchedulingPolicy policy) {
        switch (policy) {
            case SchedulingPolicy::ROUND_ROBIN:
                return round_robin_schedule(worker_id);
            case SchedulingPolicy::WORK_STEALING:
                return work_stealing_schedule(worker_id);
            case SchedulingPolicy::PRIORITY_BASED:
                return priority_based_schedule(worker_id);
            case SchedulingPolicy::LOCALITY_AWARE:
                return locality_aware_schedule(worker_id);
        }
        return TASK_INVALID_ID;
    }
    
private:
    TaskId work_stealing_schedule(int worker_id) {
        // 1. 先从自己的队列获取任务
        TaskId task = worker_queues[worker_id].dequeue();
        if (task != TASK_INVALID_ID) {
            return task;
        }
        
        // 2. 从其他worker偷取任务
        for (int i = 1; i < worker_queues.size(); ++i) {
            int target_worker = (worker_id + i) % worker_queues.size();
            task = worker_queues[target_worker].dequeue();
            if (task != TASK_INVALID_ID) {
                return task;
            }
        }
        
        return TASK_INVALID_ID;
    }
    
    TaskId priority_based_schedule(int worker_id) {
        // 先检查高优先级队列
        for (auto& pq : priority_queues) {
            TaskId task = pq.dequeue();
            if (task != TASK_INVALID_ID) {
                return task;
            }
        }
        
        // 再检查普通队列
        return worker_queues[worker_id].dequeue();
    }
};
```

### 依赖管理
```cpp
class DependencyManager {
private:
    struct TaskDependency {
        TaskId task_id;
        std::vector<TaskId> predecessors;   // 前驱任务
        std::vector<TaskId> successors;     // 后继任务
        std::atomic<int> pending_count;     // 待完成前驱数量
    };
    
    std::unordered_map<TaskId, TaskDependency> dependencies;
    
public:
    void add_dependency(TaskId task, TaskId predecessor) {
        dependencies[task].predecessors.push_back(predecessor);
        dependencies[predecessor].successors.push_back(task);
        dependencies[task].pending_count++;
    }
    
    std::vector<TaskId> task_completed(TaskId completed_task) {
        std::vector<TaskId> ready_tasks;
        
        auto& dep = dependencies[completed_task];
        for (TaskId successor : dep.successors) {
            int remaining = dependencies[successor].pending_count.fetch_sub(1) - 1;
            if (remaining == 0) {
                ready_tasks.push_back(successor);
            }
        }
        
        return ready_tasks;
    }
};
```

## 🧠 内存管理系统

### 内存池管理
```cpp
class DeviceMemoryManager {
private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool is_free;
        MemoryBlock* next;
        MemoryBlock* prev;
    };
    
    struct MemoryPool {
        MemoryBlock* free_list_head;
        MemoryBlock* allocated_list_head;
        size_t total_size;
        size_t free_size;
        std::mutex mutex;
    };
    
    std::vector<MemoryPool> pools;
    
public:
    void* allocate(size_t size, int alignment = 256) {
        // 1. 找到合适的内存池
        int pool_id = find_suitable_pool(size);
        
        // 2. 在内存池中分配
        auto& pool = pools[pool_id];
        std::lock_guard<std::mutex> lock(pool.mutex);
        
        MemoryBlock* block = find_free_block(pool, size, alignment);
        if (block == nullptr) {
            // 尝试扩展内存池
            if (!expand_pool(pool, size * 2)) {
                return nullptr;
            }
            block = find_free_block(pool, size, alignment);
        }
        
        if (block != nullptr) {
            split_block_if_needed(block, size);
            mark_block_allocated(block);
            pool.free_size -= block->size;
            return block->ptr;
        }
        
        return nullptr;
    }
    
    void deallocate(void* ptr) {
        if (ptr == nullptr) return;
        
        // 1. 找到对应的内存块
        MemoryBlock* block = find_block_by_ptr(ptr);
        if (block == nullptr) return;
        
        // 2. 标记为空闲
        mark_block_free(block);
        
        // 3. 尝试合并相邻的空闲块
        coalesce_free_blocks(block);
    }
    
private:
    MemoryBlock* find_free_block(MemoryPool& pool, size_t size, int alignment) {
        MemoryBlock* current = pool.free_list_head;
        while (current != nullptr) {
            if (current->is_free && current->size >= size) {
                // 检查对齐
                void* aligned_ptr = align_pointer(current->ptr, alignment);
                size_t aligned_size = size + ((char*)aligned_ptr - (char*)current->ptr);
                if (current->size >= aligned_size) {
                    return current;
                }
            }
            current = current->next;
        }
        return nullptr;
    }
    
    void coalesce_free_blocks(MemoryBlock* block) {
        // 向前合并
        while (block->prev != nullptr && block->prev->is_free) {
            MemoryBlock* prev = block->prev;
            prev->size += block->size;
            prev->next = block->next;
            if (block->next) {
                block->next->prev = prev;
            }
            delete block;
            block = prev;
        }
        
        // 向后合并
        while (block->next != nullptr && block->next->is_free) {
            MemoryBlock* next = block->next;
            block->size += next->size;
            block->next = next->next;
            if (next->next) {
                next->next->prev = block;
            }
            delete next;
        }
    }
};
```

### 张量生命周期管理
```cpp
class TensorLifecycleManager {
private:
    struct TensorInfo {
        size_t tensor_id;
        void* device_ptr;
        size_t size;
        int ref_count;
        int creation_time;
        int last_access_time;
        bool is_persistent;  // 是否为持久张量
    };
    
    std::unordered_map<size_t, TensorInfo> tensor_registry;
    std::vector<size_t> gc_candidates;  // 垃圾回收候选
    
public:
    void register_tensor(size_t tensor_id, void* ptr, size_t size, bool persistent = false) {
        TensorInfo info;
        info.tensor_id = tensor_id;
        info.device_ptr = ptr;
        info.size = size;
        info.ref_count = 1;
        info.creation_time = get_current_time();
        info.last_access_time = info.creation_time;
        info.is_persistent = persistent;
        
        tensor_registry[tensor_id] = info;
    }
    
    void access_tensor(size_t tensor_id) {
        auto it = tensor_registry.find(tensor_id);
        if (it != tensor_registry.end()) {
            it->second.last_access_time = get_current_time();
        }
    }
    
    void release_tensor(size_t tensor_id) {
        auto it = tensor_registry.find(tensor_id);
        if (it != tensor_registry.end()) {
            it->second.ref_count--;
            if (it->second.ref_count == 0 && !it->second.is_persistent) {
                gc_candidates.push_back(tensor_id);
            }
        }
    }
    
    void garbage_collect() {
        int current_time = get_current_time();
        
        for (size_t tensor_id : gc_candidates) {
            auto it = tensor_registry.find(tensor_id);
            if (it != tensor_registry.end()) {
                // 检查是否可以回收
                if (current_time - it->second.last_access_time > GC_THRESHOLD) {
                    device_memory_manager.deallocate(it->second.device_ptr);
                    tensor_registry.erase(it);
                }
            }
        }
        
        gc_candidates.clear();
    }
};
```

## 🌐 多GPU通信系统

### NVSHMEM集成
```cpp
class NVSHMEMManager {
private:
    int num_pes;        // 参与进程数
    int my_pe;          // 当前进程ID
    bool is_initialized;
    
public:
    bool initialize(int world_size, int rank) {
        if (is_initialized) return true;
        
        // 1. 初始化MPI
        MPI_Init(nullptr, nullptr);
        MPI_Comm_size(MPI_COMM_WORLD, &num_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_pe);
        
        // 2. 初始化NVSHMEM
        nvshmemx_init_attr_t attr;
        attr.mpi_comm = &MPI_COMM_WORLD;
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
        
        is_initialized = true;
        return true;
    }
    
    void* symmetric_malloc(size_t size) {
        return nvshmem_malloc(size);
    }
    
    void symmetric_free(void* ptr) {
        nvshmem_free(ptr);
    }
    
    // 点对点通信
    void put(void* dest, const void* source, size_t nelems, int pe) {
        nvshmem_float_put((float*)dest, (const float*)source, nelems, pe);
    }
    
    void get(void* dest, const void* source, size_t nelems, int pe) {
        nvshmem_float_get((float*)dest, (const float*)source, nelems, pe);
    }
    
    // 集合通信
    void allreduce(void* dest, const void* source, size_t nelems) {
        nvshmem_float_sum_reduce(NVSHMEM_TEAM_WORLD, (float*)dest, 
                                (const float*)source, nelems);
    }
    
    void barrier() {
        nvshmem_barrier_all();
    }
};
```

### 分布式任务协调
```cpp
class DistributedCoordinator {
private:
    NVSHMEMManager nvshmem;
    std::vector<TaskQueue> remote_task_queues;
    
public:
    void coordinate_distributed_execution(const std::vector<TaskGraph>& task_graphs) {
        // 1. 分发任务图到各个GPU
        distribute_task_graphs(task_graphs);
        
        // 2. 同步启动执行
        synchronized_start();
        
        // 3. 监控执行进度
        monitor_execution_progress();
        
        // 4. 处理跨GPU依赖
        handle_cross_gpu_dependencies();
    }
    
private:
    void distribute_task_graphs(const std::vector<TaskGraph>& task_graphs) {
        for (int gpu_id = 0; gpu_id < task_graphs.size(); ++gpu_id) {
            if (gpu_id == nvshmem.get_my_pe()) {
                // 本地GPU，直接设置
                local_task_graph = task_graphs[gpu_id];
            } else {
                // 远程GPU，通过NVSHMEM发送
                send_task_graph_to_gpu(task_graphs[gpu_id], gpu_id);
            }
        }
    }
    
    void handle_cross_gpu_dependencies() {
        while (has_pending_cross_gpu_tasks()) {
            // 检查远程完成的任务
            auto completed_remote_tasks = check_remote_task_completion();
            
            // 激活依赖这些任务的本地任务
            for (TaskId remote_task : completed_remote_tasks) {
                activate_dependent_local_tasks(remote_task);
            }
            
            // 通知其他GPU本地完成的任务
            auto completed_local_tasks = get_completed_local_tasks();
            notify_task_completion_to_remotes(completed_local_tasks);
        }
    }
};
```

## 📊 性能分析系统

### 内核级性能分析
```cpp
class PersistentKernelProfiler {
private:
    struct ProfileData {
        TaskType task_type;
        uint64_t start_time;
        uint64_t end_time;
        int worker_id;
        size_t shared_memory_used;
        int register_count;
    };
    
    ProfileData* device_profile_buffer;
    size_t buffer_size;
    std::atomic<size_t> buffer_index;
    
public:
    __device__ void record_task_start(TaskType type, int worker_id) {
        size_t index = atomicAdd(&buffer_index, 1);
        if (index < buffer_size) {
            ProfileData& data = device_profile_buffer[index];
            data.task_type = type;
            data.start_time = clock64();
            data.worker_id = worker_id;
        }
    }
    
    __device__ void record_task_end(TaskType type, int worker_id) {
        // 找到对应的开始记录
        for (size_t i = buffer_index.load() - 1; i >= 0; --i) {
            ProfileData& data = device_profile_buffer[i];
            if (data.task_type == type && data.worker_id == worker_id && data.end_time == 0) {
                data.end_time = clock64();
                break;
            }
        }
    }
    
    void analyze_performance() {
        // 拷贝数据到主机
        std::vector<ProfileData> host_data(buffer_index.load());
        cudaMemcpy(host_data.data(), device_profile_buffer, 
                  host_data.size() * sizeof(ProfileData), cudaMemcpyDeviceToHost);
        
        // 分析性能数据
        analyze_task_distribution(host_data);
        analyze_worker_utilization(host_data);
        analyze_memory_usage(host_data);
        generate_timeline_visualization(host_data);
    }
};
```

### 实时监控
```cpp
class RuntimeMonitor {
private:
    struct SystemMetrics {
        float gpu_utilization;
        float memory_utilization;
        int active_workers;
        int pending_tasks;
        float average_task_latency;
    };
    
    SystemMetrics current_metrics;
    std::vector<SystemMetrics> metrics_history;
    
public:
    void update_metrics() {
        current_metrics.gpu_utilization = get_gpu_utilization();
        current_metrics.memory_utilization = get_memory_utilization();
        current_metrics.active_workers = get_active_worker_count();
        current_metrics.pending_tasks = get_pending_task_count();
        current_metrics.average_task_latency = calculate_average_task_latency();
        
        metrics_history.push_back(current_metrics);
        
        // 保持历史记录在合理范围内
        if (metrics_history.size() > MAX_HISTORY_SIZE) {
            metrics_history.erase(metrics_history.begin());
        }
    }
    
    bool detect_performance_anomaly() {
        if (metrics_history.size() < 10) return false;
        
        // 检查GPU利用率异常
        if (current_metrics.gpu_utilization < 0.5) {
            return true;  // GPU利用率过低
        }
        
        // 检查任务延迟异常
        float recent_avg_latency = 0;
        for (int i = metrics_history.size() - 5; i < metrics_history.size(); ++i) {
            recent_avg_latency += metrics_history[i].average_task_latency;
        }
        recent_avg_latency /= 5;
        
        if (recent_avg_latency > current_metrics.average_task_latency * 1.5) {
            return true;  // 延迟异常增加
        }
        
        return false;
    }
};
```

## 🛡️ 错误处理和恢复

### 错误检测
```cpp
class ErrorHandler {
private:
    enum class ErrorType {
        CUDA_ERROR,
        MEMORY_ERROR,
        COMMUNICATION_ERROR,
        TASK_TIMEOUT,
        SYNCHRONIZATION_ERROR
    };
    
    struct ErrorInfo {
        ErrorType type;
        int error_code;
        std::string message;
        uint64_t timestamp;
        int worker_id;
    };
    
    std::vector<ErrorInfo> error_log;
    
public:
    bool handle_cuda_error(cudaError_t error, const std::string& context) {
        if (error != cudaSuccess) {
            ErrorInfo info;
            info.type = ErrorType::CUDA_ERROR;
            info.error_code = static_cast<int>(error);
            info.message = std::string(cudaGetErrorString(error)) + " in " + context;
            info.timestamp = get_current_timestamp();
            
            error_log.push_back(info);
            
            // 尝试恢复
            return attempt_cuda_recovery(error);
        }
        return true;
    }
    
    bool handle_task_timeout(TaskId task_id, int worker_id) {
        ErrorInfo info;
        info.type = ErrorType::TASK_TIMEOUT;
        info.message = "Task " + std::to_string(task_id) + " timeout on worker " + std::to_string(worker_id);
        info.timestamp = get_current_timestamp();
        info.worker_id = worker_id;
        
        error_log.push_back(info);
        
        // 重新调度任务
        return reschedule_task(task_id);
    }
    
private:
    bool attempt_cuda_recovery(cudaError_t error) {
        switch (error) {
            case cudaErrorMemoryAllocation:
                // 尝试释放一些缓存内存
                return attempt_memory_recovery();
            case cudaErrorLaunchOutOfResources:
                // 减少线程块大小重试
                return attempt_resource_recovery();
            default:
                // 重置CUDA上下文
                return attempt_context_recovery();
        }
    }
};
```

运行时系统通过持久化内核、动态调度、智能内存管理和完善的错误处理，为Mirage提供了稳定高效的执行环境，确保了系统的高性能和可靠性。