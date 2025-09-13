# 转译器层设计

## 🎯 设计目标

转译器层是Mirage系统的代码生成核心，负责：
- **多后端支持**：支持CUDA、Triton、NKI等多种目标平台
- **高效代码生成**：生成高性能的GPU内核代码
- **平台优化**：针对不同硬件架构进行特定优化
- **内存管理**：智能的内存分配和布局优化

## 🏗️ 转译器架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Transpiler Layer                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │   Transpiler    │  │ NKITranspiler    │  │TritonTrans  │ │
│  │   (主转译器)     │  │ (NeuronCore转译) │  │ (Triton转译)│ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │  MemoryPlanner  │  │  LayoutOptimizer │  │FusionAnalyzer│ │
│  │  (内存规划器)    │  │  (布局优化器)    │  │ (融合分析器) │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │  CodeGenerator  │  │  ScheduleOpt     │  │ArchSpecific │ │
│  │  (代码生成器)    │  │  (调度优化)      │  │ (架构优化)   │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 主转译器 (Transpiler)

### 核心设计
```cpp
namespace mirage::transpiler {

class Transpiler {
private:
    // 核心数据
    std::shared_ptr<kernel::Graph> g;          // 内核图
    TranspilerConfig config;                   // 转译配置
    std::vector<std::vector<size_t>> input_strides;  // 输入步长
    std::vector<std::vector<size_t>> output_strides; // 输出步长
    
    // 分布式配置
    int num_gpus;
    bool use_nvshmem;
    
    // 张量元数据
    std::vector<kernel::DTensor> all_dtensors;
    std::unordered_map<size_t, DTensorMeta> dtensor_metas;
    std::unordered_map<size_t, STensorMeta> stensor_metas;
    
    // 融合元数据
    std::unordered_map<threadblock::TBOperator const *, bool> is_fused_with_prev;
    std::unordered_map<threadblock::TBOperator const *, bool> is_fused_with_next;
    std::unordered_map<threadblock::TBOperator const *, 
                      std::vector<threadblock::TBOperator const *>> fusion_chain;
    
    // 内存分配
    size_t d_buf_size;  // 中间张量缓冲区大小
    
public:
    Transpiler(std::shared_ptr<kernel::Graph> graph,
              const TranspilerConfig& cfg,
              const std::vector<std::vector<size_t>>& in_strides);
    
    // 主要转译方法
    TranspileResult transpile();
    
private:
    // 核心转译步骤
    void resolve_distributed_config();
    void resolve_dtensor_meta();
    void resolve_tensor_layout();
    void resolve_tb_fusion();
    void plan_dtensor_memory();
    void plan_stensor_memory();
    void get_threadblock_swizzle_plan();
    
    // 代码生成
    std::string generate_cuda_code();
    std::string generate_kernel_signature();
    std::string generate_kernel_body();
};

}
```

### 转译配置
```cpp
struct TranspilerConfig {
    // 目标平台
    enum class Target {
        CUDA,
        TRITON, 
        NKI,
        ROCM
    } target = Target::CUDA;
    
    // 硬件架构
    enum class Architecture {
        AMPERE,    // RTX 30xx, A100
        ADA,       // RTX 40xx
        HOPPER,    // H100
        BLACKWELL  // B100
    } arch = Architecture::HOPPER;
    
    // 优化选项
    bool enable_tensor_core = true;         // 启用Tensor Core
    bool enable_async_copy = true;          // 启用异步拷贝
    bool enable_warp_specialization = true; // 启用warp特化
    bool enable_pipeline = true;            // 启用流水线
    
    // 内存配置
    int max_shared_memory_size = 96 * 1024; // 最大共享内存
    bool prefer_l2_cache = true;            // 偏好L2缓存
    
    // 调试选项
    bool generate_debug_info = false;       // 生成调试信息
    bool enable_bounds_check = false;       // 启用边界检查
    int verbosity_level = 0;                // 详细程度
};
```

### 转译结果
```cpp
struct TranspileResult {
    std::string cuda_code;          // 生成的CUDA代码
    std::string json_metadata;      // 元数据JSON
    std::vector<IODesc> io_descs;   // IO描述符
    size_t shared_memory_size;      // 共享内存使用量
    int num_registers;              // 寄存器使用量
    
    // 性能预估
    float estimated_latency_us;     // 预估延迟(微秒)
    float estimated_throughput;     // 预估吞吐量
    int occupancy_percent;          // 占用率百分比
};
```

## 🧠 内存管理子系统

### DTensor内存规划
```cpp
class DTensorMemoryPlanner {
private:
    struct MemoryAllocation {
        size_t offset;              // 内存偏移
        size_t size;                // 分配大小
        kernel::DTensor* tensor;    // 关联张量
        int lifetime_start;         // 生命周期开始
        int lifetime_end;           // 生命周期结束
    };
    
    std::vector<MemoryAllocation> allocations;
    size_t total_memory_size;
    
public:
    void plan_memory_allocation(const std::vector<kernel::DTensor>& tensors) {
        // 1. 分析张量生命周期
        analyze_tensor_lifetimes(tensors);
        
        // 2. 使用图着色算法分配内存
        color_graph_allocation();
        
        // 3. 优化内存布局
        optimize_memory_layout();
    }
    
private:
    void analyze_tensor_lifetimes(const std::vector<kernel::DTensor>& tensors) {
        // 构建张量依赖图
        for (auto& tensor : tensors) {
            int start_time = find_tensor_creation_time(tensor);
            int end_time = find_tensor_last_use_time(tensor);
            
            MemoryAllocation alloc;
            alloc.tensor = &tensor;
            alloc.size = tensor.size_in_bytes();
            alloc.lifetime_start = start_time;
            alloc.lifetime_end = end_time;
            
            allocations.push_back(alloc);
        }
    }
    
    void color_graph_allocation() {
        // 使用图着色算法进行内存分配
        // 生命周期重叠的张量不能共享内存
        std::vector<std::vector<MemoryAllocation*>> color_groups;
        
        for (auto& alloc : allocations) {
            bool placed = false;
            
            for (auto& group : color_groups) {
                bool can_share = true;
                for (auto* other : group) {
                    if (lifetimes_overlap(alloc, *other)) {
                        can_share = false;
                        break;
                    }
                }
                
                if (can_share) {
                    group.push_back(&alloc);
                    placed = true;
                    break;
                }
            }
            
            if (!placed) {
                color_groups.push_back({&alloc});
            }
        }
        
        // 为每个颜色组分配连续内存
        size_t current_offset = 0;
        for (auto& group : color_groups) {
            size_t group_size = 0;
            for (auto* alloc : group) {
                group_size = std::max(group_size, alloc->size);
            }
            
            for (auto* alloc : group) {
                alloc->offset = current_offset;
            }
            
            current_offset += group_size;
        }
        
        total_memory_size = current_offset;
    }
};
```

### STensor内存规划
```cpp
class STensorMemoryPlanner {
private:
    struct SmemBank {
        int bank_id;
        std::vector<threadblock::STensor*> tensors;
        int current_offset;
    };
    
    std::vector<SmemBank> banks;
    int num_banks = 32;  // 典型GPU有32个共享内存bank
    
public:
    void plan_shared_memory(const threadblock::Graph& tb_graph) {
        auto stensors = tb_graph.get_all_tensors();
        
        // 1. 分析bank冲突
        analyze_bank_conflicts(stensors);
        
        // 2. 优化布局以避免bank冲突
        optimize_bank_layout(stensors);
        
        // 3. 分配共享内存偏移
        assign_shared_memory_offsets(stensors);
    }
    
private:
    void optimize_bank_layout(const std::vector<threadblock::STensor>& tensors) {
        for (auto& tensor : tensors) {
            if (tensor.layout == layout::SmemLayout::ROW_MAJOR) {
                // 检查是否会导致bank冲突
                if (causes_bank_conflict(tensor)) {
                    // 转换为交错布局
                    tensor.layout = layout::SmemLayout::SWIZZLED;
                    apply_swizzle_pattern(tensor);
                }
            }
        }
    }
    
    bool causes_bank_conflict(const threadblock::STensor& tensor) {
        // 检查访问模式是否会导致bank冲突
        int stride = tensor.strides[tensor.strides.size() - 1];
        return (stride % num_banks) == 0;
    }
};
```

## 🔄 融合分析器

### 操作融合识别
```cpp
class FusionAnalyzer {
public:
    struct FusionPattern {
        std::vector<type::TBOperatorType> op_sequence;  // 操作序列
        std::string fusion_name;                        // 融合名称
        bool requires_sync;                            // 是否需要同步
        int shared_memory_overhead;                    // 共享内存开销
        float performance_gain;                        // 性能增益
    };
    
    std::vector<FusionPattern> identify_fusion_opportunities(
        const threadblock::Graph& tb_graph) {
        
        std::vector<FusionPattern> patterns;
        
        // 1. 识别垂直融合机会
        auto vertical_patterns = find_vertical_fusion_patterns(tb_graph);
        patterns.insert(patterns.end(), vertical_patterns.begin(), vertical_patterns.end());
        
        // 2. 识别水平融合机会
        auto horizontal_patterns = find_horizontal_fusion_patterns(tb_graph);
        patterns.insert(patterns.end(), horizontal_patterns.begin(), horizontal_patterns.end());
        
        // 3. 识别循环融合机会
        auto loop_patterns = find_loop_fusion_patterns(tb_graph);
        patterns.insert(patterns.end(), loop_patterns.begin(), loop_patterns.end());
        
        return patterns;
    }
    
private:
    std::vector<FusionPattern> find_vertical_fusion_patterns(
        const threadblock::Graph& tb_graph) {
        
        std::vector<FusionPattern> patterns;
        auto ops = tb_graph.get_operators();
        
        for (size_t i = 0; i < ops.size() - 1; ++i) {
            auto* current_op = ops[i];
            auto* next_op = ops[i + 1];
            
            // 检查是否可以融合
            if (can_fuse_vertically(current_op, next_op)) {
                FusionPattern pattern;
                pattern.op_sequence = {current_op->op_type, next_op->op_type};
                pattern.fusion_name = get_fusion_name(current_op->op_type, next_op->op_type);
                pattern.requires_sync = requires_synchronization(current_op, next_op);
                pattern.performance_gain = estimate_fusion_benefit(current_op, next_op);
                
                patterns.push_back(pattern);
            }
        }
        
        return patterns;
    }
    
    bool can_fuse_vertically(threadblock::TBOperator* op1, 
                           threadblock::TBOperator* op2) {
        // 检查数据依赖
        for (auto& output : op1->output_tensors) {
            for (auto& input : op2->input_tensors) {
                if (output.guid == input.guid) {
                    // 有直接数据依赖，可能可以融合
                    return check_fusion_constraints(op1, op2);
                }
            }
        }
        return false;
    }
    
    bool check_fusion_constraints(threadblock::TBOperator* op1,
                                threadblock::TBOperator* op2) {
        // 1. 检查共享内存限制
        int smem_usage = estimate_shared_memory_usage(op1) + 
                        estimate_shared_memory_usage(op2);
        if (smem_usage > MAX_SHARED_MEMORY) {
            return false;
        }
        
        // 2. 检查寄存器压力
        int reg_usage = estimate_register_usage(op1) + 
                       estimate_register_usage(op2);
        if (reg_usage > MAX_REGISTERS_PER_THREAD) {
            return false;
        }
        
        // 3. 检查同步需求
        if (requires_global_synchronization(op1, op2)) {
            return false;  // 不能在同一个内核中融合
        }
        
        return true;
    }
};
```

## 🎯 架构特定优化

### Hopper架构优化
```cpp
class HopperTranspiler : public Transpiler {
private:
    // Hopper特有的优化
    bool use_tma_loads = true;      // 使用TMA加载
    bool use_wgmma = true;          // 使用WGMMA指令
    bool use_async_pipeline = true; // 使用异步流水线
    
public:
    std::string generate_hopper_kernel(const threadblock::Graph& tb_graph) override {
        std::stringstream code;
        
        // 1. 生成TMA描述符
        if (use_tma_loads) {
            code << generate_tma_descriptors(tb_graph);
        }
        
        // 2. 生成异步流水线
        if (use_async_pipeline) {
            code << generate_async_pipeline(tb_graph);
        }
        
        // 3. 生成WGMMA计算
        if (use_wgmma) {
            code << generate_wgmma_computation(tb_graph);
        }
        
        return code.str();
    }
    
private:
    std::string generate_tma_descriptors(const threadblock::Graph& tb_graph) {
        std::stringstream code;
        
        for (auto& tensor : tb_graph.get_input_tensors()) {
            code << "// TMA descriptor for " << tensor.name << "\n";
            code << "CUtensorMap tma_desc_" << tensor.guid << ";\n";
            code << "cuTensorMapEncodeTiled(&tma_desc_" << tensor.guid << ",\n";
            code << "    CU_TENSOR_MAP_DATA_TYPE_FLOAT16,\n";
            code << "    " << tensor.dims.size() << ",\n";
            code << "    tensor_" << tensor.guid << "_ptr,\n";
            code << "    tensor_" << tensor.guid << "_dims,\n";
            code << "    tensor_" << tensor.guid << "_strides,\n";
            code << "    tile_dims, tile_elementStrides,\n";
            code << "    CU_TENSOR_MAP_INTERLEAVE_NONE,\n";
            code << "    CU_TENSOR_MAP_SWIZZLE_NONE,\n";
            code << "    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,\n";
            code << "    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);\n\n";
        }
        
        return code.str();
    }
    
    std::string generate_wgmma_computation(const threadblock::Graph& tb_graph) {
        std::stringstream code;
        
        for (auto& op : tb_graph.get_operators()) {
            if (op->op_type == type::TB_MATMUL_OP) {
                auto* matmul_op = static_cast<threadblock::TBMatmulOp*>(op);
                
                code << "// WGMMA computation\n";
                code << "{\n";
                code << "  uint64_t desc_a = make_smem_desc(smem_a_ptr);\n";
                code << "  uint64_t desc_b = make_smem_desc(smem_b_ptr);\n";
                code << "  \n";
                code << "  wgmma::gemm_f16f16f32_m64n256k16(\n";
                code << "    acc,\n";
                code << "    desc_a,\n";
                code << "    desc_b,\n";
                code << "    scale_d);\n";
                code << "}\n\n";
            }
        }
        
        return code.str();
    }
};
```

### Blackwell架构优化
```cpp
class BlackwellTranspiler : public Transpiler {
private:
    bool use_fp4_precision = true;      // 使用FP4精度
    bool use_enhanced_wgmma = true;     // 使用增强WGMMA
    bool use_distributed_smem = true;   // 使用分布式共享内存
    
public:
    std::string generate_blackwell_kernel(const threadblock::Graph& tb_graph) override {
        std::stringstream code;
        
        // Blackwell特有的优化代码生成
        if (use_fp4_precision) {
            code << generate_fp4_operations(tb_graph);
        }
        
        if (use_enhanced_wgmma) {
            code << generate_enhanced_wgmma(tb_graph);
        }
        
        return code.str();
    }
};
```

## 🔧 多后端转译器

### Triton转译器
```cpp
namespace mirage::triton_transpiler {

class TritonTranspiler {
private:
    std::string triton_template;
    
public:
    std::string transpile_to_triton(const kernel::Graph& graph) {
        std::stringstream triton_code;
        
        // 1. 生成Triton函数签名
        triton_code << generate_triton_signature(graph);
        
        // 2. 生成Triton内核体
        triton_code << generate_triton_body(graph);
        
        // 3. 添加JIT编译装饰器
        triton_code << generate_triton_decorators(graph);
        
        return triton_code.str();
    }
    
private:
    std::string generate_triton_signature(const kernel::Graph& graph) {
        std::stringstream sig;
        sig << "@triton.jit\n";
        sig << "def mirage_kernel(\n";
        
        // 输入参数
        for (auto& input : graph.get_input_tensors()) {
            sig << "    " << input.name << "_ptr,\n";
            sig << "    " << input.name << "_stride_0,\n";
            // ... 更多步长参数
        }
        
        sig << "):\n";
        return sig.str();
    }
    
    std::string generate_triton_body(const kernel::Graph& graph) {
        std::stringstream body;
        
        // 生成Triton计算逻辑
        for (auto& op : graph.get_operators()) {
            body << generate_triton_op(op);
        }
        
        return body.str();
    }
};

}
```

### NKI转译器
```cpp
namespace mirage::nki_transpiler {

class NKITranspiler {
public:
    std::string transpile_to_nki(const kernel::Graph& graph) {
        std::stringstream nki_code;
        
        // 1. 导入NKI库
        nki_code << "#include <neuron_intrinsics.h>\n";
        nki_code << "#include <nki.h>\n\n";
        
        // 2. 生成NKI内核
        nki_code << generate_nki_kernel(graph);
        
        return nki_code.str();
    }
    
private:
    std::string generate_nki_kernel(const kernel::Graph& graph) {
        std::stringstream kernel;
        
        kernel << "void mirage_nki_kernel(\n";
        
        // 参数列表
        for (auto& tensor : graph.get_all_tensors()) {
            kernel << "    nki::tensor<" << get_nki_dtype(tensor.data_type) 
                  << ", " << tensor.dims.size() << "> " << tensor.name << ",\n";
        }
        
        kernel << ") {\n";
        
        // 内核体
        for (auto& op : graph.get_operators()) {
            kernel << generate_nki_operation(op);
        }
        
        kernel << "}\n";
        
        return kernel.str();
    }
};

}
```

## 📊 性能优化策略

### 内存带宽优化
```cpp
class MemoryBandwidthOptimizer {
public:
    void optimize_memory_access_patterns(std::string& cuda_code) {
        // 1. 向量化内存访问
        vectorize_memory_accesses(cuda_code);
        
        // 2. 合并内存访问
        coalesce_memory_accesses(cuda_code);
        
        // 3. 预取优化
        add_prefetch_instructions(cuda_code);
    }
    
private:
    void vectorize_memory_accesses(std::string& code) {
        // 将标量访问转换为向量访问
        std::regex scalar_load(R"((\w+) = \*\((\w+)\+(\d+)\))");
        std::string vector_load = "float4 $1_vec = *reinterpret_cast<float4*>($2+$3)";
        code = std::regex_replace(code, scalar_load, vector_load);
    }
};
```

### 寄存器优化
```cpp
class RegisterOptimizer {
public:
    void optimize_register_usage(std::string& cuda_code) {
        // 1. 寄存器重用分析
        analyze_register_reuse(cuda_code);
        
        // 2. 减少寄存器压力
        reduce_register_pressure(cuda_code);
        
        // 3. 优化变量生命周期
        optimize_variable_lifetime(cuda_code);
    }
};
```

转译器层通过多后端支持、智能内存管理和架构特定优化，为Mirage系统提供了高效的代码生成能力，确保在不同平台上都能获得最佳性能。