# 图构建层设计

## 🎯 设计目标

图构建层是Mirage系统的核心抽象层，负责：
- **双层图表示**：支持内核图(KNGraph)和线程块图(TBGraph)的分层抽象
- **张量管理**：提供设备张量(DTensor)和共享内存张量(STensor)的统一接口
- **操作符抽象**：定义丰富的计算操作符集合
- **融合支持**：支持复杂的操作融合模式

## 🏗️ 双层图架构

### 架构概览
```
┌─────────────────────────────────────┐
│            KNGraph                  │
│         (内核级图)                   │
│  ┌─────────────┐  ┌─────────────┐   │
│  │   DTensor   │  │ KNOperator  │   │
│  │  (设备张量)  │  │ (内核操作符) │   │
│  └─────────────┘  └─────────────┘   │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│            TBGraph                  │
│        (线程块级图)                  │
│  ┌─────────────┐  ┌─────────────┐   │
│  │   STensor   │  │ TBOperator  │   │
│  │ (共享内存张量)│  │(线程块操作符)│   │
│  └─────────────┘  └─────────────┘   │
└─────────────────────────────────────┘
```

## 📊 KNGraph (内核图) 设计

### 核心职责
- **全局视图**：管理整个计算的全局流程
- **设备内存**：处理GPU全局内存中的张量
- **内核调度**：协调多个CUDA内核的执行
- **分布式支持**：处理多GPU间的通信

### 类结构设计
```cpp
namespace mirage::kernel {

class Graph {
private:
    dim3 gpu_dim;                              // GPU网格维度
    std::vector<std::unique_ptr<KNOperator>> operators; // 操作符列表
    std::vector<DTensor> tensors;              // 张量列表
    std::unordered_map<size_t, DTensor*> guid_to_tensor; // GUID到张量映射
    
public:
    // 构造函数
    Graph(dim3 gpu_dim = {1, 1, 1}, bool disable_fingerprint = false);
    
    // 输入操作
    DTensor new_input(const std::vector<int>& dims,
                     const std::vector<size_t>& strides,
                     DataType data_type,
                     DmemLayout layout);
    
    // 输出操作
    void mark_output(const DTensor& tensor);
    
    // 计算操作
    DTensor matmul(const DTensor& A, const DTensor& B);
    DTensor add(const DTensor& A, const DTensor& B);
    DTensor rms_norm(const DTensor& input, const DTensor& weight);
    
    // 融合操作
    DTensor rmsnorm_linear(const DTensor& input,
                          const DTensor& norm_weight,
                          const DTensor& linear_weight);
};

}
```

### DTensor (设备张量) 设计
```cpp
struct DTensor {
    size_t guid;                    // 全局唯一标识符
    std::vector<int> dims;          // 张量维度
    std::vector<size_t> strides;    // 步长信息
    DataType data_type;             // 数据类型
    DmemLayout layout;              // 内存布局
    KNOperator* owner_op;           // 拥有此张量的操作符
    
    // 工具方法
    size_t num_elements() const;
    size_t size_in_bytes() const;
    bool is_contiguous() const;
};
```

### KNOperator (内核操作符) 层次
```cpp
class KNOperator {
public:
    KNOperatorType op_type;
    std::vector<DTensor> input_tensors;
    std::vector<DTensor> output_tensors;
    
    virtual ~KNOperator() = default;
    virtual bool can_fuse_with(const KNOperator* other) const;
    virtual std::string get_signature() const;
};

// 具体操作符类型
enum class KNOperatorType {
    KN_INPUT_OP,
    KN_OUTPUT_OP,
    KN_MATMUL_OP,
    KN_ELEMENT_UNARY_OP,
    KN_ELEMENT_BINARY_OP,
    KN_REDUCTION_OP,
    KN_RMS_NORM_OP,
    KN_CUSTOMIZED_OP,    // 自定义融合操作
    KN_ALL_REDUCE_OP
};
```

## 🧵 TBGraph (线程块图) 设计

### 核心职责
- **局部优化**：优化线程块内的计算模式
- **共享内存**：管理共享内存中的数据流
- **线程协作**：协调线程块内线程间的合作
- **内存层次**：优化不同内存层次的访问

### 类结构设计
```cpp
namespace mirage::threadblock {

class Graph {
private:
    dim3 grid_dim, block_dim;       // 网格和线程块维度
    int forloop_range;              // 循环范围
    int reduction_dimx;             // 归约维度
    std::vector<std::unique_ptr<TBOperator>> operators; // 操作符
    std::vector<STensor> tensors;   // 共享内存张量
    
public:
    Graph(dim3 grid_dim, dim3 block_dim, int forloop_range, int reduction_dimx);
    
    // 输入加载
    STensor new_input(const kernel::DTensor& dtensor,
                     int3 input_map,
                     int forloop_dim,
                     SmemLayout layout);
    
    // 输出存储
    kernel::DTensor mark_output(const STensor& stensor,
                               int3 output_map,
                               int forloop_dim,
                               TBEpilogueType epilogue);
    
    // 计算操作
    STensor matmul(const STensor& A, const STensor& B);
    STensor elementwise_add(const STensor& A, const STensor& B);
    STensor reduction_sum(const STensor& input, int dim);
};

}
```

### STensor (共享内存张量) 设计
```cpp
struct STensor {
    size_t guid;
    std::vector<int> dims;
    std::vector<int> strides;
    DataType data_type;
    SmemLayout layout;              // 共享内存布局
    TBOperator* owner_op;
    
    // 共享内存特有属性
    int smem_offset;                // 共享内存偏移
    bool is_input_tensor;           // 是否为输入张量
    bool is_output_tensor;          // 是否为输出张量
};
```

### 共享内存布局管理
```cpp
enum class SmemLayout {
    ROW_MAJOR,
    COLUMN_MAJOR,
    SWIZZLED,                       // 交错布局
    BANK_CONFLICT_FREE              // 避免bank冲突的布局
};

class SmemLayoutManager {
public:
    static int calculate_offset(const STensor& tensor);
    static bool has_bank_conflict(const SmemLayout& layout);
    static SmemLayout optimize_layout(const std::vector<STensor>& tensors);
};
```

## 🔧 操作符系统设计

### 基础操作符
```cpp
// 矩阵乘法操作符
class KNMatmulOp : public KNOperator {
    TransposeType transpose_a, transpose_b;
    float alpha, beta;
    
public:
    bool can_fuse_with_elementwise() const;
    CutlassGemmConfig get_cutlass_config() const;
};

// 元素级操作符
class KNElementUnaryOp : public KNOperator {
    ElementUnaryType op_type;  // RELU, GELU, SILU, EXP, etc.
    float scalar_param;        // 标量参数
};

class KNElementBinaryOp : public KNOperator {
    ElementBinaryType op_type; // ADD, MUL, DIV, etc.
    bool broadcast_input0, broadcast_input1;
};
```

### 融合操作符
```cpp
// RMSNorm + Linear融合
class KNRMSNormLinearOp : public KNOperator {
    float eps;                     // RMSNorm epsilon
    bool has_bias;                 // 是否有偏置
    
public:
    TBGraph generate_threadblock_graph() const override;
    std::string generate_cuda_code() const override;
};

// SiLU + 乘法 + Linear融合
class KNSiLUMulLinearOp : public KNOperator {
    bool with_residual;            // 是否包含残差连接
    
public:
    bool can_fuse_with_norm() const;
    int estimate_shared_memory_usage() const;
};
```

### 自定义操作符
```cpp
class KNCustomizedOp : public KNOperator {
    TBGraph bgraph;                // 内嵌的线程块图
    std::string custom_code;       // 自定义CUDA代码
    
public:
    void set_threadblock_graph(const TBGraph& graph);
    void add_custom_kernel_code(const std::string& code);
    bool verify_correctness() const;
};
```

## 🎯 图构建流程

### 1. 图初始化
```cpp
// 创建内核图
auto kn_graph = std::make_unique<kernel::Graph>(
    dim3{num_gpus, 1, 1},  // GPU网格
    false                   // 启用指纹
);

// 创建线程块图
auto tb_graph = std::make_unique<threadblock::Graph>(
    dim3{grid_x, grid_y, 1},    // 线程块网格
    dim3{block_x, block_y, 1},  // 线程块大小
    forloop_range,              // 循环范围
    reduction_dimx              // 归约维度
);
```

### 2. 张量定义
```cpp
// 定义输入张量
auto input = kn_graph->new_input(
    {batch_size, seq_len, hidden_size},  // 维度
    {seq_len * hidden_size, hidden_size, 1}, // 步长
    DataType::BFLOAT16,                  // 数据类型
    DmemLayout::ROW_MAJOR               // 内存布局
);

// 定义权重张量
auto weight = kn_graph->new_input(
    {hidden_size, hidden_size},
    {hidden_size, 1},
    DataType::BFLOAT16,
    DmemLayout::COLUMN_MAJOR
);
```

### 3. 操作定义
```cpp
// 基础操作
auto matmul_out = kn_graph->matmul(input, weight);
auto norm_out = kn_graph->rms_norm(matmul_out, norm_weight);

// 融合操作
auto fused_out = kn_graph->rmsnorm_linear(
    input, norm_weight, linear_weight
);
```

### 4. 输出标记
```cpp
// 标记输出张量
kn_graph->mark_output(fused_out);
```

## 🔍 图优化策略

### 操作融合
```cpp
class FusionOptimizer {
public:
    // 垂直融合：连续操作融合
    bool can_fuse_vertically(const KNOperator* op1, const KNOperator* op2);
    
    // 水平融合：并行操作融合
    bool can_fuse_horizontally(const std::vector<KNOperator*>& ops);
    
    // 循环融合：循环操作优化
    void fuse_forloop_operations(TBGraph& tb_graph);
};
```

### 内存优化
```cpp
class MemoryOptimizer {
public:
    // 张量生命周期分析
    void analyze_tensor_lifetime(const Graph& graph);
    
    // 内存复用
    void optimize_memory_reuse(Graph& graph);
    
    // 布局优化
    DmemLayout choose_optimal_layout(const DTensor& tensor);
};
```

### 调度优化
```cpp
class ScheduleOptimizer {
public:
    // 依赖分析
    std::vector<std::vector<KNOperator*>> build_dependency_graph(const Graph& graph);
    
    // 并行度分析
    int estimate_parallelism(const KNOperator* op);
    
    // 调度序列生成
    std::vector<KNOperator*> generate_execution_order(const Graph& graph);
};
```

## 📈 性能考量

### 内存访问优化
- **合并访问**：确保内存访问的合并性
- **缓存友好**：优化缓存命中率
- **Bank冲突避免**：共享内存访问优化

### 计算优化
- **指令级并行**：最大化ILP
- **数据重用**：减少冗余计算
- **流水线**：计算和内存访问重叠

### 同步优化
- **最小同步**：减少不必要的同步点
- **异步执行**：支持异步操作
- **批处理**：批量处理相似操作

## 🛠️ 调试和诊断

### 图可视化
```cpp
class GraphVisualizer {
public:
    void export_to_dot(const Graph& graph, const std::string& filename);
    void generate_html_report(const Graph& graph);
    void print_graph_statistics(const Graph& graph);
};
```

### 性能分析
```cpp
class GraphProfiler {
public:
    void profile_memory_usage(const Graph& graph);
    void analyze_computation_intensity(const Graph& graph);
    void estimate_execution_time(const Graph& graph);
};
```

图构建层通过双层抽象和丰富的操作符系统，为高效的LLM推理计算提供了强大的表达能力和优化空间。