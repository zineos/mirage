# 搜索优化层设计

## 🎯 设计目标

搜索优化层是Mirage系统的智能核心，负责：
- **自动图优化**：自动搜索最优的计算图结构
- **符号执行**：使用符号推理进行图变换验证
- **正确性保证**：确保优化后图的语义正确性
- **性能导向**：基于性能模型指导搜索过程

## 🏗️ 核心组件架构

```
┌─────────────────────────────────────────────────────────────┐
│                Search & Optimization Layer                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │KernelGraphGen   │  │  SymbolicGraph   │  │  Verifier   │ │
│  │   (图生成器)     │  │   (符号图)       │  │  (验证器)    │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │  AbstractExpr   │  │   RangeProp      │  │ DimStrategy │ │
│  │  (抽象表达式)    │  │  (范围传播)      │  │ (维度策略)   │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🔍 KernelGraphGenerator (图生成器)

### 核心职责
- **搜索空间探索**：系统性地探索图变换空间
- **启发式引导**：使用启发式算法加速搜索
- **并行搜索**：支持多线程并行搜索
- **结果管理**：管理和排序搜索结果

### 类设计
```cpp
namespace mirage::search {

class KernelGraphGenerator {
private:
    GeneratorConfig config;                    // 生成器配置
    DimStrategy dim_strategy;                  // 维度策略
    std::shared_ptr<Verifier> verifier;       // 验证器
    std::vector<json> generated_graphs;       // 生成的图
    
    // 统计信息
    std::atomic<int> num_total_random_tests;
    std::atomic<int> num_valid_kernel_graphs;
    std::atomic<int> num_total_states;
    
public:
    KernelGraphGenerator(const kernel::Graph& computation_graph,
                        const GeneratorConfig& config,
                        const char* filename,
                        bool verbose = false);
    
    // 主要搜索方法
    void generate_kernel_graphs();
    void generate_kernel_graphs_symbolic();
    
    // 搜索控制
    void set_search_budget(int max_iterations);
    void set_verification_mode(VerificationMode mode);
    void enable_parallel_search(int num_threads);
};

}
```

### 搜索算法
```cpp
// 搜索配置
struct GeneratorConfig {
    int max_num_dims;              // 最大维度数
    int max_split_k;               // 最大分割因子
    bool enable_fusion;            // 启用操作融合
    bool enable_reordering;        // 启用操作重排序
    bool enable_tiling;            // 启用分块优化
    SearchStrategy strategy;       // 搜索策略
    int search_budget;             // 搜索预算
};

enum class SearchStrategy {
    BREADTH_FIRST,     // 广度优先搜索
    DEPTH_FIRST,       // 深度优先搜索
    BEST_FIRST,        // 最佳优先搜索
    GENETIC,           // 遗传算法
    SIMULATED_ANNEALING // 模拟退火
};
```

### 搜索过程
```cpp
void KernelGraphGenerator::generate_next_operator(
    SearchContext& context,
    std::function<bool(const SearchContext&)> verify,
    std::vector<SerializedSearchContext>& verified_graphs,
    size_t search_depth,
    bool is_new_thread_start) {
    
    // 1. 检查搜索深度限制
    if (search_depth >= config.max_search_depth) {
        return;
    }
    
    // 2. 生成可能的下一个操作符
    auto candidates = generate_operator_candidates(context);
    
    // 3. 评估每个候选
    for (auto& candidate : candidates) {
        SearchContext new_context = context;
        new_context.apply_operator(candidate);
        
        // 4. 验证新图的正确性
        if (verify(new_context)) {
            verified_graphs.push_back(new_context.serialize());
            
            // 5. 递归搜索更深层次
            generate_next_operator(new_context, verify, verified_graphs, 
                                 search_depth + 1, false);
        }
    }
}
```

## 🔮 SymbolicGraph (符号图)

### 设计目标
- **符号表示**：使用符号变量表示张量维度
- **约束求解**：基于约束求解进行维度推理
- **等价性验证**：验证图变换的等价性
- **抽象执行**：在符号层面执行图变换

### 核心类设计
```cpp
class SymbolicKNGraph {
private:
    std::vector<std::shared_ptr<SymbolicKNOperator>> operators;
    std::vector<SymbolicDTensor> tensors;
    TensorDimConstraints constraints;
    
public:
    // 构造和析构
    SymbolicKNGraph(const kernel::Graph& concrete_graph);
    
    // 符号操作
    SymbolicDTensor add_symbolic_operator(SymbolicKNOperatorType type,
                                         const std::vector<SymbolicDTensor>& inputs);
    
    // 约束管理
    void add_dimension_constraint(const TensorDimConstraint& constraint);
    bool solve_constraints();
    
    // 实例化
    bool instantiate_to_concrete_graph(kernel::Graph& output_graph,
                                     const DimVarAssignments& assignments);
};

class SymbolicTBGraph {
private:
    std::vector<std::shared_ptr<SymbolicTBOperator>> operators;
    std::vector<SymbolicSTensor> tensors;
    
public:
    // 符号线程块图操作
    void add_symbolic_matmul(const SymbolicSTensor& A, const SymbolicSTensor& B);
    void add_symbolic_reduction(const SymbolicSTensor& input, int reduction_dim);
};
```

### 符号张量表示
```cpp
struct SymbolicDTensor {
    size_t guid;
    std::vector<SymbolicTensorDim> dims;    // 符号维度
    DataType data_type;
    DmemLayout layout;
    
    // 符号维度可以是：
    // 1. 具体数值：SymbolicTensorDim(1024)
    // 2. 符号变量：SymbolicTensorDim("N")
    // 3. 表达式：SymbolicTensorDim("N * 2 + 1")
};

class SymbolicTensorDim {
private:
    std::shared_ptr<TensorDimExpr> expr;    // 维度表达式
    
public:
    SymbolicTensorDim(int concrete_value);
    SymbolicTensorDim(const std::string& var_name);
    SymbolicTensorDim(const TensorDimExpr& expression);
    
    // 符号运算
    SymbolicTensorDim operator+(const SymbolicTensorDim& other) const;
    SymbolicTensorDim operator*(const SymbolicTensorDim& other) const;
    SymbolicTensorDim operator/(const SymbolicTensorDim& other) const;
};
```

### 维度约束系统
```cpp
class TensorDimConstraints {
private:
    std::vector<TensorDimConstraint> constraints;
    z3::context z3_ctx;
    z3::solver z3_solver;
    
public:
    // 约束添加
    void add_equality_constraint(const SymbolicTensorDim& lhs,
                               const SymbolicTensorDim& rhs);
    void add_range_constraint(const SymbolicTensorDim& dim,
                            int min_val, int max_val);
    void add_divisibility_constraint(const SymbolicTensorDim& dim,
                                   int divisor);
    
    // 约束求解
    bool is_satisfiable();
    std::vector<DimVarAssignments> get_all_solutions();
    DimVarAssignments get_optimal_solution(const OptimizationObjective& obj);
};
```

## ✅ Verifier (验证系统)

### 验证策略
```cpp
class Verifier {
public:
    virtual ~Verifier() = default;
    virtual bool verify(const kernel::Graph& original,
                       const kernel::Graph& optimized) = 0;
};

// 概率验证器
class ProbabilisticVerifier : public Verifier {
private:
    int num_random_tests;
    float tolerance;
    
public:
    bool verify(const kernel::Graph& original,
               const kernel::Graph& optimized) override {
        for (int i = 0; i < num_random_tests; ++i) {
            auto inputs = generate_random_inputs(original);
            auto output1 = execute_graph(original, inputs);
            auto output2 = execute_graph(optimized, inputs);
            
            if (!outputs_match(output1, output2, tolerance)) {
                return false;
            }
        }
        return true;
    }
};

// 形式化验证器
class FormalVerifier : public Verifier {
private:
    z3::context ctx;
    z3::solver solver;
    
public:
    bool verify(const kernel::Graph& original,
               const kernel::Graph& optimized) override {
        // 1. 将图转换为Z3表达式
        auto expr1 = graph_to_z3_expr(original);
        auto expr2 = graph_to_z3_expr(optimized);
        
        // 2. 检查等价性
        solver.add(expr1 != expr2);
        return solver.check() == z3::unsat;  // 不满足则等价
    }
};
```

### 输出匹配验证
```cpp
class OutputMatcher {
public:
    static bool tensors_match(const std::vector<float>& tensor1,
                            const std::vector<float>& tensor2,
                            float rtol = 1e-5, float atol = 1e-8) {
        if (tensor1.size() != tensor2.size()) return false;
        
        for (size_t i = 0; i < tensor1.size(); ++i) {
            float diff = std::abs(tensor1[i] - tensor2[i]);
            float threshold = atol + rtol * std::abs(tensor2[i]);
            if (diff > threshold) return false;
        }
        return true;
    }
    
    static bool shapes_match(const std::vector<int>& shape1,
                           const std::vector<int>& shape2) {
        return shape1 == shape2;
    }
};
```

## 🧮 AbstractExpr (抽象表达式)

### 表达式系统
```cpp
class AbstractExpr {
public:
    virtual ~AbstractExpr() = default;
    virtual std::string to_string() const = 0;
    virtual std::shared_ptr<AbstractExpr> simplify() const = 0;
    virtual bool equals(const AbstractExpr& other) const = 0;
};

// 变量表达式
class VarExpr : public AbstractExpr {
private:
    std::string var_name;
    
public:
    VarExpr(const std::string& name) : var_name(name) {}
    std::string to_string() const override { return var_name; }
};

// 常数表达式
class ConstExpr : public AbstractExpr {
private:
    int value;
    
public:
    ConstExpr(int val) : value(val) {}
    std::string to_string() const override { return std::to_string(value); }
};

// 二元操作表达式
class BinaryOpExpr : public AbstractExpr {
private:
    std::shared_ptr<AbstractExpr> left, right;
    BinaryOp op_type;  // ADD, MUL, DIV, etc.
    
public:
    BinaryOpExpr(std::shared_ptr<AbstractExpr> l,
                std::shared_ptr<AbstractExpr> r,
                BinaryOp op) : left(l), right(r), op_type(op) {}
    
    std::shared_ptr<AbstractExpr> simplify() const override {
        auto simplified_left = left->simplify();
        auto simplified_right = right->simplify();
        
        // 常数折叠
        if (auto const_left = std::dynamic_pointer_cast<ConstExpr>(simplified_left)) {
            if (auto const_right = std::dynamic_pointer_cast<ConstExpr>(simplified_right)) {
                return fold_constants(const_left, const_right, op_type);
            }
        }
        
        return std::make_shared<BinaryOpExpr>(simplified_left, simplified_right, op_type);
    }
};
```

### 表达式求值
```cpp
class AbstractExprEvaluator {
public:
    static int evaluate(const AbstractExpr& expr,
                       const std::unordered_map<std::string, int>& var_assignments) {
        if (auto var_expr = dynamic_cast<const VarExpr*>(&expr)) {
            auto it = var_assignments.find(var_expr->get_name());
            if (it != var_assignments.end()) {
                return it->second;
            }
            throw std::runtime_error("Unbound variable: " + var_expr->get_name());
        }
        
        if (auto const_expr = dynamic_cast<const ConstExpr*>(&expr)) {
            return const_expr->get_value();
        }
        
        if (auto binary_expr = dynamic_cast<const BinaryOpExpr*>(&expr)) {
            int left_val = evaluate(*binary_expr->get_left(), var_assignments);
            int right_val = evaluate(*binary_expr->get_right(), var_assignments);
            return apply_binary_op(left_val, right_val, binary_expr->get_op());
        }
        
        throw std::runtime_error("Unknown expression type");
    }
};
```

## 📊 RangePropagation (范围传播)

### 范围表示
```cpp
class IKNRange {
private:
    int min_val, max_val;
    bool is_unbounded_above, is_unbounded_below;
    
public:
    IKNRange(int min_v, int max_v) : min_val(min_v), max_val(max_v),
                                    is_unbounded_above(false), is_unbounded_below(false) {}
    
    static IKNRange unbounded() {
        IKNRange range(0, 0);
        range.is_unbounded_above = true;
        range.is_unbounded_below = true;
        return range;
    }
    
    // 范围运算
    IKNRange operator+(const IKNRange& other) const;
    IKNRange operator*(const IKNRange& other) const;
    IKNRange intersect(const IKNRange& other) const;
    bool contains(int value) const;
};
```

### 范围传播算法
```cpp
class RangePropagator {
public:
    static std::vector<IKNRange> propagate_ranges(
        const SymbolicKNGraph& graph,
        const std::vector<IKNRange>& input_ranges) {
        
        std::unordered_map<size_t, IKNRange> tensor_ranges;
        
        // 初始化输入张量范围
        for (size_t i = 0; i < input_ranges.size(); ++i) {
            tensor_ranges[i] = input_ranges[i];
        }
        
        // 按拓扑序遍历操作符
        for (auto& op : graph.get_operators()) {
            auto output_ranges = propagate_through_operator(*op, tensor_ranges);
            for (size_t i = 0; i < output_ranges.size(); ++i) {
                tensor_ranges[op->get_output_tensors()[i].guid] = output_ranges[i];
            }
        }
        
        return extract_output_ranges(graph, tensor_ranges);
    }
    
private:
    static std::vector<IKNRange> propagate_through_operator(
        const SymbolicKNOperator& op,
        const std::unordered_map<size_t, IKNRange>& input_ranges) {
        
        switch (op.get_type()) {
            case SymbolicKNOperatorType::MATMUL:
                return propagate_matmul(op, input_ranges);
            case SymbolicKNOperatorType::ADD:
                return propagate_elementwise_add(op, input_ranges);
            case SymbolicKNOperatorType::REDUCTION:
                return propagate_reduction(op, input_ranges);
            default:
                return {IKNRange::unbounded()};
        }
    }
};
```

## 🎯 DimStrategy (维度策略)

### 策略配置
```cpp
struct DimStrategy {
    std::vector<int> allowed_tile_sizes;      // 允许的分块大小
    std::vector<int> allowed_split_factors;   // 允许的分割因子
    int max_num_splits;                       // 最大分割数
    bool enable_dynamic_shapes;               // 启用动态形状
    bool prefer_power_of_2;                   // 偏好2的幂次
    
    // 维度选择策略
    std::vector<int> generate_candidate_dims(int original_dim) const {
        std::vector<int> candidates;
        
        // 添加原始维度
        candidates.push_back(original_dim);
        
        // 添加分块维度
        for (int tile_size : allowed_tile_sizes) {
            if (original_dim % tile_size == 0) {
                candidates.push_back(tile_size);
            }
        }
        
        // 添加分割维度
        for (int factor : allowed_split_factors) {
            if (original_dim % factor == 0) {
                candidates.push_back(original_dim / factor);
            }
        }
        
        return candidates;
    }
};
```

## 🚀 搜索优化流程

### 完整搜索流程
```cpp
void optimize_computation_graph(const kernel::Graph& input_graph,
                              std::vector<kernel::Graph>& optimized_graphs) {
    // 1. 初始化搜索器
    KernelGraphGenerator generator(input_graph, config, "checkpoint.json");
    
    // 2. 符号图生成
    generator.generate_kernel_graphs_symbolic();
    
    // 3. 验证和筛选
    auto candidates = generator.get_generated_graphs();
    std::vector<kernel::Graph> verified_graphs;
    
    for (auto& candidate_json : candidates) {
        kernel::Graph candidate = deserialize_graph(candidate_json);
        if (verifier->verify(input_graph, candidate)) {
            verified_graphs.push_back(candidate);
        }
    }
    
    // 4. 性能评估和排序
    std::sort(verified_graphs.begin(), verified_graphs.end(),
             [](const kernel::Graph& a, const kernel::Graph& b) {
                 return estimate_performance(a) > estimate_performance(b);
             });
    
    optimized_graphs = std::move(verified_graphs);
}
```

搜索优化层通过符号执行、约束求解和形式化验证，为Mirage系统提供了强大的自动优化能力，确保在提升性能的同时保证正确性。