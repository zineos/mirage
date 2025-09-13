# Python API层设计

## 🎯 设计目标

Python API层是Mirage Persistent Kernel的用户接口层，设计目标是：
- **简单易用**：几十行代码即可使用MegaKernel
- **功能完整**：支持完整的LLM推理流程
- **性能高效**：最小化Python层开销
- **集成友好**：与PyTorch等框架无缝集成

## 🏗️ 核心组件架构

### 主要类结构
```python
mirage/
├── __init__.py           # 主入口和全局配置
├── persistent_kernel.py  # 持久化内核主接口
├── kernel.py            # 内核图接口
├── threadblock.py       # 线程块图接口
├── core.py              # C++绑定核心功能
└── utils.py             # 工具函数
```

## 🔧 PersistentKernel 主接口

### 类设计
```python
class PersistentKernel:
    def __init__(self, 
                 world_size: int,
                 mpi_rank: int, 
                 num_workers: int,
                 num_local_schedulers: int,
                 num_remote_schedulers: int,
                 meta_tensors: List[torch.Tensor],
                 profiler_tensor: torch.Tensor = None):
        """
        初始化持久化内核
        
        Args:
            world_size: GPU总数
            mpi_rank: 当前GPU rank
            num_workers: worker线程数
            num_local_schedulers: 本地调度器数量  
            num_remote_schedulers: 远程调度器数量
            meta_tensors: 元数据张量列表 [step, tokens]
            profiler_tensor: 性能分析张量
        """
```

### 核心方法

#### 1. 张量管理
```python
def attach_input(self, 
                torch_tensor: torch.Tensor, 
                name: str) -> DTensor:
    """
    附加PyTorch张量作为输入
    
    Args:
        torch_tensor: PyTorch张量
        name: 张量名称（用于CUDA代码引用）
        
    Returns:
        DTensor: 设备张量对象
    """

def new_tensor(self,
               dims: Tuple[int, ...],
               dtype: DataType,
               name: str,
               io_category: str = "cuda_tensor") -> DTensor:
    """
    创建新的张量
    
    Args:
        dims: 张量维度
        dtype: 数据类型
        name: 张量名称
        io_category: IO类别 ("cuda_tensor" 或 "nvshmem_tensor")
        
    Returns:
        DTensor: 新创建的设备张量
    """
```

#### 2. 操作定义
```python
def rmsnorm_linear_layer(self,
                        input: DTensor,
                        weight_norm: DTensor,
                        weight_linear: DTensor,
                        output: DTensor,
                        grid_dim: Tuple[int, int, int],
                        block_dim: Tuple[int, int, int]):
    """
    RMSNorm + Linear融合层
    
    Args:
        input: 输入张量
        weight_norm: RMSNorm权重
        weight_linear: Linear权重  
        output: 输出张量
        grid_dim: 网格维度
        block_dim: 线程块维度
    """

def attention(self,
              input: DTensor,
              weight_qkv: DTensor,
              weight_o: DTensor,
              output: DTensor,
              attention_type: str = "multi_head"):
    """
    注意力机制
    
    Args:
        input: 输入张量
        weight_qkv: QKV权重矩阵
        weight_o: 输出权重矩阵
        output: 输出张量
        attention_type: 注意力类型
    """
```

#### 3. 编译和执行
```python
def compile(self) -> None:
    """
    编译计算图为CUDA代码
    
    执行以下步骤：
    1. 图优化搜索
    2. 符号验证
    3. 代码转译
    4. CUDA编译
    5. 动态库加载
    """

def __call__(self) -> None:
    """
    执行编译后的MegaKernel
    
    启动持久化内核并执行推理
    """
```

## 📊 数据类型系统

### DTensor (设备张量)
```python
@dataclass
class DTensor:
    """设备张量表示"""
    guid: int                    # 全局唯一标识
    dims: List[int]             # 张量维度
    strides: List[int]          # 步长信息
    data_type: DataType         # 数据类型
    layout: DmemLayout          # 内存布局
    owner_op: Optional[KNOperator] = None  # 所属操作
```

### 支持的数据类型
```python
class DataType(Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16" 
    BFLOAT16 = "bfloat16"
    INT32 = "int32"
    INT64 = "int64"
```

### 内存布局类型
```python
class DmemLayout(Enum):
    ROW_MAJOR = "row_major"      # 行优先
    COLUMN_MAJOR = "column_major" # 列优先
    CUSTOM = "custom"            # 自定义布局
```

## 🔗 与PyTorch集成

### 张量转换
```python
def torch_to_mirage(torch_tensor: torch.Tensor) -> DTensor:
    """PyTorch张量转换为Mirage DTensor"""
    return DTensor(
        dims=list(torch_tensor.shape),
        strides=list(torch_tensor.stride()),
        data_type=_torch_to_mirage_dtype(torch_tensor.dtype),
        layout=DmemLayout.ROW_MAJOR
    )

def mirage_to_torch(dtensor: DTensor) -> torch.Tensor:
    """Mirage DTensor转换为PyTorch张量"""
    return torch.empty(
        dtensor.dims,
        dtype=_mirage_to_torch_dtype(dtensor.data_type),
        device='cuda'
    )
```

### 自动梯度支持
```python
class MirageFunction(torch.autograd.Function):
    """支持自动梯度的Mirage函数包装"""
    
    @staticmethod
    def forward(ctx, *inputs):
        # 执行Mirage MegaKernel
        return mirage_kernel(*inputs)
    
    @staticmethod  
    def backward(ctx, *grad_outputs):
        # 反向传播（未来支持）
        pass
```

## 🛠️ 工具函数和配置

### 全局配置
```python
class GlobalConfig:
    """全局配置管理"""
    gpu_device_id: int = 0
    bypass_compile_errors: bool = False
    enable_profiling: bool = False
    cache_dir: str = "/tmp/mirage_cache"

# 全局配置实例
global_config = GlobalConfig()

def set_gpu_device_id(device_id: int):
    """设置GPU设备ID"""
    global_config.gpu_device_id = device_id

def bypass_compile_errors(value: bool = True):
    """设置是否跳过编译错误"""
    global_config.bypass_compile_errors = value
```

### 工具函数
```python
def new_kernel_graph() -> KNGraph:
    """创建新的内核图"""
    return KNGraph(core.CyKNGraph())

def new_threadblock_graph(grid_dim: Tuple[int, int, int],
                         block_dim: Tuple[int, int, int],
                         forloop_range: int,
                         reduction_dimx: int) -> TBGraph:
    """创建新的线程块图"""
    return TBGraph(core.CyTBGraph(grid_dim, block_dim, forloop_range, reduction_dimx))
```

## 🔍 使用示例

### 基本使用流程
```python
import mirage as mi
import torch

# 1. 初始化持久化内核
mpk = mi.PersistentKernel(
    world_size=1,
    mpi_rank=0,
    num_workers=96,
    num_local_schedulers=48,
    num_remote_schedulers=0,
    meta_tensors=[step_tensor, tokens_tensor]
)

# 2. 附加输入张量
input_tensor = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
x = mpk.attach_input(input_tensor, "input")

# 3. 创建权重张量
w_norm = mpk.new_tensor(
    dims=(hidden_size,),
    dtype=mi.float16,
    name="w_norm"
)

w_linear = mpk.new_tensor(
    dims=(hidden_size, hidden_size),
    dtype=mi.float16, 
    name="w_linear"
)

# 4. 定义计算
output = mpk.new_tensor(
    dims=(batch_size, seq_len, hidden_size),
    dtype=mi.float16,
    name="output"
)

mpk.rmsnorm_linear_layer(
    input=x,
    weight_norm=w_norm,
    weight_linear=w_linear,
    output=output,
    grid_dim=(96, 1, 1),
    block_dim=(128, 1, 1)
)

# 5. 编译和执行
mpk.compile()
mpk()  # 执行推理
```

### 完整LLM示例
```python
# Qwen3模型示例
def create_qwen3_megakernel(model_config):
    mpk = mi.PersistentKernel(
        world_size=model_config.world_size,
        mpi_rank=model_config.rank,
        num_workers=96,
        num_local_schedulers=48,
        meta_tensors=[step_tensor, tokens_tensor]
    )
    
    # 嵌入层
    embed_out = mpk.embedding_layer(tokens, embed_weights)
    
    # Transformer层
    x = embed_out
    for layer_id in range(model_config.num_layers):
        # RMSNorm + QKV投影
        qkv_out = mpk.rmsnorm_linear_layer(x, norm_weights[layer_id], qkv_weights[layer_id])
        
        # 注意力计算
        attn_out = mpk.attention(qkv_out, o_weights[layer_id])
        
        # 残差连接
        x = mpk.add(x, attn_out)
        
        # FFN
        ffn_out = mpk.silu_mul_linear(x, gate_weights[layer_id], up_weights[layer_id], down_weights[layer_id])
        x = mpk.add(x, ffn_out)
    
    # 输出层
    logits = mpk.linear_layer(x, lm_head_weights)
    
    return mpk
```

## 🚀 性能优化

### 内存管理优化
- **零拷贝**：直接使用PyTorch张量的内存
- **内存池**：复用临时张量内存
- **预分配**：提前分配所需内存

### 执行优化
- **JIT编译**：运行时编译优化
- **缓存机制**：缓存编译结果
- **批处理**：支持动态批大小

### 调试支持
- **详细日志**：完整的执行日志
- **性能分析**：内核执行时间分析
- **内存追踪**：内存使用情况监控

## 🔧 错误处理

### 异常类型
```python
class MirageError(Exception):
    """Mirage基础异常"""
    pass

class CompilationError(MirageError):
    """编译错误"""
    pass

class RuntimeError(MirageError):
    """运行时错误"""
    pass

class InputNotFoundError(MirageError):
    """输入张量未找到"""
    pass
```

### 错误恢复
- **编译错误跳过**：可选择跳过非关键编译错误
- **运行时检查**：执行前的参数验证
- **资源清理**：异常时的资源清理机制

Python API层通过简洁的接口设计和完善的功能支持，为用户提供了高效易用的MegaKernel开发体验。