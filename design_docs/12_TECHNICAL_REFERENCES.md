# 技术参考和相关论文

## 📚 核心技术论文

### Mirage系统论文
1. **Mirage: A Multi-Level Superoptimizer for Tensor Programs**
   - 作者：Mengdi Wu, Xinhao Cheng, Shengyu Liu, et al.
   - 会议：OSDI 2025
   - arXiv：https://arxiv.org/abs/2405.05751
   - 核心贡献：多级张量程序超优化器，Mirage系统的理论基础

### 持久化内核相关
2. **Persistent RNNs: Stashing Recurrent Weights On-Chip**
   - 作者：Greg Diamos, Shubho Sengupta, Bryan Catanzaro, et al.
   - 会议：ICML 2016
   - 核心思想：持久化内核的早期探索，为MPK提供了理论基础

3. **Cooperative Persistent Threads**
   - 作者：Duane Merrill, Michael Garland
   - 会议：CGO 2016
   - 核心贡献：持久化线程的协作模式，影响了MPK的任务调度设计

### GPU内存优化
4. **Unified Memory for CUDA Beginners**
   - NVIDIA Developer Blog
   - 核心内容：统一内存管理，为MPK的内存优化提供参考

5. **CUTLASS: Fast Linear Algebra in CUDA C++**
   - NVIDIA Research
   - GitHub：https://github.com/NVIDIA/cutlass
   - 核心贡献：高性能GEMM实现，MPK内核优化的重要参考

## 🧠 LLM推理优化

### 注意力机制优化
6. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**
   - 作者：Tri Dao, Daniel Y. Fu, Stefano Ermon, et al.
   - 会议：NeurIPS 2022
   - arXiv：https://arxiv.org/abs/2205.14135
   - 核心贡献：IO感知的注意力优化，MPK注意力内核的重要参考

7. **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**
   - 作者：Tri Dao
   - arXiv：https://arxiv.org/abs/2307.08691
   - 核心改进：更好的并行化和工作分区策略

8. **PagedAttention: Efficient Memory Management for Dynamic Neural Network Serving**
   - 作者：Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, et al.
   - 会议：SOSP 2023
   - 核心贡献：动态内存管理的注意力机制

### 模型并行化
9. **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**
   - 作者：Mohammad Shoeybi, Mostofa Patwary, Raul Puri, et al.
   - arXiv：https://arxiv.org/abs/1909.08053
   - 核心贡献：张量并行和流水线并行策略

10. **PaLM: Scaling Language Modeling with Pathways**
    - 作者：Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, et al.
    - Journal：JMLR 2022
    - 核心贡献：大规模模型的分布式训练和推理

### 推理加速
11. **FasterTransformer: NVIDIA's Library for Fast Transformer Inference**
    - NVIDIA GitHub：https://github.com/NVIDIA/FasterTransformer
    - 核心贡献：Transformer模型推理优化库

12. **TensorRT-LLM: A High-Performance Large Language Model Inference Library**
    - NVIDIA Developer
    - 核心贡献：LLM推理的TensorRT优化

## ⚡ 编译器优化

### 张量编译器
13. **TVM: An Automated End-to-End Optimizing Compiler for Deep Learning**
    - 作者：Tianqi Chen, Thierry Moreau, Ziheng Jiang, et al.
    - 会议：OSDI 2018
    - 核心贡献：端到端的深度学习编译器

14. **Ansor: Generating High-Performance Tensor Programs for Deep Learning**
    - 作者：Lianmin Zheng, Chengfan Jia, Minmin Sun, et al.
    - 会议：OSDI 2020
    - 核心贡献：自动张量程序生成和优化

15. **Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations**
    - 作者：Philippe Tillet, H.T. Kung, David Cox
    - 会议：PLDI 2019
    - 核心贡献：分块神经网络计算的中间语言

### 图优化
16. **TASO: Optimizing Deep Learning Computation with Automatic Generation of Graph Substitutions**
    - 作者：Zhihao Jia, Oded Padon, James Thomas, et al.
    - 会议：SOSP 2019
    - 核心贡献：自动图替换优化

17. **PET: Optimizing Tensor Programs with Partially Equivalent Transformations and Automated Corrections**
    - 作者：Haojie Wang, Jidong Zhai, Mingyu Gao, et al.
    - 会议：OSDI 2021
    - 核心贡献：部分等价变换的张量程序优化

## 🔍 形式化验证

### 符号执行
18. **SAGE: Whitebox Fuzzing for Security Testing**
    - 作者：Patrice Godefroid, Michael Y. Levin, David Molnar
    - 会议：CACM 2012
    - 核心贡献：白盒模糊测试和符号执行

19. **KLEE: Unassisted and Automatic Generation of High-Coverage Tests for Complex Systems Programs**
    - 作者：Cristian Cadar, Daniel Dunbar, Dawson Engler
    - 会议：OSDI 2008
    - 核心贡献：符号执行引擎，为MPK验证系统提供参考

### SMT求解器
20. **Z3: An Efficient SMT Solver**
    - 作者：Leonardo de Moura, Nikolaj Bjørner
    - 会议：TACAS 2008
    - 核心贡献：高效的SMT求解器，MPK形式化验证的核心工具

21. **Satisfiability Modulo Theories: Introduction and Applications**
    - 作者：Clark Barrett, Roberto Sebastiani, Sanjit A. Seshia, Cesare Tinelli
    - Journal：CACM 2009
    - 核心内容：SMT理论基础

## 🌐 分布式系统

### GPU通信
22. **NCCL: Optimized primitives for collective multi-GPU communication**
    - NVIDIA Research
    - GitHub：https://github.com/NVIDIA/nccl
    - 核心贡献：多GPU集合通信优化

23. **NVSHMEM: Enabling Efficient Multi-GPU Communication**
    - NVIDIA Research
    - 核心贡献：GPU间直接内存访问

### 分布式训练
24. **Horovod: Fast and Easy Distributed Deep Learning in TensorFlow**
    - 作者：Alexander Sergeev, Mike Del Balso
    - arXiv：https://arxiv.org/abs/1802.05799
    - 核心贡献：分布式深度学习框架

25. **PyTorch Distributed: Experiences on Accelerating Data Parallel Training**
    - 作者：Shen Li, Yanli Zhao, Rohan Varma, et al.
    - 会议：VLDB 2020
    - 核心贡献：数据并行训练加速

## 🏛️ GPU架构和优化

### GPU架构
26. **NVIDIA H100 Tensor Core GPU Architecture**
    - NVIDIA Whitepaper
    - 核心内容：Hopper架构详细介绍

27. **NVIDIA A100 Tensor Core GPU Architecture**
    - NVIDIA Whitepaper
    - 核心内容：Ampere架构技术细节

### CUDA编程
28. **CUDA C++ Programming Guide**
    - NVIDIA Developer Documentation
    - 核心内容：CUDA编程最佳实践

29. **CUDA C++ Best Practices Guide**
    - NVIDIA Developer Documentation
    - 核心内容：CUDA性能优化指南

### 内存优化
30. **Unified Memory Programming**
    - NVIDIA Developer Blog Series
    - 核心内容：统一内存编程模式

31. **CUDA Memory Management**
    - NVIDIA Developer Documentation
    - 核心内容：CUDA内存管理最佳实践

## 🔬 性能分析和调优

### 性能分析工具
32. **NVIDIA Nsight Systems**
    - NVIDIA Developer Tools
    - 核心功能：系统级性能分析

33. **NVIDIA Nsight Compute**
    - NVIDIA Developer Tools
    - 核心功能：内核级性能分析

### 性能建模
34. **Roofline: An Insightful Visual Performance Model for Multicore Architectures**
    - 作者：Samuel Williams, Andrew Waterman, David Patterson
    - 会议：CACM 2009
    - 核心贡献：性能分析的Roofline模型

35. **Performance Analysis of GPU-Accelerated Applications**
    - 多篇相关论文和技术报告
    - 核心内容：GPU应用性能分析方法论

## 🧪 测试和验证

### 模糊测试
36. **AFL++: Combining Incremental Steps of Fuzzing Research**
    - 作者：Andrea Fioraldi, Dominik Maier, Heiko Eißfeldt, Marc Heuse
    - 会议：WOOT 2020
    - 核心贡献：先进的模糊测试工具

37. **libFuzzer: A Library for Coverage-Guided Fuzz Testing**
    - LLVM Project
    - 核心贡献：覆盖率引导的模糊测试库

### 属性测试
38. **QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs**
    - 作者：Koen Claessen, John Hughes
    - 会议：ICFP 2000
    - 核心贡献：属性测试框架的开创性工作

## 📊 基准测试

### MLPerf基准
39. **MLPerf: An Industry Standard Benchmark Suite for Machine Learning Performance**
    - 作者：Vijay Janapa Reddi, Christine Cheng, David Kanter, et al.
    - Journal：IEEE Micro 2020
    - 核心贡献：机器学习性能标准基准

40. **MLPerf Inference Benchmark**
    - MLCommons
    - 网站：https://mlcommons.org/en/inference/
    - 核心内容：推理性能基准测试

### GPU基准测试
41. **SHOC: The Scalable HeterOgeneous Computing Benchmark Suite**
    - 作者：Anthony Danalis, Gabriel Marin, Collin McCurdy, et al.
    - 会议：GPGPU 2010
    - 核心贡献：异构计算基准测试套件

## 🔧 相关工具和库

### 深度学习框架
42. **PyTorch: An Imperative Style, High-Performance Deep Learning Library**
    - 作者：Adam Paszke, Sam Gross, Francisco Massa, et al.
    - 会议：NeurIPS 2019
    - 核心贡献：动态图深度学习框架

43. **TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems**
    - 作者：Martín Abadi, Ashish Agarwal, Paul Barham, et al.
    - arXiv：https://arxiv.org/abs/1603.04467
    - 核心贡献：大规模机器学习系统

### 高性能计算库
44. **cuBLAS: CUDA Basic Linear Algebra Subroutines**
    - NVIDIA Developer Documentation
    - 核心功能：GPU基础线性代数运算

45. **cuDNN: NVIDIA CUDA Deep Neural Network Library**
    - NVIDIA Developer Documentation
    - 核心功能：深度神经网络GPU加速库

## 📈 未来发展方向

### 新兴架构
46. **Cerebras-CS2: The Largest Chip Ever Built**
    - Cerebras Systems
    - 核心内容：大芯片架构的探索

47. **GraphCore IPU Architecture**
    - GraphCore Technical Papers
    - 核心内容：智能处理单元架构

### 量化和压缩
48. **8-bit Optimizers via Block-wise Quantization**
    - 作者：Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer
    - 会议：ICLR 2022
    - 核心贡献：块级量化优化器

49. **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale**
    - 作者：Tim Dettmers, Mike Lewis, Shaden Smith, Luke Zettlemoyer
    - 会议：NeurIPS 2022
    - 核心贡献：大规模Transformer的8位矩阵乘法

### 新型内存技术
50. **High Bandwidth Memory (HBM) and Its Applications**
    - 多篇技术报告和论文
    - 核心内容：高带宽内存技术

## 📖 推荐阅读顺序

### 初学者路径
1. 从Mirage核心论文开始了解系统概念
2. 阅读FlashAttention理解注意力优化
3. 学习CUDA编程指南掌握GPU编程基础
4. 研究TVM了解张量编译器概念

### 进阶开发者路径
1. 深入研究持久化内核相关论文
2. 学习Z3和形式化验证方法
3. 掌握分布式系统和GPU通信技术
4. 研究性能分析和优化技术

### 研究者路径
1. 全面阅读编译器优化相关论文
2. 深入研究图优化和符号执行
3. 关注最新的GPU架构发展
4. 探索新兴的AI加速器架构

## 🔗 有用的资源链接

### 官方文档
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog)
- [PyTorch Documentation](https://pytorch.org/docs/)

### 开源项目
- [Mirage Project](https://github.com/mirage-project/mirage)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [TVM](https://github.com/apache/tvm)
- [Triton](https://github.com/openai/triton)

### 会议和期刊
- **OSDI**: Operating Systems Design and Implementation
- **SOSP**: Symposium on Operating Systems Principles  
- **PLDI**: Programming Language Design and Implementation
- **NeurIPS**: Neural Information Processing Systems
- **ICML**: International Conference on Machine Learning

### 在线课程和教程
- [NVIDIA Deep Learning Institute](https://www.nvidia.com/en-us/training/)
- [CS149: Parallel Computing (Stanford)](https://cs149.stanford.edu/)
- [CS267: Applications of Parallel Computing (UC Berkeley)](https://sites.google.com/lbl.gov/cs267-spr2021)

这些技术参考为Mirage Persistent Kernel的设计和实现提供了坚实的理论基础和实践指导，是深入理解系统架构和持续改进的重要资源。