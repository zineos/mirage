# Mirage Persistent Kernel (MPK) 设计文档

本文件夹包含了Mirage Persistent Kernel项目的完整架构设计文档。

## 📚 文档结构

### 🏗️ 总体架构
- [`01_OVERVIEW.md`](./01_OVERVIEW.md) - 项目总体概览和核心技术原理
- [`02_ARCHITECTURE.md`](./02_ARCHITECTURE.md) - 整体架构设计和模块关系

### 🔧 核心模块设计
- [`03_PYTHON_API.md`](./03_PYTHON_API.md) - Python API层设计
- [`04_GRAPH_CONSTRUCTION.md`](./04_GRAPH_CONSTRUCTION.md) - 图构建层设计 (KNGraph & TBGraph)
- [`05_SEARCH_OPTIMIZATION.md`](./05_SEARCH_OPTIMIZATION.md) - 搜索优化层设计
- [`06_TRANSPILER.md`](./06_TRANSPILER.md) - 转译器层设计
- [`07_RUNTIME_SYSTEM.md`](./07_RUNTIME_SYSTEM.md) - 运行时系统设计
- [`08_CUDA_KERNELS.md`](./08_CUDA_KERNELS.md) - CUDA内核层设计

### 🚀 技术实现
- [`09_PERFORMANCE_OPTIMIZATION.md`](./09_PERFORMANCE_OPTIMIZATION.md) - 性能优化机制
- [`10_MULTI_GPU_SUPPORT.md`](./10_MULTI_GPU_SUPPORT.md) - 多GPU支持设计
- [`11_QUALITY_ASSURANCE.md`](./11_QUALITY_ASSURANCE.md) - 质量保证体系

### 📖 参考资料
- [`12_TECHNICAL_REFERENCES.md`](./12_TECHNICAL_REFERENCES.md) - 技术参考和相关论文

## 🎯 阅读建议

### 新用户入门
1. 先阅读 [`01_OVERVIEW.md`](./01_OVERVIEW.md) 了解项目背景和核心概念
2. 然后查看 [`02_ARCHITECTURE.md`](./02_ARCHITECTURE.md) 理解整体架构
3. 根据兴趣选择具体模块文档深入了解

### 开发者参考
- 前端开发：重点关注 [`03_PYTHON_API.md`](./03_PYTHON_API.md)
- 编译器开发：重点关注 [`05_SEARCH_OPTIMIZATION.md`](./05_SEARCH_OPTIMIZATION.md) 和 [`06_TRANSPILER.md`](./06_TRANSPILER.md)
- 运行时开发：重点关注 [`07_RUNTIME_SYSTEM.md`](./07_RUNTIME_SYSTEM.md) 和 [`08_CUDA_KERNELS.md`](./08_CUDA_KERNELS.md)
- 性能优化：重点关注 [`09_PERFORMANCE_OPTIMIZATION.md`](./09_PERFORMANCE_OPTIMIZATION.md)

### 架构师参考
- 系统设计：[`02_ARCHITECTURE.md`](./02_ARCHITECTURE.md)
- 技术选型：[`12_TECHNICAL_REFERENCES.md`](./12_TECHNICAL_REFERENCES.md)
- 质量保证：[`11_QUALITY_ASSURANCE.md`](./11_QUALITY_ASSURANCE.md)

## 🔄 文档维护

这些文档会随着项目的发展持续更新。如果您发现任何不准确或需要补充的内容，请提交issue或PR。

## 📞 联系方式

- 项目主页: https://github.com/mirage-project/mirage
- Slack频道: [Join Slack](https://join.slack.com/t/miragesystem/shared_invite/zt-37reobr1i-SKjxeYF3GXdPDoCvtVbjTQ)
- 技术博客: [Blog Post](https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17)