# Loss Landscape Core

一个干净、模块化的PyTorch库，用于可视化神经网络损失地形。

从原始[loss-landscape](https://github.com/zingyi-li/Loss-Surfaces)项目中提取并重构，具有：
- 针对单GPU使用的简化API
- 移除MPI依赖
- 模块化架构便于集成
- **⭐ 新增：自定义指标函数支持** - 一次计算多个指标
- **⭐ 新增：自动检测和多指标可视化**
- 完全支持1D曲线、2D轮廓、3D曲面和ParaView导出
- 与标准PyTorch损失的完全向后兼容性

## 功能特性

✨ **主要特性**：

- ✅ **简化API** - 易于在单GPU机器上使用
- ✅ **1D/2D损失地形** - 计算损失曲线和2D曲面
- ✅ **多种可视化类型**
  - 轮廓图（线性和填充）
  - 带颜色条的热力图
  - 3D曲面图
  - ParaView兼容的VTP导出
- ✅ **⭐ 自定义指标函数** - 新增！
  - 定义自己的指标函数
  - 同时计算多个损失值和指标
  - 返回指标字典
  - 自动存储和可视化
- ✅ **⭐ 自动指标检测** - 新增！
  - 自动检测所有计算的指标
  - 为每个指标生成单独的可视化
  - 使用`surf_name='auto'`绘制所有内容
- ✅ **灵活的criterion类型**
  - 标准PyTorch损失模块（CrossEntropyLoss、MSELoss等）
  - 返回指标字典的自定义可调用函数
- ✅ **类型安全的自动检测**
  - 自动区分PyTorch损失和自定义函数
  - 无需额外标志或配置
- ✅ **完全向后兼容**
  - 所有现有代码继续正常工作
  - 现有HDF5文件仍可读取
  - 绘图函数支持两种模式

## 安装

```bash
pip install torch torchvision h5py matplotlib scipy seaborn numpy
```

## 快速开始

### 1D损失曲线（标准损失）

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from loss_landscape_core import LossLandscape

# 设置模型和数据
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)

# 使用标准PyTorch损失创建地形
landscape = LossLandscape(model, dataloader,
                         criterion=nn.CrossEntropyLoss(),
                         use_cuda=True)

# 计算1D地形
result = landscape.compute_1d(
    directions='random',
    xrange=(-1, 1, 51),
    normalize='filter',
    ignore='biasbn'
)

# 可视化
landscape.plot_1d(loss_max=5, show=True)
```

### 2D损失曲面

```python
# 计算2D地形
result = landscape.compute_2d(
    xrange=(-1, 1, 51),
    yrange=(-1, 1, 51),
    normalize='filter'
)

# 绘制轮廓和3D曲面
landscape.plot_2d_contour(vmin=0.1, vmax=10, vlevel=0.5)
landscape.plot_2d_surface(show=True)
```

### ⭐ 自定义指标（新增！）

同时计算多个自定义指标：

```python
import torch
import torch.nn as nn
from loss_landscape_core import LossLandscape

def compute_custom_metrics(net, dataloader, use_cuda):
    """
    计算多个指标的自定义指标函数。

    参数：
        net: PyTorch模型
        dataloader: 评估用的数据加载器
        use_cuda: 是否使用GPU

    返回：
        包含指标值的字典
    """
    net.eval()
    device = 'cuda' if use_cuda else 'cpu'

    total_ce = 0.0
    total_smooth_l1 = 0.0
    total_correct = 0
    total_samples = 0

    criterion_ce = nn.CrossEntropyLoss()
    criterion_smooth_l1 = nn.SmoothL1Loss()

    with torch.no_grad():
        for inputs, targets in dataloader:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)

            # 计算不同的损失函数
            ce_loss = criterion_ce(outputs, targets)
            targets_onehot = torch.nn.functional.one_hot(
                targets, num_classes=outputs.size(1)).float()
            smooth_l1_loss = criterion_smooth_l1(outputs, targets_onehot)

            # 累加指标
            total_ce += ce_loss.item() * inputs.size(0)
            total_smooth_l1 += smooth_l1_loss.item() * inputs.size(0)

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += inputs.size(0)

    # 返回多个指标作为字典
    return {
        'ce_loss': total_ce / total_samples,
        'smooth_l1_loss': total_smooth_l1 / total_samples,
        'accuracy': 100.0 * total_correct / total_samples
    }

# 与LossLandscape一起使用
landscape = LossLandscape(model, dataloader,
                         criterion=compute_custom_metrics,  # 传入函数
                         use_cuda=True)

# 使用自定义指标计算2D地形
result = landscape.compute_2d(
    xrange=(-1, 1, 11),
    yrange=(-1, 1, 11)
)

# result['losses']现在是一个字典：
# {
#     'ce_loss': shape (11, 11),
#     'smooth_l1_loss': shape (11, 11),
#     'accuracy': shape (11, 11)
# }

# 自动绘制所有指标
landscape.plot_2d_contour(surf_name='auto', vmin=0.1, vmax=10)
landscape.plot_2d_surface(surf_name='auto')

# 或绘制特定指标
landscape.plot_2d_contour(surf_name='train_loss_ce_loss', vmin=0.1, vmax=5)
```

### 导出到ParaView（高质量渲染）

```python
# 导出2D曲面用于ParaView渲染
vtp_file = landscape.export_paraview(surf_name='train_loss_ce_loss',
                                     log=False, zmax=10, interp=-1)
```

然后用[ParaView](https://www.paraview.org/)打开`.vtp`文件进行专业可视化。

## 自定义指标指南

### 什么是自定义指标？

自定义指标函数允许你：
1. **计算多个损失值** - 同时比较不同的损失函数
2. **跟踪各种指标** - 准确率、F1分数、精度、召回等
3. **联合可视化** - 查看不同指标如何在权重空间中变化
4. **分析权衡** - 理解不同目标之间的关系

### 工作原理

**类型检测机制：**

库自动检测criterion类型：

```python
# 标准PyTorch损失（nn.Module）→ 标准模式
criterion = nn.CrossEntropyLoss()

# 自定义指标函数（可调用，非nn.Module）→ 自定义模式
def my_metrics(net, dataloader, use_cuda):
    return {'metric1': value1, 'metric2': value2}
```

**存储格式：**

- **标准模式**：保存为`'train_loss'`、`'train_acc'`（向后兼容）
- **自定义模式**：保存为`'train_loss_metric_name'`、`'train_loss_metric2'`等

**自动绘图：**

```python
# 绘制所有检测到的指标
landscape.plot_2d_contour(surf_name='auto')  # 检测所有train_loss_*键

# 绘制特定指标
landscape.plot_2d_contour(surf_name='train_loss_f1_score')

# 默认：绘制标准损失（向后兼容）
landscape.plot_2d_contour()  # 绘制'train_loss'
```

### 函数签名

自定义指标函数必须遵循以下签名：

```python
def my_metrics(net: nn.Module,
               dataloader: DataLoader,
               use_cuda: bool) -> dict:
    """
    为模型在给定数据加载器上计算指标。

    参数：
        net: PyTorch模型（处于eval模式）
        dataloader: 评估用的数据加载器
        use_cuda: 模型是否在GPU上

    返回：
        将指标名称映射到值的字典：
        {'metric_name': float_value, ...}
    """
    pass
```

### 完整示例

见`example_custom_metrics.py`获取完整的工作示例，包含：
- 标准损失模式（向后兼容）
- 1D地形与自定义指标
- 2D地形与自定义指标
- 所有指标的自动绘图

## API参考

### LossLandscape类

用于损失地形计算和可视化的主类。

**构造函数：**
```python
LossLandscape(net, dataloader, criterion=None, use_cuda=False, surf_file=None)
```

**参数：**
- `net`：PyTorch模型
- `dataloader`：用于评估的数据加载器
- `criterion`：损失函数或指标函数
  - 可以是`nn.Module`（CrossEntropyLoss、MSELoss等）← 标准模式
  - 可以是返回`dict`指标的`callable` ← 自定义模式
  - 默认值：`nn.CrossEntropyLoss()`
- `use_cuda`：是否使用GPU（默认值：False）
- `surf_file`：保存HDF5结果的路径（默认值：'loss_surface.h5'）

**方法：**

#### `compute_1d()`
计算1D损失地形。

```python
result = landscape.compute_1d(
    directions='random',      # 或'target'（带target_model）
    xrange=(-1, 1, 51),       # (min, max, num_points)
    dir_type='weights',       # 或'states'
    normalize='filter',       # filter|layer|weight|dfilter|dlayer
    ignore='biasbn',          # 忽略偏差和BN参数
    target_model=None,        # 如果directions='target'则必需
    save=True                 # 保存到HDF5文件
)
```

**返回值：**

**标准模式**（使用nn.Module criterion）：
```python
{
    'losses': np.array([...]),      # 损失值的1D数组
    'accuracies': np.array([...]),  # 准确率的1D数组
    'xcoordinates': np.array([...]) # X坐标
}
```

**自定义模式**（使用callable criterion）：
```python
{
    'losses': {
        'metric1': np.array([...]),
        'metric2': np.array([...]),
        ...
    },
    'xcoordinates': np.array([...])
}
```

#### `compute_2d()`
计算2D损失地形。

```python
result = landscape.compute_2d(
    xrange=(-1, 1, 51),     # (min, max, num_points)
    yrange=(-1, 1, 51),     # (min, max, num_points)
    dir_type='weights',     # 或'states'
    normalize='filter',     # filter|layer|weight|dfilter|dlayer
    ignore='biasbn',        # 忽略偏差和BN参数
    x_target=None,          # 可选的X方向目标模型
    y_target=None,          # 可选的Y方向目标模型
    save=True               # 保存到HDF5文件
)
```

**返回值：**

**标准模式：**
```python
{
    'losses': np.array([...]).shape(nx, ny),      # 2D数组
    'accuracies': np.array([...]).shape(nx, ny),  # 2D数组
    'xcoordinates': np.array([...]),
    'ycoordinates': np.array([...])
}
```

**自定义模式：**
```python
{
    'losses': {
        'metric1': np.array([...]).shape(nx, ny),
        'metric2': np.array([...]).shape(nx, ny),
        ...
    },
    'xcoordinates': np.array([...]),
    'ycoordinates': np.array([...])
}
```

#### `plot_1d()`
可视化1D地形。

```python
landscape.plot_1d(xmin=-1, xmax=1, loss_max=5, log=False, show=False)
```

#### `plot_2d_contour()`
绘制2D轮廓可视化。

```python
landscape.plot_2d_contour(
    surf_name='train_loss',  # 要绘制的指标名称
                             # 使用'auto'绘制所有检测到的指标
    vmin=0.1,                # 轮廓级别的最小值
    vmax=10,                 # 轮廓级别的最大值
    vlevel=0.5,              # 轮廓级别之间的间距
    show=False               # 是否显示图表
)
```

**参数：**
- `surf_name`：
  - `'train_loss'`（默认值）：绘制标准损失（向后兼容）
  - `'auto'`：自动检测并绘制所有指标（新增！）
  - `'train_loss_metric_name'`：绘制特定的自定义指标

**输出文件：**
- `*_2dcontour.pdf`：轮廓线
- `*_2dcontourf.pdf`：填充轮廓
- `*_2dheat.pdf`：热力图

#### `plot_2d_surface()`
绘制3D曲面可视化。

```python
landscape.plot_2d_surface(
    surf_name='train_loss',  # 要绘制的指标名称
                             # 使用'auto'绘制所有检测到的指标
    show=False               # 是否显示图表
)
```

**输出文件：**
- `*_3dsurface.pdf`：3D曲面图

#### `export_paraview()`
导出为ParaView VTP格式。

```python
vtp_file = landscape.export_paraview(
    surf_name='train_loss',  # 导出哪个曲面
    log=False,               # 使用对数尺度
    zmax=-1,                 # 裁剪最大z值（-1：不裁剪）
    interp=-1                # 插值分辨率（-1：不插值）
)
```

## 关键概念

### 标准模式 vs 自定义模式

| 方面 | 标准模式 | 自定义模式 |
|------|---------|-----------|
| Criterion | `nn.Module`（损失函数） | `callable`函数 |
| 返回值 | 单个损失+准确率 | 指标字典 |
| 例子 | `nn.CrossEntropyLoss()` | `def my_metrics(...)` |
| 输出键 | `'train_loss'`、`'train_acc'` | `'train_loss_metric1'`、`'train_loss_metric2'` |
| 绘图 | 固定为`'train_loss'` | 自动检测所有指标 |
| 向后兼容 | ✅ 是（原始行为） | ✅ 是（新增功能） |

### 方向类型

- **`weights`**：权重空间中的方向（包括所有参数）
- **`states`**：state_dict空间中的方向（包括BN运行统计）

使用`weights`进行通用分析，当分析BatchNorm层时使用`states`。

### 标准化方法

- **`filter`**：在过滤器级别标准化（推荐）
  - 每个过滤器与原始权重的范数相同
  - 适用于卷积网络
- **`layer`**：在层级别标准化
- **`weight`**：按权重幅度缩放
- **`dfilter`**：每个过滤器的单位范数
- **`dlayer`**：每层的单位范数

### 忽略选项

- **`biasbn`**：忽略偏差和批归一化参数
  - 将其方向分量设置为零
  - 建议用于大多数分析

## 模块结构

```
loss_landscape_core/
├── core/                  # 核心功能
│   ├── direction.py       # 方向生成和标准化
│   ├── evaluator.py       # 损失和准确率评估
│   └── perturbation.py    # 权重扰动
├── utils/                 # 工具
│   ├── storage.py         # HDF5文件I/O
│   └── projection.py      # 投影和角度计算
├── viz/                   # 可视化
│   ├── plot_1d.py         # 1D绘图
│   ├── plot_2d.py         # 2D绘图（支持多指标）
│   └── paraview.py        # ParaView导出
├── api.py                 # 高级API（支持自定义指标）
├── __init__.py            # 包初始化
└── README.md              # 本文件
```

## 高级使用

### 使用目标方向（模型间插值）

可视化两个训练模型之间的损失地形：

```python
# 用不同超参数训练两个模型
model1 = train_model(lr=0.1, batch_size=128)
model2 = train_model(lr=0.01, batch_size=256)

# 创建地形
landscape = LossLandscape(model1, dataloader)

# 计算从model1到model2的地形
result = landscape.compute_1d(
    directions='target',
    target_model=model2,
    xrange=(0, 1, 51)  # 0 = model1, 1 = model2
)
```

### 多个自定义指标示例

```python
def compute_comprehensive_metrics(net, dataloader, use_cuda):
    """计算多个损失函数和指标。"""

    criterion_ce = nn.CrossEntropyLoss()
    criterion_smooth_l1 = nn.SmoothL1Loss()
    criterion_mse = nn.MSELoss()

    total_ce = 0.0
    total_smooth_l1 = 0.0
    total_mse = 0.0
    total_correct = 0
    total_samples = 0

    net.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)

            # 多个损失函数
            ce_loss = criterion_ce(outputs, targets)
            targets_onehot = torch.nn.functional.one_hot(
                targets, num_classes=outputs.size(1)).float()
            smooth_l1_loss = criterion_smooth_l1(outputs, targets_onehot)
            mse_loss = criterion_mse(outputs, targets_onehot)

            # 累加
            total_ce += ce_loss.item() * inputs.size(0)
            total_smooth_l1 += smooth_l1_loss.item() * inputs.size(0)
            total_mse += mse_loss.item() * inputs.size(0)

            # 准确率
            _, pred = outputs.max(1)
            total_correct += (pred == targets).sum().item()
            total_samples += inputs.size(0)

    return {
        'ce_loss': total_ce / total_samples,
        'smooth_l1': total_smooth_l1 / total_samples,
        'mse_loss': total_mse / total_samples,
        'accuracy': 100.0 * total_correct / total_samples
    }

landscape = LossLandscape(model, dataloader,
                         criterion=compute_comprehensive_metrics)

# 计算和可视化所有指标
result = landscape.compute_2d(xrange=(-1, 1, 21), yrange=(-1, 1, 21))
landscape.plot_2d_contour(surf_name='auto')  # 生成16个PDF文件！
```

### 保存和加载曲面

计算的损失曲面自动保存为HDF5文件：

```python
# 加载之前计算的曲面
import h5py

f = h5py.File('loss_surface.h5', 'r')
print(f.keys())

# 标准模式键：['xcoordinates', 'ycoordinates', 'train_loss', 'train_acc']
# 自定义模式键：['xcoordinates', 'ycoordinates', 'train_loss_metric1', 'train_loss_metric2', ...]

losses = f['train_loss'][:]
f.close()
```

### 直接低级函数

为获得更多控制，直接使用核心模块：

```python
from loss_landscape_core.core import direction, evaluator, perturbation
from loss_landscape_core import utils

# 创建方向
d = direction.create_random_direction(model, dir_type='weights')

# 扰动权重
original_weights = direction.get_weights(model)
perturbation.set_weights(model, original_weights, [d], step=0.5)

# 评估
loss, acc = evaluator.eval_loss(model, criterion, dataloader, use_cuda=True)

# 恢复
perturbation.set_weights(model, original_weights)
```

## 输出文件

库生成以下输出文件：

### HDF5数据文件（`.h5`）

**标准模式**（使用nn.Module criterion）：
```
键：
  - xcoordinates, ycoordinates: 坐标
  - train_loss: 训练损失值
  - train_acc: 训练准确率
```

**自定义模式**（使用callable criterion）：
```
键：
  - xcoordinates, ycoordinates: 坐标
  - train_loss_ce_loss: CE损失值
  - train_loss_smooth_l1: Smooth L1损失值
  - train_loss_accuracy: 准确率值
  - （每个自定义指标的更多内容）
```

### PDF可视化文件

为每个指标生成4个文件：

```
*_train_loss_metricname_2dcontour.pdf    # 轮廓线
*_train_loss_metricname_2dcontourf.pdf   # 填充轮廓
*_train_loss_metricname_2dheat.pdf       # 热力图
*_train_loss_metricname_3dsurface.pdf    # 3D曲面
```

**3个指标示例：**
```
共8个文件：
  - 指标1的4个文件
  - 指标2的4个文件
  - 指标3的4个文件
```

### ParaView文件（`.vtp`）

与ParaView兼容的VTK格式，用于专业渲染：
```
loss_surface.h5_train_loss_ce_loss.vtp
loss_surface.h5_train_loss_accuracy.vtp
```

## 示例脚本

提供了两个示例脚本：

### 1. `example_custom_metrics.py`

演示：
- 标准损失模式（向后兼容）
- 1D地形与自定义指标
- 2D地形与自定义指标
- 所有指标的自动绘图

运行：
```bash
python example_custom_metrics.py
```

### 2. `test_2d_landscape_fast_multi_metrics.py`

快速2D地形演示，带优化：
- CIFAR-10上的ResNet56（1/10数据子集）
- 11×11网格（vs标准的21×21）
- 3个指标：CE损失、Smooth L1损失、准确率
- 约2-3分钟运行时间
- 自动生成12个可视化

运行：
```bash
python test_2d_landscape_fast_multi_metrics.py
```

## 获得更好结果的提示

1. **使用标准化数据**：确保数据加载器使用标准化数据（对有意义的损失值很重要）

2. **充分采样**：每个维度至少使用51个点（2D为51×51）以捕获曲面特征

3. **适当的损失范围**：调整轮廓图中的`vmin`/`vmax`以突出有趣的特征

4. **对数尺度**：在ParaView导出中对具有大动态范围的地形使用`log=True`

5. **分辨率**：对于出版质量的图形，导出到ParaView并以更高分辨率渲染

6. **自定义指标**：要获得自定义指标的最佳结果：
   - 确保在所有数据点上一致计算指标
   - 谨慎使用批归一化（考虑`dir_type='states'`）
   - 在计算指标之前标准化输出

## 性能考虑

### 计算时间

- **1D地形**：O(num_points) - 按采样点线性
- **2D地形**：O(num_points_x × num_points_y) - 二次方
- 每个点需要通过所有数据的一次前向传递

### 内存使用

- 存储2D地形：~4 MB每个指标（21×21 float32数组）
- 多个指标线性相加

### GPU要求

- 支持任何带CUDA支持的NVIDIA GPU
- 所需GPU内存：~2×模型 + 1×批大小
- 如果CUDA不可用，自动回退到CPU

## 故障排除

### CUDA内存不足

- 减小DataLoader中的批大小
- 使用较小的数据集子集
- 使用`use_cuda=False`以CPU模式运行

### 大动态范围

在ParaView导出中使用`log=True`以对数尺度可视化

### 缺少指标

确保自定义指标函数一致地返回所有键：

```python
def my_metrics(net, dataloader, use_cuda):
    # 必须每次返回相同键的字典
    return {
        'loss': loss_val,
        'accuracy': acc_val
    }  # 每次相同的键
```

## 引用

如果在研究中使用本库，请引用原始工作：

```bibtex
@inproceedings{li2018visualizing,
  title={Visualizing the Loss Landscape of Neural Nets},
  author={Li, Hao and Xu, Zheng and Taylor, Gavin and Studer, Christoph and Goldstein, Tom},
  booktitle={Neural Information Processing Systems},
  year={2018}
}
```

## 许可证

MIT许可证 - 见原始[loss-landscape](https://github.com/zingyi-li/Loss-Surfaces)仓库

## 参考资料

- 原始论文：[Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913)
- 原始代码：https://github.com/zingyi-li/Loss-Surfaces
- PyTorch文档：https://pytorch.org/docs/
- ParaView：https://www.paraview.org/

## 最新动态

### 最近添加

- **自定义指标支持**：用自定义函数定义和计算多个指标
- **自动检测**：自动检测并绘制所有计算的指标
- **类型安全API**：智能criterion类型检测（无需标志）
- **增强绘图**：`plot_2d_contour()`和`plot_2d_surface()`现在支持`surf_name`参数
- **更好的文档**：全面的示例和API参考

### 向后兼容性

所有更改完全向后兼容：
- 使用`nn.CrossEntropyLoss()`的现有代码继续工作
- 默认行为不变
- 新功能是可选的

## 支持

如有问题、疑问或建议：
1. 检查示例脚本
2. 查阅API参考
3. 查看原始论文和代码
