# Aesthetic

毕业论文《基于深度学习的图像美学质量评价系统设计与实现》代码部分

## 运行环境

1. 操作系统：Ubuntu 20.04
2. CPU：Xeon(R) Gold 6130，6 CPU cores
3. GPU：NVIDIA V00 32GB显存
4. 内存：25 GB

## 用到的库

- PyTorch +
  ignite：用于模型的训练工作，使用ignite简化代码编写，[PyTorch](https://pytorch.org) [PyTorch-Ignite](https://pytorch-ignite.ai)
- PySide6 + QML：用于编写用户界面和交互逻辑，[Qt for Python](https://doc.qt.io/qtforpython-6/)
- Pandas + Jupyter Notebook + Seaborn + Matplotlib + scikit-learn：用于数据分析和预处理

## 依赖安装

```shell
# Create anaconda virtual environment
conda create -n aesthetic python=3.10
conda activate aesthetic

# Dependencies for PyTorch and Ignite
conda install pytorch torchvision torchaudio cpuonly -c pytorch                        # Use CPU
# conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia  # Use CUDA 11.7
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia  # Use CUDA 11.8
conda install ignite -c pytorch

# Dependencies for data analyzing
conda install scikit-learn seaborn pandas jupyter -c conda-forge

# Other dependencies
pip install pyside6 BeautifulReport click pydantic loguru tensorboardX tensorboard
```

## 数据集及模型权重

本文使用的数据集以及训练好的模型权重文件如下：[分享链接](https://pan.baidu.com/s/1CtrsivRk3dOUEUNJPzHJjA?pwd=ew77)

### 数据集

本文使用的数据集来自三个数据集经过处理之后得到的结果：
1. AVA数据集：随机选取10000张图片，并保存其对应的标签
2. AADB数据集：随机选取10000张图片，并保存其对应的标签
3. CUHK-PQ数据集：在HighQuality和LowQuality中各自随机选取5000张图片，共计10000张

### 模型权重

在启动桌面应用程序时需要使用到训练好的模型权重文件，现对其命名规则进行说明。

| 是否使用Attention | Kernel Size | 是否使用DWA | 文件名称   |
|---------------|-------------|---------|--------|
| ✖             | 3           | ✖       | 030.pt |
| ✖             | 5           | ✖       | 050.pt |
| ✔             | 3           | ✖       | 130.pt |
| ✔             | 3           | ✔       | 131.pt |
| ✔             | 5           | ✖       | 150.pt |
| ✔             | 5           | ✔       | 151.pt |

如果需要使用自己训练的模型，那么请按照上述规则保存模型文件，并放置在*pretrained*文件夹中

## 最终目录结构

```text
.
+---common     # 存放一些公用的操作和变量
+---data       # 数据被放在这里
+---datasets   # 数据集的定义
+---models     # 模型和损失函数的定义
+---notebooks  # 存放了数据分析的一些Jupyter Notebooks
+---outputs    # 训练和测试过程的输出
|   +---checkpoints
|   \---logs
+---pretrained # 将预训练的模型放在这里，用与桌面应用程序
+---tests      # 一些单元测试，用以检验程序能否正常运行
+---train      # 训练模型的代码
\---ui         # 应用封装的界面
```

## 运行应用

```shell
python main.py test   # 启动测试流程，测试代码能否正常运行

python main.py train  # 训练模型

python main.py run    # 启动桌面应用程序
```

其中训练模型需要用到的的参数放在了*config.json*文件中，具体含义参见*train/config.py*文件。
