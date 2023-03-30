# Aesthetic
基于深度学习的图像美学质量评价系统

## 各种库的版本
- PyTorch 2.0.0：用于神经网络的训练和模型的保存
- PySide6 6.4.2：用于实现界面和用户交互逻辑
- pandas + Jupyter Notebook：用于进行数据预处理和数据分析
- seaborn + matplotlib：用于进行数据分析及图标绘制
- scikit-learn：用于数据预处理，将AADB的美学标签进行二值化处理

## Installation
```shell
# Create anaconda virtual environment
conda create -n aesthetic python=3.10
conda activate aesthetic

conda install pytorch torchvision torchaudio cpuonly -c pytorch  # Use CPU
# conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia  # Use CUDA 11.7
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia  # Use CUDA 11.8

conda install scikit-learn, seaborn, pandas jupyter -c conda-forge

pip install pyside6, BeautifulReport, click, pydantic
```
