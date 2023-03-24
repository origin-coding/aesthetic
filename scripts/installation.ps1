# Create anaconda virtual environment
conda create -n aesthetic python=3.10
conda activate aesthetic

# Install pandas and jupyter notebook
conda install -c conda-forge pandas jupyter

# Install PyTorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch  # Use CPU
# conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia  # Use CUDA 11.7
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia  # Use CUDA 11.8

# Install PySide6
pip install pyside6

# Install scikit-learn
conda install scikit-learn

# Install matplotlib and seaborn
conda install seaborn
