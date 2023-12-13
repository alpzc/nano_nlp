# PYTORCH WITH CUDA SUPPORT INSTALLATION ON UBUNTU 22.04

## Installing Miniconda and Jupyter Notebook
1. Download the latest version of Miniconda from [here](https://docs.conda.io/projects/miniconda/en/latest/#latest-miniconda-installer-links). You can also opt to install Anaconda as well from [here](https://docs.anaconda.com/free/anaconda/install/linux/).

    - For Miniconda, download the file with the extension ```.sh```.
    - For Anaconda, download the file with the extension ```.sh``` or ```.bash```.
    - Follow the instructions on their respective websites to install them.

2. Follow the instructions on the screen. When asked to add Miniconda or Anaconda to the PATH, answer yes. This is necessary to be able to use conda commands in the terminal itself.

3. Close the terminal and open it again. Check if Miniconda or Anaconda was installed correctly by executing the following command:
    ```bash
    conda --version
    ```
    - Output should be <ins> similar</ins> to this:
        ```
        conda 4.10.3
        ```

4. Install jupyter notebook:
    ```bash
    conda install -y jupyter
    ```
    - To see that jupyter notebook was installed correctly, execute the following command:
        ```bash
        jupyter notebook
        ```

5. Create a conda environment (for this, we will use Python 3.8, but any other version can be used as long as it is [compatible with PyTorch](https://pytorch.org/get-started/locally/)):
    ```bash
    conda create -n env_name python=3.8
    ```
6. Activate the environment by restarting terminal and executing the following command:
    ```bash
    conda activate env_name
    ```
7. To link jupyter notebook to the environment, execute the following command:
    ```bash
    conda install nb_conda
    ```
## Installing Pytorch with CUDA support
1. Ensure you have an Nvidia driver installed. To check this, execute the following command:
    ```bash
    nvidia-smi
    ```

    - The first lines of the output will look <ins> similar</ins> to this:
    ```
    NVIDIA-SMI 535.129.03    Driver Version: 535.129.03   CUDA Version: 12.2
    ```

    This output says that the *maximum version of CUDA that can be supported is 12.2* and the current version of the Nvidia Drivers is 535.129.03. More information about these [here](https://stackoverflow.com/questions/53422407/different-cuda-versions-shown-by-nvcc-and-nvidia-smi). The output of ```nvidia-smi``` **DOES NOT** mean that you have CUDA 12.0 installed, but that the Nvidia Drivers installed are compatible with CUDA 12.0.3, as mentioned [in this discussion](https://stackoverflow.com/questions/60987997/why-torch-cuda-is-available-returns-false-even-after-installing-pytorch-with)
    - Additional information about compatibilities between CUDA (cudatoolkit), Nvidia Drivers, and Pytorch can be found in the following links:
        - [Wikipedia page on CUDA and Nvidia Architechture](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)
        -  [CUDA compatibility with Nvidia Drivers](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)

2. Go to the [PyTorch Versions website](https://pytorch.org/get-started/previous-versions/) to see the instructions of how to install the different versions of Pytorch bundled with the compatible cudatoolkit. In the example case, given the `nvidia-smi` output, we will install the version of PyTorch compatible with CUDA 12.1. Keep in mind that the version of CUDA Toolkit installed must be compatible with the PyTorch version installed and should also comply with <ins>your</ins> output of ```nvidia-smi```. From the page, the command we want is:
    ```bash
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    ```

## Registering the environment in Jupyter Notebook
1. Execute the following command, which will install the kernel for the environment in Jupyter Notebook:
    ```bash
    python -m ipykernel install --user --name env_name --display-name "Python 3.8 (env_name)"
    ```
2. Open Jupyter Notebook and check if the environment is available in the kernel options.

## Testing the Installation of Pytorch with GPU support
1. Open Jupyter Notebook and create a new notebook using the environment created.
2. Execute the following code:
    ```python
    import torch
    import sys 
    import pandas as pd
    import sklearn as sk

    print(f"PyTorch Version: {torch.__version__}")
    print(f"Python {sys.version}")
    print(f"Pandas {pd.__version__}")
    print(f"Scikit-Learn {sk.__version__}")
    print("CUDA is available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))
    ```
3. If everything was installed correctly, the output should be <ins> similar</ins> to this:
    ```
    PyTorch Version: 1.9.0
    Python 3.8.10 | packaged by conda-forge | (default, May 11 2021, 06:25:23)
    [GCC 9.3.0]
    Pandas 1.3.0
    Scikit-Learn 0.24.2
    CUDA is available: True
    CUDA device count: 1
    CUDA device name: NVIDIA GeForce GTX 1050 Ti
    ```


