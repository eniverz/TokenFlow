# Setup Environment

First, create a conda environment:

```bash
conda create -n tokenflow python==3.11 -y
conda activate tokenflow
```

For deepspeed and some operators, you need to install cuda toolkit. For torchcodec as video backend, you need to install ffmpeg. You can install them by conda:
```bash
conda install -c nvidia/label/cuda-12.8.0 -c conda-forge cuda-toolkit
conda install -c conda-forge "ffmpeg=6.1"
```

Then, install dependencies:

```bash
pip install -r requirements.txt
```

> If cannot use flash-attn or causal-conv1d, please install and compile by yourself:
> ```bash
> pip install ninja packaging psutil
> MAX_JOBS=8 pip install flash-attn==2.8.3 --no-build-isolation
> MAX_JOBS=8 pip install causal-conv1d==1.7.0 --no-build-isolation
> ```

