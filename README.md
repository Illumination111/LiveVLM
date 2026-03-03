# LiveVLM
LiveVLM: Efficient Online Video Understanding via Streaming-Oriented KV Cache and Retrieval (DAC'26)

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/sjtu-zhao-lab/LiveVLM.git
cd LiveVLM
```

### 2. Create the environment
```bash
conda create -n livevlm python=3.10 -y
conda activate livevlm

pip install -U torch==2.1.2 torchvision==0.16.2 
pip install numpy==1.26.1
pip install flash-attn==2.7.3 --no-build-isolation --no-cache-dir
pip install -e .
```

## Example Evaluation of MLVU
Update the corresponding arguments in `run_eval.sh`, then execute the following command:
```bash
bash benchmarks/mlvu/scripts/run_eval.sh
```

## Citation
If you find LiveVLM useful in your research, please consider citing our work.

## Acknowledgements
Our code is based on [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) and [ReKV](https://github.com/Becomebright/ReKV).