# Installation

Use the existing LingBotVA environment when available:

```bash
export PATH=/workspace/basics/miniconda3/envs/lingbotVA/bin:$PATH
```

Install Python dependencies if the environment is not already prepared:

```bash
pip install -r requirements.txt
```

FA4 training requires the local FlashAttention-4 CuTe package to be importable
on H100/SM90 machines.  If it is not installed yet:

```bash
pip install -e dependencies/flash-attention/flash_attn/cute[dev]
```

Build or select the prepared MOT dataset, then launch training:

```bash
bash 1shell/train_mot_mixed_8gpu.sh
```
