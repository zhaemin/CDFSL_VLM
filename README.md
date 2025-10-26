# CDFSL_VLM
##  Install
```bash
cd cdfsl_attr
conda create -n <name> python=3.10.8
pip install -r deps/requirements.txt
pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 torchaudio==2.8.0 --extra-index-url https://download.pytorch.org/whl/cu126
```

##  Execution

### 1. LayerNorm Tuning
Run the script below to fine-tune only the LayerNorm parameters:
```bash
bash scripts/run_ln_only.sh
```

### 2. Attribute-based Method
Run the script below to execute the attribute-based adaptation approach:
```bash
bash scripts/run_ln_attr.sh
```
