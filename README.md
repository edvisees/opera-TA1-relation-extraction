# Setup:

```bash
conda env create -f ./conda_env_xiang_ner_bert.yml
```

# Usage:

```bash
source activate ner_bert
cd xiangyang_code/python/
python main_bert.py --ltf ltf_dir --out_folder output_folder
source deactivate
```
