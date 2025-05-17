# FINSABER

<img src="https://github.com/waylonli/FINSABER/blob/main/figs/framework.png" width="500">

## 1. Environment Setup

```bash
conda create -n finsaber python=3.10
conda activate finsaber
pip install -r requirements.txt
```

Rename `.env.example` to `.env` and set the environment variables. 
- `OPENAI_API_KEY` is required to run LLM-based strategies. 
- `HF_ACCESS_TOKEN` is optional.



