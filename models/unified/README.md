# Unified Fusion Artifacts

- `fusion_config.json`: weighted score fusion policy used by `python main.py infer unified`.

Update this config through:

```powershell
python main.py train unified --interaction-weight 0.55 --transaction-weight 0.45 --high-threshold 0.70 --medium-threshold 0.40
```
