# Fruit CNN practice project

Small image-classification exercise (apple / banana / orange) before HVAC symbol detection.

## Folders

| Folder | Purpose |
|--------|---------|
| `data/` | Source photo (`photo1phone.jpg`) for manual crops |
| `dataset/` | Training images: `apple/`, `banana/`, `orange/` |

## Commands (from repo root `Training/`)

```powershell
python projects/fruit/prepare_fruit_crops.py
python projects/fruit/train_fruit_cnn.py
```

Add more `.jpg` / `.png` images under each class folder, then re-run training.
