# Dataset Contents

This dataset contains a zipped folder `data.zip` with the data under `dataset/` (images and labels in YOLO format).

## Usage (Kaggle Notebook)
```python
import zipfile, os
zip_path = '/kaggle/input/'  # auto-mounted datasets live here
# Replace <owner>-<slug> with your dataset slug used on Kaggle

ds_root = '/kaggle/input/<owner>-<slug>'
zip_file = os.path.join(ds_root, 'data.zip')
extract_to = '/kaggle/working'
with zipfile.ZipFile(zip_file, 'r') as zf:
    zf.extractall(extract_to)

!ls -R /kaggle/working/dataset/
```

Then point your training config (e.g., Ultralytics YOLO) to the extracted `dataset/` path.