# Chạy SmallCap trên Kaggle (T4 x2)

| # | Notebook | Output cần lưu |
|---|---|---|
| 1 | Extract features | `features/train.hdf5`, `features/val.hdf5` |
| 2 | Retrieve captions | `retrieved_caps_resnet50x64.json`, `coco_index` |
| 3 | Train + Infer + Eval | `experiments/rag_7M/` (~9.3 GB), `test_preds.json`, `test_res.txt` |

---

# Notebook 1 — Extract features

```python
!git clone -b kaggle https://github.com/TinhNguyenTrung09092004/smallcap.git /kaggle/working/smallcap
%cd /kaggle/working/smallcap
```

```python
!python src/extract_features.py \
    --annotations_path "/kaggle/input/datasets/shtvkumar/karpathy-splits/dataset_coco.json" \
    --train_images_dir "/kaggle/input/datasets/nadaibrahim/coco2014/train2014/train2014" \
    --val_images_dir   "/kaggle/input/datasets/nadaibrahim/coco2014/val2014/val2014" \
    --features_dir /kaggle/working/features \
    --batch_size 256 --num_workers 4
```

=> `features/train.hdf5`, `features/val.hdf5` (18.2 GB fp32)

---

# Notebook 2 — Retrieve captions

```python
!pip install -q faiss-cpu
!pip install -q git+https://github.com/openai/CLIP.git
```

```python
!python src/retrieve_caps.py \
    --annotations_path "/kaggle/input/datasets/shtvkumar/karpathy-splits/dataset_coco.json" \
    --train_images_dir "/kaggle/input/datasets/nadaibrahim/coco2014/train2014/train2014" \
    --val_images_dir   "/kaggle/input/datasets/nadaibrahim/coco2014/val2014/val2014" \
    --datastore_dir /kaggle/working/datastore \
    --captions_path /kaggle/working/data/retrieved_caps_resnet50x64.json
```


Output: `retrieved_caps_resnet50x64.json`, `coco_index` 

---

# Notebook 3 — Train + Infer + Eval

Chạy liền một mạch trong cùng một notebook: train ghi thẳng vào `/kaggle/working/experiments`,
là thư mục ghi được, nên bước infer đọc luôn tại chỗ, không cần lưu dataset trung gian.

Tổng thời gian ~7 h train + ~1 h infer, vừa với giới hạn 12 h. Nếu sợ sát giờ thì hạ
`--n_epochs`.

### Cell 1 — lấy mã nguồn và cài Java

Java chỉ cần cho METEOR và PTBTokenizer; hai file `.jar` đó đã có sẵn trong repo nên không
phải chạy `get_stanford_models.sh` (script đó chỉ phục vụ SPICE, đã bỏ).

```python
!git clone -b kaggle https://github.com/TinhNguyenTrung09092004/smallcap.git /kaggle/working/smallcap
%cd /kaggle/working/smallcap
!apt-get -qq install -y openjdk-11-jre-headless
```

### Cell 2 — train

```python
!python train.py \
    --features_dir /kaggle/input/datasets/nguyntrungtnh/ntt-smallcap \
    --annotations_path "/kaggle/input/datasets/shtvkumar/karpathy-splits/dataset_coco.json" \
    --captions_path /kaggle/input/datasets/nguyntrungtnh/ntt-smallcap/retrieved_caps_resnet50x64.json \
    --experiments_dir /kaggle/working/experiments \
    --n_epochs 10 --batch_size 32
```

`--batch_size 32` chứ không phải 64: tham số này là `per_device_train_batch_size`, `Trainer`
bật `DataParallel` trên 2 GPU nên batch thực tế thành 64 — đúng con số paper.

### Cell 3 — chọn checkpoint cuối

Số bước không biết trước, nên lấy bằng code thay vì gõ tay:

```python
import glob, os, re
MODEL_DIR = '/kaggle/working/experiments/rag_7M'
ckpts = glob.glob(MODEL_DIR + '/checkpoint-*')
CKPT = os.path.basename(max(ckpts, key=lambda p: int(re.search(r'checkpoint-(\d+)$', p).group(1))))
print(CKPT)
```

### Cell 4 — infer trên split test

Split test đọc ảnh trực tiếp từ `val2014` vì notebook 1 chỉ trích đặc trưng cho train và val.

```python
!python infer.py \
    --model_path {MODEL_DIR} \
    --checkpoint_path {CKPT} \
    --annotations_path "/kaggle/input/datasets/shtvkumar/karpathy-splits/dataset_coco.json" \
    --captions_path /kaggle/input/datasets/nguyntrungtnh/ntt-smallcap/retrieved_caps_resnet50x64.json \
    --images_dir "/kaggle/input/datasets/nadaibrahim/coco2014/val2014/val2014/" \
    --infer_test
```

`--images_dir` phải có dấu `/` ở cuối, [infer.py](../infer.py) nối chuỗi trực tiếp chứ không
dùng `os.path.join`. Luôn truyền `--checkpoint_path`: bỏ trống thì nó chạy hết cả 10
checkpoint, mỗi cái ~1 h.

### Cell 5 — đánh giá

```python
!python coco-caption/run_eval.py \
    coco-caption/annotations/captions_testKarpathy.json \
    {MODEL_DIR}/{CKPT}/test_preds.json
!cat {MODEL_DIR}/{CKPT}/test_res.txt
```
