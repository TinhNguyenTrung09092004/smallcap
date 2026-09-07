# Chạy SmallCap trên Kaggle (T4 x2)

| # | Notebook | GPU | Internet | Thời gian ước tính | Output cần lưu |
|---|---|---|---|---|---|
| 1 | Extract features | T4 x1 | Bật | ~1.5–2 h | `features/train.hdf5`, `features/val.hdf5` (18.2 GB fp32) |
| 2 | Retrieve captions | T4 x1 | Bật | ~3–5 h | `data/retrieved_caps_resnet50x64.json` (~250 MB) |
| 3 | Train | T4 x2 | Bật | ~40 min / epoch | `experiments/rag_7M/checkpoint-*/` (~9.3 GB cho 10 epoch) |
| 4 | Infer + Eval | T4 x1 | Bật | ~1 h | `test_preds.json`, `test_res.txt` |

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

### Cell 5 — kiểm tra kết quả

```python
import h5py
for split in ['train', 'val']:
    f = h5py.File(f'/kaggle/working/features/{split}.hdf5', 'r')
    print(split, len(f.keys()), 'images', f[list(f.keys())[0]].shape, f[list(f.keys())[0]].dtype)
    f.close()
!du -sh /kaggle/working/features
```

Kỳ vọng: train 113.287 ảnh (train + restval), val 5.000 ảnh, shape `(50, 768)`, dtype `float32`.

### Lưu dataset

Save Version → **Save & Run All (Commit)**. Xong vào tab Output → **New Dataset**,
đặt tên ví dụ `smallcap-features`. Notebook 3 sẽ add dataset này.

Xoá bớt thứ không cần trước khi commit để output nhẹ:

```python
!rm -rf /kaggle/working/smallcap
```

---

# Notebook 2 — Retrieve captions

Settings: **GPU T4**, **Internet = On**. Chạy song song với notebook 1 được.

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
    --captions_path /kaggle/working/data/retrieved_caps_resnet50x64.json \
    --batch_size 64
```

Đây là bước tốn thời gian nhất. CLIP RN50x64 nhận ảnh 448px và nặng hơn ViT-B/32 rất nhiều,
phải encode cả 123.287 ảnh — dự kiến 3–5 giờ trên T4. Script này **không** resume được,
nếu quá 12 giờ thì phải chia theo dải ảnh.

`faiss-cpu` đủ dùng: `IndexFlatIP` chạy trên BLAS, tìm 123k query trên ~566k caption mất
khoảng 25–40 phút. `faiss-gpu` trên Kaggle hay lỗi bánh xe cài đặt.

Output cần lưu: `data/retrieved_caps_resnet50x64.json`. File `datastore/coco_index` (~2.3 GB)
chỉ cần nếu sau này bạn muốn truy hồi cho ảnh mới — không cần cho train/eval.

---

# Notebook 3 — Train

Settings: **GPU T4 x2**, **Internet = On**. Add 2 dataset: features (nb 1) + retrieved caps (nb 2).

```python
!python train.py \
    --features_dir /kaggle/input/smallcap-features/features \
    --annotations_path "/kaggle/input/datasets/shtvkumar/karpathy-splits/dataset_coco.json" \
    --captions_path /kaggle/input/smallcap-caps/retrieved_caps_resnet50x64.json \
    --experiments_dir /kaggle/working/experiments \
    --n_epochs 10 --batch_size 32
```

Ba điểm cần biết:

- `--batch_size 32`, **không phải 64**. Tham số này ánh xạ sang
  `per_device_train_batch_size`, mà HF `Trainer` tự bật `DataParallel` khi thấy 2 GPU, nên
  batch thực tế = 32 × 2 = 64 — đúng con số của paper. Nếu để 64 thì batch thực tế thành
  128 và không còn khớp paper nữa.
- Lưu checkpoint mỗi epoch, giữ cả 10 (`save_total_limit=n_epochs`). Mỗi checkpoint ≈ 930 MB
  (model 218M tham số fp32 ≈ 873 MB + optimizer state chỉ cho 7M tham số cross-attention
  ≈ 56 MB), tổng ~9.3 GB cho 10 epoch. Vừa với trần 20 GB.
- CLIP encoder **không** chạy lúc train vì đặc trưng đã trích sẵn và truyền vào qua
  `encoder_outputs`; chỉ GPT-2 chạy. Nên ~40 phút/epoch, 10 epoch ≈ 6–7 giờ, lọt trong một
  session 12 giờ. Xem ETA của tqdm ở epoch đầu để xác nhận.

---

# Notebook 4 — Infer + Eval

Settings: **GPU T4**, **Internet = On**. Cần Java cho METEOR/SPICE.

```python
!apt-get -qq install -y openjdk-11-jre-headless
!bash coco-caption/get_stanford_models.sh
```

Split test lấy ảnh trực tiếp từ `val2014` (notebook 1 không trích đặc trưng cho test):

```python
!python infer.py \
    --model_path /kaggle/input/smallcap-model/experiments/rag_7M \
    --checkpoint_path checkpoint-XXXX \
    --annotations_path "/kaggle/input/datasets/shtvkumar/karpathy-splits/dataset_coco.json" \
    --captions_path /kaggle/input/smallcap-caps/retrieved_caps_resnet50x64.json \
    --images_dir "/kaggle/input/datasets/nadaibrahim/coco2014/val2014/val2014/" \
    --infer_test
```

`--images_dir` phải có dấu `/` ở cuối, [infer.py](../infer.py) nối chuỗi trực tiếp chứ không
dùng `os.path.join`. Checkpoint nằm trong input read-only nên `infer.py` sẽ không ghi được
`test_preds.json` vào đó — copy checkpoint sang `/kaggle/working` trước:

```python
!cp -r /kaggle/input/smallcap-model/experiments /kaggle/working/experiments
```

rồi trỏ `--model_path /kaggle/working/experiments/rag_7M`.

```python
!python coco-caption/run_eval.py \
    coco-caption/annotations/captions_testKarpathy.json \
    /kaggle/working/experiments/rag_7M/checkpoint-XXXX/test_preds.json
!cat /kaggle/working/experiments/rag_7M/checkpoint-XXXX/test_res.txt
```

Kết quả gồm BLEU-1..4, METEOR, ROUGE-L, CIDEr, SPICE.

---

## Tương thích transformers

Mã nguồn gốc viết cho transformers ~4.2x và **không** chạy được trên 5.0.0. Đã port, đây là
những gì đã đổi:

| Chỗ hỏng | Đã sửa thành |
|---|---|
| `CLIPFeatureExtractor` (v5 gỡ toàn bộ lớp `*FeatureExtractor`) | `AutoImageProcessor` |
| `Trainer(tokenizer=...)` | `Trainer(processing_class=...)` |
| `TrainingArguments(overwrite_output_dir=...)` | Bỏ hẳn, v5 không còn field này |
| `_split_heads()`, `_merge_heads()`, `_attn()` của `GPT2Attention` | Viết lại `ThisGPT2Attention.forward` theo `eager_attention_forward` + `ALL_ATTENTION_FUNCTIONS` của v5 |
| `layer_past` dạng tuple | `past_key_values` dạng `EncoderDecoderCache`, có nhánh cache riêng cho cross-attention |
| `class SmallCap(PreTrainedModel)` | `(PreTrainedModel, GenerationMixin)` — từ v4.50 `PreTrainedModel` không còn tự kèm `GenerationMixin`, thiếu là mất `.generate()` |
| `prepare_inputs_for_generation` và `_reorder_cache` tự viết | Xoá, dùng bản mặc định của `GenerationMixin`; v5 `VisionEncoderDecoderModel` cũng không tự viết nữa |
| `from_pretrained` ép `_fast_init=False` | Xoá, v5 không còn tham số này |
| `config.use_return_dict` | Cố định `True` |
| Import `load_tf_weights_in_gpt2`, `SequenceSummary`, `model_parallel_utils` | Xoá, v5 bỏ backend TF và các tiện ích này |

Phần cốt lõi của paper giữ nguyên: cross-attention vẫn rút chiều theo
`cross_attention_reduce_factor`, `head_dim` giảm tương ứng, và `eager_attention_forward`
chia thang theo `value.size(-1) ** 0.5` nên hệ số scale khớp bản gốc.
