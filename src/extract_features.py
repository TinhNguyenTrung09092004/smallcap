import os
import argparse
import json
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from transformers import logging
from transformers import AutoImageProcessor, CLIPVisionModel

logging.set_verbosity_error()


class ImageDataset(Dataset):
    def __init__(self, items, feature_extractor):
        self.items = items
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(item['path']).convert('RGB')
        pixel_values = self.feature_extractor(image, return_tensors='pt').pixel_values[0]
        return pixel_values, item['cocoid']


def load_data(annotations_path, train_images_dir, val_images_dir):
    annotations = json.load(open(annotations_path))['images']
    image_dirs = {'train2014': train_images_dir, 'val2014': val_images_dir}
    data = {'train': [], 'val': []}

    for item in annotations:
        entry = {'path': os.path.join(image_dirs[item['filepath']], item['filename']),
                 'cocoid': str(item['cocoid'])}
        if item['split'] == 'train' or item['split'] == 'restval':
            data['train'].append(entry)
        elif item['split'] == 'val':
            data['val'].append(entry)

    return data


def encode_split(items, split, args, feature_extractor, clip_encoder, device):
    out_path = os.path.join(args.features_dir, '{}.hdf5'.format(split))
    h5py_file = h5py.File(out_path, 'a')

    todo = [item for item in items if item['cocoid'] not in h5py_file]
    print('{}: {} images total, {} left to encode'.format(split, len(items), len(todo)))

    if len(todo) == 0:
        h5py_file.close()
        return

    dtype = 'float16' if args.fp16 else 'float32'
    loader = DataLoader(ImageDataset(todo, feature_extractor), batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=False)

    for pixel_values, cocoids in tqdm(loader, desc=split):
        with torch.no_grad():
            encodings = clip_encoder(pixel_values=pixel_values.to(device)).last_hidden_state.cpu().numpy()
        for cocoid, encoding in zip(cocoids, encodings):
            h5py_file.create_dataset(cocoid, (50, 768), data=encoding, dtype=dtype)
        h5py_file.flush()

    h5py_file.close()


def main(args):
    os.makedirs(args.features_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor = AutoImageProcessor.from_pretrained(args.encoder_name)
    clip_encoder = CLIPVisionModel.from_pretrained(args.encoder_name).to(device)
    clip_encoder.eval()

    data = load_data(args.annotations_path, args.train_images_dir, args.val_images_dir)

    for split in args.splits:
        encode_split(data[split], split, args, feature_extractor, clip_encoder, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract CLIP visual features')
    parser.add_argument("--annotations_path", type=str, default="data/dataset_coco.json", help="JSON file with annotations in Karpathy splits")
    parser.add_argument("--train_images_dir", type=str, default="data/images/train2014", help="Directory holding the COCO train2014 images")
    parser.add_argument("--val_images_dir", type=str, default="data/images/val2014", help="Directory holding the COCO val2014 images")
    parser.add_argument("--features_dir", type=str, default="features/", help="Directory where extracted features are written")

    parser.add_argument("--encoder_name", type=str, default="openai/clip-vit-base-patch32", help="Encoder name as found on HuggingFace or stored locally")
    parser.add_argument("--splits", type=str, nargs='+', default=['train', 'val'], choices=['train', 'val'], help="Which splits to encode")

    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers used to decode images")
    parser.add_argument("--fp16", action="store_true", default=False, help="Store features as float16 to halve disk usage")

    args = parser.parse_args()

    main(args)
