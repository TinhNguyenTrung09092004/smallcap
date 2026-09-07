import os
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
import clip
import torch
import faiss
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_coco_data(coco_data_path, train_images_dir, val_images_dir):
    """We load in all images and only the train captions."""

    annotations = json.load(open(coco_data_path))['images']
    image_dirs = {'train2014': train_images_dir, 'val2014': val_images_dir}
    images = []
    captions = []
    for item in annotations:
        if item['split'] == 'restval':
             item['split'] = 'train'
        if item['split'] == 'train':
            for sentence in item['sentences']:
                captions.append({'image_id': item['cocoid'],  'caption': ' '.join(sentence['tokens'])})
        images.append({'image_id': item['cocoid'],
                       'path': os.path.join(image_dirs[item['filepath']], item['filename'])})

    return images, captions

def filter_captions(data):

    decoder_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    bs = 512

    image_ids = [d['image_id'] for d in data]
    caps = [d['caption'] for d in data]
    encodings = []
    for idx in range(0, len(data), bs):
        encodings += tokenizer.batch_encode_plus(caps[idx:idx+bs], return_tensors='np', padding=True)['input_ids'].tolist()

    filtered_image_ids, filtered_captions = [], []

    assert len(image_ids) == len(caps) and len(caps) == len(encodings)
    for image_id, cap, encoding in zip(image_ids, caps, encodings):
        if len(encoding) <= 25:
            filtered_image_ids.append(image_id)
            filtered_captions.append(cap)

    return filtered_image_ids, filtered_captions

def encode_captions(captions, model, device):

    bs = 256
    encoded_captions = []

    for idx in tqdm(range(0, len(captions), bs)):
        with torch.no_grad():
            input_ids = clip.tokenize(captions[idx:idx+bs]).to(device)
            encoded_captions.append(model.encode_text(input_ids).cpu().numpy())

    encoded_captions = np.concatenate(encoded_captions)

    return encoded_captions

def encode_images(images, model, feature_extractor, device, bs=64):

    image_ids = [i['image_id'] for i in images]

    image_features = []

    for idx in tqdm(range(0, len(images), bs)):
        image_input = [feature_extractor(Image.open(i['path'])) for i in images[idx:idx+bs]]
        with torch.no_grad():
            image_features.append(model.encode_image(torch.tensor(np.stack(image_input)).to(device)).cpu().numpy())

    image_features = np.concatenate(image_features)

    return image_ids, image_features

def get_nns(captions, images, k=15):
    xq = images.astype(np.float32)
    xb = captions.astype(np.float32)
    faiss.normalize_L2(xb)
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    faiss.normalize_L2(xq)
    D, I = index.search(xq, k)

    return index, I

def filter_nns(nns, xb_image_ids, captions, xq_image_ids):
    """ We filter out nearest neighbors which are actual captions for the query image, keeping 7 neighbors per image."""
    retrieved_captions = {}
    for nns_list, image_id in zip(nns, xq_image_ids):
        good_nns = []
        for nn in nns_list:
            if xb_image_ids[nn] == image_id:
                continue
            good_nns.append(captions[nn])
            if len(good_nns) == 7:
                break
        assert len(good_nns) == 7
        retrieved_captions[image_id] = good_nns
    return retrieved_captions

def main(args):

    os.makedirs(args.datastore_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.captions_path) or '.', exist_ok=True)

    print('Loading data')
    images, captions = load_coco_data(args.annotations_path, args.train_images_dir, args.val_images_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, feature_extractor = clip.load(args.retrieval_encoder, device=device)

    print('Filtering captions')
    xb_image_ids, captions = filter_captions(captions)

    print('Encoding captions')
    encoded_captions = encode_captions(captions, clip_model, device)

    print('Encoding images')
    xq_image_ids, encoded_images = encode_images(images, clip_model, feature_extractor, device, bs=args.batch_size)

    print('Retrieving neighbors')
    index, nns = get_nns(encoded_captions, encoded_images)
    retrieved_caps = filter_nns(nns, xb_image_ids, captions, xq_image_ids)

    print('Writing files')
    faiss.write_index(index, os.path.join(args.datastore_dir, 'coco_index'))
    json.dump(captions, open(os.path.join(args.datastore_dir, 'coco_index_captions.json'), 'w'))

    json.dump(retrieved_caps, open(args.captions_path, 'w'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrieve captions for each image')
    parser.add_argument("--annotations_path", type=str, default="data/dataset_coco.json", help="JSON file with annotations in Karpathy splits")
    parser.add_argument("--train_images_dir", type=str, default="data/images/train2014", help="Directory holding the COCO train2014 images")
    parser.add_argument("--val_images_dir", type=str, default="data/images/val2014", help="Directory holding the COCO val2014 images")

    parser.add_argument("--datastore_dir", type=str, default="datastore/", help="Directory where the FAISS index and its captions are written")
    parser.add_argument("--captions_path", type=str, default="data/retrieved_caps_resnet50x64.json", help="JSON file where retrieved captions are written")

    parser.add_argument("--retrieval_encoder", type=str, default="RN50x64", help="Visual encoder used for retrieving captions")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size used to encode images")

    args = parser.parse_args()

    main(args)
