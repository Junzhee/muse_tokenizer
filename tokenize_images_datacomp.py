import os
import numpy as np
import einops
import argparse
from torchvision import transforms
import json
from tqdm import tqdm
import torch
from muse import VQGANModel
from base64 import b64encode
import torch.multiprocessing as mp
import torch.distributed as dist
import json
import os
import time
import random
import braceexpand
import webdataset as wds
import pandas as pd
from multiprocessing import Process, Queue, Manager
from queue import Empty
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['NCCL_DEBUG'] = 'INFO'
os.system("taskset -p 0xffffffffffffffffffffffff %d" % os.getpid())
EXPECTED_CHUNK_SIZE = 10000


def get_dataset(dataset_type, path, s3):
    if s3:
        path = f"pipe:aws s3 cp {path} -"

    if dataset_type == "laion":
        dataset = (
            wds.WebDataset(path)
            .decode(wds.imagehandler("torchrgb"))
            .to_tuple("jpg", "json")
        )
        dataset = dataset.map(remove_keys)

        return dataset
    elif dataset_type == "datacomp":
        dataset = (
            wds.WebDataset(path)
            .decode(wds.imagehandler("torchrgb"))
            .to_tuple("jpg;png;webp", "json", "__key__", "__url__")
        )
        dataset = dataset.map(transform_and_remove_keys)

        return dataset
    elif dataset_type == "mmc4":

        def resize_image(sample):
            keys = ["png", "jpg", "jpeg"]
            for key in keys:
                if key in sample:
                    image = np.array(sample[key].resize((256, 256))).astype(np.float32)
                    image = image.transpose(2, 0, 1) / 255.0
                    sample["image"] = torch.from_numpy(image)
            return sample

        dataset = (
            wds.WebDataset(path)
            .decode("pil")
            .map(resize_image)
            .to_tuple("image", "__key__")
        )
        return dataset


def transform_and_remove_keys(sample):
    image, metadata, key, url = sample

    # image = transforms.functional.resize(image, (224, 224))
    # image = transforms.functional.normalize(image, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    try:
        # CLIP transform without resizing
        image = transforms.functional.resize(image, (224, 224))
        image = transforms.functional.normalize(image, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    except Exception as e:
        print(f"Error processing image: {e}")
        return None
    
    new_dictionary = {}
    new_dictionary['key'] = metadata['key']
    new_dictionary['caption'] = metadata['caption']
    new_dictionary['uid'] = metadata['uid']
    new_dictionary['path'] = url
    return image, new_dictionary, key


def save_batch_to_json(batch_data, output_dir):
    batch_idx, rows = batch_data
    df = pd.DataFrame(rows)
    grouped = df.groupby("path")

    for path, group in grouped:
        basename = os.path.basename(path)
        output_path = os.path.join(output_dir, os.path.splitext(basename)[0] + ".json")

        # Write the new dataframe to a JSON file directly
        if os.path.exists(output_path):
            # Read existing JSON file into a dataframe to append new data
            existing_df = pd.read_json(output_path, orient='records', lines=True)
            updated_df = pd.concat([existing_df, group])
            # Write/overwrite the JSON file with the updated dataframe
            updated_df.to_json(output_path, orient='records', lines=True, index=False)
        else:
            # Write the new dataframe to a JSON file directly
            group.to_json(output_path, orient='records', lines=True, index=False)


def process_chunk(
    rank,
    world_size,
    paths,
    ckpt_path,
    output_dir,
    num_workers,
    batch_size,
    s3,
    dataset_type,
):
    
    net = VQGANModel.from_pretrained(ckpt_path).cuda()
    net.to(rank)  
    print('Model loaded..')

    worker_paths = paths[rank::world_size]
    print (f"Rank: {rank} processing {len(worker_paths)} shards : {worker_paths} ")
    
    dataset = get_dataset(dataset_type, worker_paths, s3)

    dataloader = torch.utils.data.DataLoader(
        dataset, #.batched(batch_size),
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )

    num_chunks = len(worker_paths)

    batch_idx = 0
    seen_keys = set()
    batch_data_list = []
    print("Start tokenizing......")
    
    for batch_data in tqdm(
                dataloader,
                total=int(np.ceil(EXPECTED_CHUNK_SIZE * num_chunks / batch_size)),
                desc=f"Rank : {rank}",
                position=rank,
                leave=False,
            ):

        data, metas, key_ = batch_data
        
        if any(k in seen_keys for k in key_):
            print(f"Rank {rank}: Skipping batch due to duplicate keys: {key_}")
            continue
        seen_keys.update(key_)
   
        try:
            image_tensor = data.to(rank)
        except OSError as e:
            print(f"Skipping batch due to error: {e}")
            continue
        
        data_list = []
        rows = {}

        with torch.no_grad():
            _, tokens = net.encode(image_tensor)
            tokens_save = einops.rearrange(
                tokens.cpu().numpy().astype(np.int32), '(b t) d -> b (t d)', b=tokens.size(0)
            )
            
            for i in range(image_tensor.size(0)):

                data = b64encode(tokens_save[i].tobytes()).decode('utf-8')
                data_list.append(data)

        for k, v in metas.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy().tolist()
            rows[k] = v

        rows["__key__"] = key_
        rows["tokens"] = data_list

        batch_data_list.append((batch_idx, rows))
        batch_idx += 1

        save_batch_to_json((batch_idx, rows), output_dir)


def main():

    parser = argparse.ArgumentParser(description='Tokenize the Dataset')
    parser.add_argument('--dataset_path', type=str, default='/home/niudt/dataset/laion4m', help='path of the laion 2b', required=False)
    parser.add_argument('--save_file', type=str, default='./output_dir', help='Name of the save folder', required=False)
    parser.add_argument('--batch_size', type=int, default=128, help='support 256 for 80G A100', required=False)
    parser.add_argument('--ckpt_path', type=str, default="./ckpt/laion", help='path of the vqgan ckpt', required=False)
    parser.add_argument('--start_shard', type=str, default="0010000", help='Starting shard number', required=True)
    parser.add_argument('--end_shard', type=str, default="0010007", help='Ending shard number', required=True)
    parser.add_argument('--num_workers', type=int, default=1, help='', required=False)
    parser.add_argument('--s3', type=str, default=False, help='', required=False)
    parser.add_argument('--dataset', type=str, default="datacomp", help='', required=False)
    parser.add_argument('--num_gpus', type=int, default=1, help='', required=True)

    args = parser.parse_args()

    start_shard = int(args.start_shard)
    end_shard = int(args.end_shard)
    save_file = args.save_file
    # path_template = f"/p/fastdata/mmlaion/datacomp/datacomp_1B/flat/{{{start_shard}..{end_shard}}}.tar"
    # path_template = f"/p/scratch/laionize/m3_data/datacomp_deduplicated/{{{start_shard}..{end_shard}}}_unique.tar"
    # paths = list(braceexpand.braceexpand(path_template))

    paths = []
    for shard in range(start_shard, end_shard + 1):
        shard_str = f"{shard:07d}"
        json_file = os.path.join(save_file, f"{shard_str}_unique.json")
        if not os.path.exists(json_file):
            tar_file = f"/p/scratch/laionize/m3_data/datacomp_deduplicated/{shard_str}_unique.tar"
            paths.append(tar_file)

    start = time.time()

    if not os.path.exists(save_file):
        os.makedirs(save_file)

    if args.num_gpus > 1:
        mp.spawn(
            process_chunk,
            args=(
                args.num_gpus,
                paths,
                args.ckpt_path,
                save_file,
                args.num_workers,
                args.batch_size,
                args.s3,
                args.dataset,
            ),
            nprocs=args.num_gpus,
        )
    else:
        process_chunk(0, 1, paths, args.ckpt_path, save_file, args.num_workers, args.batch_size, args.s3, args.dataset)

    print(f"Processing {len(paths)} shards took {time.time() - start} seconds")

if __name__ == "__main__":
    main()


# def writer_worker(q, output_dir):
    
#     while True:
#         try:
#             sample = q.get(timeout=100)

#             if sample is None:  # None is the signal to stop.
#                 break
#             rows, embeddings = sample
#             df = pd.DataFrame(rows)

#             grouped = df.groupby("path")
#             for path, group in grouped:
#                 basename = os.path.basename(path)
#                 output_path = os.path.join(
#                     output_dir, os.path.splitext(basename)[0] + ".json"
#                 )
#                 # Remove the path column as it's no longer needed
#                 # group = group.drop(columns=["path"])
                
#                 # Check if JSON file exists and append or write accordingly
#                 if os.path.exists(output_path):
#                     # Read existing JSON file into a dataframe to append new data
#                     existing_df = pd.read_json(output_path, orient='records', lines=True)
#                     updated_df = pd.concat([existing_df, group])
#                     # Write/overwrite the JSON file with the updated dataframe
#                     updated_df.to_json(output_path, orient='records', lines=True, index=False)
#                 else:
#                     # Write the new dataframe to a JSON file directly
#                     group.to_json(output_path, orient='records', lines=True, index=False)

#         except Empty:
#             continue


# def process_chunk(
#     rank,
#     world_size,
#     paths,
#     ckpt_path,
#     output_dir,
#     num_workers,
#     batch_size,
#     s3,
#     dataset_type,
#     q,
# ):

#     # Load the pre-trained vq model from the hub
    
#     net = VQGANModel.from_pretrained(ckpt_path).cuda()
#     net.to(rank)  # Move model to the correct device
#     print('Model loaded..')

#     worker_paths = paths[rank::world_size]
#     print (f"Rank: {rank} processing {len(worker_paths)} shards : {worker_paths} ")
    

#     dataset = get_dataset(dataset_type, worker_paths, s3)

#     dataloader = torch.utils.data.DataLoader(
#         dataset, #.batched(batch_size),
#         batch_size=batch_size,
#         pin_memory=True,
#         num_workers=num_workers,
#     )

#     writer_p = Process(target=writer_worker, args=(q, output_dir))
#     writer_p.start()

#     num_chunks = len(worker_paths)

#     batch_idx = 0
#     seen_keys = set()
#     print("Start tokenizing......")
    
#     for batch_data in tqdm(
#                 dataloader,
#                 total=int(np.ceil(EXPECTED_CHUNK_SIZE * num_chunks / batch_size)),
#                 desc=f"Rank : {rank}",
#                 position=rank,
#                 leave=False,
#             ):

#         data, metas, key_ = batch_data
        
#         if any(k in seen_keys for k in key_):
#             print(f"Rank {rank}: Skipping batch due to duplicate keys: {key_}")
#             continue
#         seen_keys.update(key_)
   

#         try:
#             image_tensor = data.to(rank)
#         except OSError as e:
#             print(f"Skipping batch due to error: {e}")
#             continue
        
#         data_list = []
#         rows = {}

#         with torch.no_grad():
#             _, tokens = net.encode(image_tensor)
#             tokens_save = einops.rearrange(
#                 tokens.cpu().numpy().astype(np.int32), '(b t) d -> b (t d)', b=tokens.size(0)
#             )
            
#             for i in range(image_tensor.size(0)):

#                 data = b64encode(tokens_save[i].tobytes()).decode('utf-8')
#                 data_list.append(data)

#         for k, v in metas.items():
#             if isinstance(v, torch.Tensor):
#                 v = v.cpu().numpy().tolist()
#             rows[k] = v
        
#         rows["__key__"] = key_
#         rows["tokens"] = data_list

#         q.put((rows, data_list))
#         batch_idx += 1
    
#     q.put(None)
#     writer_p.join()