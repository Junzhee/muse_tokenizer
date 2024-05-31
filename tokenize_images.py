import os
import numpy as np
import einops
import argparse
import shutil
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision.utils import save_image
import json
import wandb
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import Image
from muse import VQGANModel
from base64 import b64encode

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pickle
import json
import base64
import os
# from tqdm.notebook import tqdm, trange
from torch.utils.data.distributed import DistributedSampler
import time

import random


os.environ['NCCL_DEBUG'] = 'INFO'


class CustomImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

# def get_dataset(dataset_type, path, s3):
#     if s3:
#         path = f"pipe:aws s3 cp {path} -"

#     if dataset_type == "laion":
#         dataset = (
#             wds.WebDataset(path)
#             .decode(wds.imagehandler("torchrgb"))
#             .to_tuple("jpg", "json")
#         )
#         dataset = dataset.map(remove_keys)

#         return dataset
#     elif dataset_type == "datacomp":
#         dataset = (
#             wds.WebDataset(path)
#             .decode(wds.imagehandler("torchrgb"))
#             .to_tuple("jpg;png;webp", "json", "__key__", "__url__")
#         )
#         dataset = dataset.map(transform_and_remove_keys)

#         return dataset
#     elif dataset_type == "mmc4":

#         def resize_image(sample):
#             keys = ["png", "jpg", "jpeg"]
#             for key in keys:
#                 if key in sample:
#                     image = np.array(sample[key].resize((256, 256))).astype(np.float32)
#                     image = image.transpose(2, 0, 1) / 255.0
#                     sample["image"] = torch.from_numpy(image)
#             return sample

#         dataset = (
#             wds.WebDataset(path)
#             .decode("pil")
#             .map(resize_image)
#             .to_tuple("image", "__key__")
#         )
#         return dataset

def setup(rank, world_size):
    """
    Setup for distributed computing.
    """
    # Print the current rank:
    print(f"Setup on rank {rank}.")

    random_number = random.randint(0, 66666)

    # Converting the number to a string
    number_as_string = str(random_number)
    os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = number_as_string
    os.environ['MASTER_PORT'] = '12345'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    print(f"Setup finish on rank {rank}.")

def cleanup():
    """
    Cleanup the distributed setup.
    """
    dist.destroy_process_group()



def main(rank, world_size):
    print(f"Running on rank {rank}.")
    setup(rank, world_size)
    print(f"Setup done on rank {rank}.")

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Tokenize the Dataset')
    parser.add_argument('--dataset_path', type=str, default='/home/niudt/dataset/laion4m', help='path of the laion 2b', required=False)
    parser.add_argument('--save_file', type=str, default='./output_dir', help='Name of the save folder', required=False)
    parser.add_argument('--batch_size', type=int, default=128, help='support 256 for 80G A100', required=False)
    parser.add_argument('--ckpt_path', type=str, default="./ckpt/laion", help='path of the vqgan ckpt', required=False)
    args = parser.parse_args()

    # Build folder paths using the provided epic argument
    dataset_path = args.dataset_path
    batch_size = args.batch_size
    work_root = args.save_file


    # Load the pre-trained vq model from the hub
    net = VQGANModel.from_pretrained(args.ckpt_path).cuda()
    net.to(rank)  # Move model to the correct device
    print('Model loaded..')
    net = DDP(net, device_ids=[rank])  # Wrap the model in DDP
    net.eval()
    print('DDP Model loaded...')

    # create image list: this maybe very take time, becuase it create the list of laion2b
    image_list = []
    subset = os.listdir(dataset_path)
    for _subset in subset:
        path = os.path.join(dataset_path, _subset)
        for item in tqdm(os.listdir(path)):
            if not item[-3:] == 'jpg':
                continue
            image_path = os.path.join(path, item)
            image_list.append(image_path)
   
    # image_list = image_list[:10000]
    print(f"Total {len(image_list)} images...")

    transform = transforms.Compose(
        [transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(256),
            transforms.ToTensor(),]
    )

    dataset = CustomImageDataset(image_list, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            sampler=sampler, 
                            num_workers=1,
                            shuffle=False,
                            pin_memory=True,
                            prefetch_factor=2)

    start_time = time.time()
 
    for batch_index, batch in enumerate(tqdm(dataloader)):
        data_list = []  # Initialize a list for the current batch

        x = batch
        x = x.to(net.device)
        with torch.no_grad():
            _, tokens = net.module.encode(x)


            tokens_save = einops.rearrange(
                tokens.cpu().numpy().astype(np.int32), '(b t) d -> b (t d)', b=tokens.size(0)
            )
            for i in range(x.size(0)):
                data = {'tokens': b64encode(tokens_save[i].tobytes()).decode('utf-8')}
                data_list.append(json.dumps(data))

        json_file_path = f"{work_root}/output_{rank}.json"

        # Determine the mode in which to open the file ('w' for the first batch, 'a' for subsequent batches)
        mode = 'w' if batch_index == 0 else 'a'
        with open(json_file_path, mode) as fout:
            for data in data_list:
                fout.write(data + '\n')
    
    print(f"Time taken: {time.time() - start_time} seconds.")

if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # Number of available GPUs
    print(f"Running on {world_size} GPUs.")
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)