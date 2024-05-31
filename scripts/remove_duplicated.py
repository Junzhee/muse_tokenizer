import os
import braceexpand
import multiprocessing
import torch.multiprocessing as mp
import argparse

os.system("taskset -p 0xffffffffffffffffffffffff %d" % os.getpid())


def remove_duplicates_from_tar(tar_path, output_path):
    import tarfile

    # Open the original tar file
    with tarfile.open(tar_path, 'r') as tar:
        members = tar.getmembers()
        
        # Create a set to track unique files
        seen_files = set()
        unique_members = []
        
        for member in members:
            if member.name not in seen_files:
                seen_files.add(member.name)
                unique_members.append(member)
            else:
                print(f"Duplicate file found and skipped: {member.name}")
                
        # Write unique members to the new tar file
        with tarfile.open(output_path, 'w') as out_tar:
            for member in unique_members:
                out_tar.addfile(member, tar.extractfile(member))
            


def process_tar(paths, output_base_dir):
    processed_paths = []
    os.makedirs(output_base_dir, exist_ok=True)
    
    for path in paths:
        base_name = os.path.basename(path)
        processed_path = os.path.join(output_base_dir, base_name.replace('.tar', '_unique.tar'))
        remove_duplicates_from_tar(path, processed_path)
        processed_paths.append(processed_path)
    
    return processed_paths


def worker(rank, paths, output_base_dir, num_workers):
    chunk_size = len(paths) // num_workers
    start = rank * chunk_size
    end = (rank + 1) * chunk_size if rank != num_workers - 1 else len(paths)
    chunk_paths = paths[start:end]
    process_tar(chunk_paths, output_base_dir)


def main():
    parser = argparse.ArgumentParser(description='Tokenize the Dataset')
    parser.add_argument('--start_shard', type=str, default="0010000", help='Starting shard number', required=True)
    parser.add_argument('--end_shard', type=str, default="0010007", help='Ending shard number', required=True)
    
    args = parser.parse_args()


    start_shard = args.start_shard
    end_shard = args.end_shard

    num_workers = 96
    output_base_dir = "/p/scratch/laionize/m3_data/datacomp_deduplicated/"

    path_template = f"/p/fastdata/mmlaion/datacomp/datacomp_1B/flat/{{{start_shard}..{end_shard}}}.tar"
    paths = list(braceexpand.braceexpand(path_template))

    mp.spawn(worker, args=(paths, output_base_dir, num_workers), nprocs=num_workers, join=True)

    print("Completed.....")

if __name__ == "__main__":
    main()
  