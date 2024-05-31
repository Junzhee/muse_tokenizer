import pandas as pd
import os



def check_json_file(output_path):
    if not os.path.exists(output_path):
        print(f"The file {output_path} does not exist.")
        return
    
    # Read the JSON file into a DataFrame
    df = pd.read_json(output_path, orient='records', lines=True)
    
    # Check the total length of the DataFrame
    total_length = len(df)
    
    # Check the number of unique items in the DataFrame
    # Assuming 'item_column' is the name of the column with items to check uniqueness
    unique_items = df['key'].nunique()
    print(output_path)
    print(f"Total length: {total_length}")
    print(f"Number of unique items: {unique_items}")


def check_directory(dir_path):
    pass
    


output_path = "/p/scratch/ccstdl/xu17/jz/muse_tokenizer/output_dir/tmp/0040004_unique.json"
check_json_file(output_path)
