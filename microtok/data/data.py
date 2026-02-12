from datasets import load_dataset
from tqdm import tqdm

print("Things loaded.. ")

def batch_iterator(BATCH_SIZE = 10_000, Dataset="HuggingFaceFW/fineweb-edu", split="train", name="sample-10BT", streaming=True, trust_remote_code=True):
    dataset = load_dataset(
        Dataset, 
        name=name, 
        split=split, 
        streaming=streaming, 
        trust_remote_code=trust_remote_code
    )
    batch = []

    print("Starting stream... (This will take hours)")
    for example in tqdm(dataset, desc="Processing Rows", unit=" docs"):
        batch.append(example["text"])
        
        if len(batch) == BATCH_SIZE:
            yield batch
            batch = []
            
    # Yield the last chunk
    if batch:
        yield batch
