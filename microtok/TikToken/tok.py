from tokenizers import Tokenizer, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
from tokenizers.normalizers import NFKC
from transformers import PreTrainedTokenizerFast

# GPT-4 (cl100k_base) regex pattern
# Note: we use the 'tokenizers' library implementation of the regex logic
GPT4_SPLIT_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"

def Trainer(batch_iterator, vocab_size=64_000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"], save_to="tokenizer_tiktoken"):
    print(f"""
--- TikToken Training Configuration ---
Vocab Size: {vocab_size}
Special Tokens: {special_tokens}
Save Path: {save_to}
---------------------------------------
""")
    # Initialize BPE (TikToken uses BPE)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # Normalization
    tokenizer.normalizer = NFKC()
    
    # Pre-tokenization: 
    # 1. Split using GPT-4 Regex
    # 2. ByteLevel encoding (crucial for TikToken/GPT-4 style)
    tokenizer.pre_tokenizer = Sequence([
        Split(pattern=Regex(GPT4_SPLIT_PATTERN), behavior="isolated"),
        ByteLevel(add_prefix_space=False, use_regex=False) 
    ])
    
    tokenizer.decoder = decoders.ByteLevel()

    # Automate special tokens map
    special_tokens_map = {}
    for t in special_tokens:
        t_lower = t.lower()
        if "pad" in t_lower: special_tokens_map["pad_token"] = t
        elif "unk" in t_lower: special_tokens_map["unk_token"] = t
        elif "cls" in t_lower: special_tokens_map["cls_token"] = t
        elif "sep" in t_lower: special_tokens_map["sep_token"] = t
        elif "mask" in t_lower: special_tokens_map["mask_token"] = t
    
    trainer = BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=special_tokens,
        show_progress=True
    )

    print(f"Starting TikToken-style training (Vocab: {vocab_size})...")
    tokenizer.train_from_iterator(batch_iterator, trainer=trainer)
    print("\nTraining Complete!")
    
    folder_name = save_to
    tokenizer.save(f"{folder_name}.json")

    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{folder_name}.json")
    hf_tokenizer.add_special_tokens(special_tokens_map)
    hf_tokenizer.save_pretrained(folder_name)
    print(f"Saved to '{folder_name}'")

    # --- TEST ---
    print("\n--- TEST RESULTS ---")
    text = "Hello world! I am writing code today."
    print(f"Input:  '{text}'")
    print(f"Tokens: {hf_tokenizer.tokenize(text)}")
