from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFKC
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


def Trainer(batch_iterator, vocab_size=64_000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"], save_to = "tokenizer"):
    print(f"""
--- Training Configuration ---
Vocab Size: {vocab_size}
Special Tokens: {special_tokens}
Save Path: {save_to}
------------------------------
""")

    # special tokens map creation
    special_tokens_map = {}
    for t in special_tokens:
        t_lower = t.lower()
        if "pad" in t_lower: special_tokens_map["pad_token"] = t
        elif "unk" in t_lower: special_tokens_map["unk_token"] = t
        elif "cls" in t_lower: special_tokens_map["cls_token"] = t
        elif "sep" in t_lower: special_tokens_map["sep_token"] = t
        elif "mask" in t_lower: special_tokens_map["mask_token"] = t

    unk_token = special_tokens_map.get("unk_token", "[UNK]")
    tokenizer = Tokenizer(BPE(unk_token=unk_token))
    tokenizer.normalizer = NFKC() # Recommended: Normalizes unicode characters
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    trainer = BpeTrainer(
            vocab_size=vocab_size, # 64k
            special_tokens=special_tokens,
            show_progress=True 
        )

    print(f"Starting training on WHOLE dataset (Vocab: {vocab_size})...")
    tokenizer.train_from_iterator(batch_iterator, trainer=trainer)
    print("\nTraining Complete!")

    # Enable automatic insertion of [CLS] and [SEP]
    cls_token = special_tokens_map.get("cls_token", "[CLS]")
    sep_token = special_tokens_map.get("sep_token", "[SEP]")
    
    tokenizer.post_processor = TemplateProcessing(
        single=f"{cls_token} $A {sep_token}",
        pair=f"{cls_token} $A {sep_token} $B:1 {sep_token}:1",
        special_tokens=[
            (cls_token, tokenizer.token_to_id(cls_token)),
            (sep_token, tokenizer.token_to_id(sep_token)),
        ],
    )

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