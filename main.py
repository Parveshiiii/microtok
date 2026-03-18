import logging

# Configure logging ONCE before importing any modules
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

from microtok import BPETrainer, batch_iterator


def main():
    # Uses default BATCH_SIZE=10,000 and VOCAB_SIZE=64,000
    batch_iter = batch_iterator()
    BPETrainer(batch_iter)

if __name__ == "__main__":
    main()