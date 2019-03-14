import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_CONFIG = {
    'vocab_file': os.path.join(BASE_DIR, 'ckpt/vocab.txt')
}

if __name__ == '__main__':
    print(BASE_DIR)
