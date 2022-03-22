# Run `python -m awe.prepare`

import awe.data.glove

def main():
    awe.data.glove.download_embeddings()

if __name__ == '__main__':
    main()
