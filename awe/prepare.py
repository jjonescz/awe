"""
Downloads GloVe embeddings, currently used when building demo inference image.

Run as `python -m awe.prepare`.
"""

import awe.data.glove


def main():
    # HACK: Avoid progress bar (it clutters Docker logs).
    awe.data.glove.disable_progress()

    awe.data.glove.download_embeddings()

if __name__ == '__main__':
    main()
