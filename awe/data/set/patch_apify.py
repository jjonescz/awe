# Run: `python -m awe.data.set.patch_apify`

import json

import ijson
from tqdm.auto import tqdm


def main():
    input_path = 'data/apify/notinoEn/augmented_dataset.json'
    output_path = 'data/apify/notinoEn/slim_dataset.json'
    with open(input_path, mode='rb') as input_file:
        with open(output_path, mode='w', encoding='utf-8') as output_file:
            output_file.write('[\n')
            rows = ijson.items(input_file, 'item')
            after_first = False
            for input_row in tqdm(rows, desc=input_path, total=2000):
                output_row = {
                    k: v
                    for k, v in input_row.items()
                    if k == 'url' or k.startswith('selector_')
                }
                if after_first:
                    output_file.write(',\n')
                else:
                    after_first = True
                json.dump(output_row, output_file)
            output_file.write(']\n')

if __name__ == '__main__':
    main()
