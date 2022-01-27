from tqdm.auto import tqdm

import awe.qa.collater
import awe.qa.parser
import awe.qa.sampler
from awe import awe_graph


def validate(
    pages: list[awe_graph.HtmlPage],
    collater: awe.qa.collater.Collater
):
    truncated = 0
    for page in (progress := tqdm(pages, desc='pages')):
        page: awe_graph.HtmlPage
        samples = awe.qa.sampler.get_samples([page])
        encodings = collater.get_encodings(samples)

        # Decode.
        for batch_idx, sample in enumerate(samples):
            start = encodings['start_positions'][batch_idx]
            end = encodings['end_positions'][batch_idx]
            tokens = encodings['input_ids'][batch_idx][start:end + 1]
            decoded = collater.tokenizer.decode(tokens)
            expected = sample.values[0]
            if not awe.qa.parser.coarse_words_equal(decoded, expected):
                if (
                    start == collater.tokenizer.model_max_length and
                    end == collater.tokenizer.model_max_length
                ):
                    truncated += 1
                    progress.set_postfix({ 'truncated': truncated })
                else:
                    raise RuntimeError(
                        f'Inconsistent words ({decoded=}, {expected=}, ' +
                        f'{page.file_path=}).'
                    )
