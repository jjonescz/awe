import transformers

def clamp_span(
    start: int,
    end: int,
    encodings: transformers.BatchEncoding,
    batch_idx: int
):
    """
    Clamps `span` so that it points to one word (i.e., one text fragment) only.
    """

    word_idx = encodings.token_to_word(batch_idx, start)
    span = encodings.word_to_tokens(batch_idx, word_idx, sequence_index=1)
    return span.start, span.end - 1
