from .c3_processor import (c3Processor, c3_collate_fn, load_and_cache_c3_examples)

mrc_processors = {
    'c3': c3Processor,
}

mrc_output_modes = {
    'c3': "classification",
}
