from .c3_processor import (c3Processor, c3_collate_fn, load_and_cache_c3_examples)
from .chid_processor import (chid_collate_fn, load_and_cache_chid_examples)

example_loaders = {
		'chid': load_and_cache_chid_examples,
    'c3': load_and_cache_c3_examples,
}

mrc_processors = {
    'c3': c3Processor,
}

mrc_output_modes = {
		'chid': "classification",
    'c3': "classification",
}

collate_fns = {
		'chid': chid_collate_fn,
    'c3': c3_collate_fn,
}