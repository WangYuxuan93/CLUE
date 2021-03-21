from .c3_processor import (c3Processor, c3_collate_fn, load_and_cache_c3_examples)
from .chid_processor import (chid_collate_fn, load_and_cache_chid_examples)
from .cmrc2018_processor import (cmrc2018_collate_fn, load_and_cache_cmrc2018_examples)

label_lists = {
		'chid': ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    'c3': ["0", "1", "2", "3"],
}

example_loaders = {
		'chid': load_and_cache_chid_examples,
    'c3': load_and_cache_c3_examples,
    'cmrc2018': load_and_cache_cmrc2018_examples,
}

mrc_processors = {
    'c3': c3Processor,
}

mrc_output_modes = {
		'chid': "classification",
    'c3': "classification",
    'cmrc2018': "qa",
}

collate_fns = {
		'chid': chid_collate_fn,
    'c3': c3_collate_fn,
    'cmrc2018': cmrc2018_collate_fn
}