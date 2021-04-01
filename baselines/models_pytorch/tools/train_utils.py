import shutil
import re
from pathlib import Path
import json
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def load_labels_from_json(path):
    null_label = "_<PAD>"
    word_label = "_<WORD>"
    with open(path, 'r') as f:
        data = json.load(f)
        parser_label2id = data["instance2index"]
    if null_label not in parser_label2id:
        parser_label2id[null_label] = len(parser_label2id)
    if word_label not in parser_label2id:
        parser_label2id[word_label] = len(parser_label2id)
    return parser_label2id


def delete_old_checkpoints(output_dir, best_checkpoint, save_limit=1):
    # delete old checkpoints other than the best
    ordering_and_checkpoint_path = []

    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"checkpoint-*")]

    for path in glob_checkpoints:
        regex_match = re.match(f".*checkpoint-([0-9]+)", path)
        if regex_match and regex_match.groups():
            ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    
    #print ("checkpoints:\n", checkpoints_sorted)
    if best_checkpoint:
        checkpoints_sorted.remove(best_checkpoint)
        if save_limit == 1:
            checkpoints_to_delete = checkpoints_sorted
        else:
            checkpoints_to_delete = checkpoints_sorted[:-save_limit+1]
    else:
        checkpoints_to_delete = checkpoints_sorted[:-save_limit]
    for checkpoint in checkpoints_to_delete:
        logger.info("Deleting older checkpoint [{}] due to save_limit".format(checkpoint))
        shutil.rmtree(checkpoint)