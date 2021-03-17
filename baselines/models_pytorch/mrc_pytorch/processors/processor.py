import csv
import torch
import logging
import sys
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


def floyd(heads, max_len):
    INF = 1e8
    inf = torch.ones_like(heads, device=heads.device, dtype=heads.dtype) * INF
    # replace 0 with infinite
    dist = torch.where(heads==0, inf.long(), heads.long())
    for k in range(max_len):
        for i in range(max_len):
            for j in range(max_len):
                if dist[i][k] != INF and dist[k][j] != INF and dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    zero = torch.zeros_like(heads, device=heads.device).long()
    dist = torch.where(dist==INF, zero, dist)
    return dist

def compute_distance(heads, mask, debug=False):
    if debug:
        torch.set_printoptions(profile="full")

    lengths = [sum(m) for m in mask]
    dists = []
    logger.info("Start computing distance ...")
    # for each sentence
    for i in range(len(heads)):
        if i % 100 == 0:
            print ("%d..."%i, end="")
            sys.stdout.flush()
        if debug:
            print ("heads:\n", heads[i])
            print ("mask:\n", mask[i])
            print ("lengths:\n", lengths[i])
        dist = floyd(heads[i], lengths[i])
        dists.append(dist)
        if debug:
            print ("dist:\n", dist)
    return dists