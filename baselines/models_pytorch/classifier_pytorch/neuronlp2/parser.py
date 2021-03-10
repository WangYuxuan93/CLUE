import os
import sys
import gc
import json
import nltk

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

import time
import argparse
import math
import numpy as np
import torch
import random
import string

from neuronlp2.io import conllx_data
#from neuronlp2.models.biaffine_parser import BiaffineParser
from transformers import AutoTokenizer

from transformers.utils import logging
logger = logging.get_logger(__name__)

PAD = "_PAD"
ROOT = "_ROOT"
END = "_END"
NUM_SYMBOLIC_TAGS = 3

def convert_tokens_to_ids(tokenizer, tokens):

    all_wordpiece_list = []
    all_first_index_list = []

    for toks in tokens:
        wordpiece_list = []
        first_index_list = []
        for token in toks:
            if token == PAD:
                token = tokenizer.pad_token
            elif token == ROOT:
                token = tokenizer.cls_token
            elif token == END:
                token = tokenizer.sep_token
            wordpiece = tokenizer.tokenize(token)
            # add 1 for cls_token <s>
            first_index_list.append(len(wordpiece_list)+1)
            wordpiece_list += wordpiece
            #print (wordpiece)
        #print (wordpiece_list)
        #print (first_index_list)
        bpe_ids = tokenizer.convert_tokens_to_ids(wordpiece_list)
        #print (bpe_ids)
        bpe_ids = tokenizer.build_inputs_with_special_tokens(bpe_ids)
        #print (bpe_ids)
        all_wordpiece_list.append(bpe_ids)
        all_first_index_list.append(first_index_list)

    all_wordpiece_max_len = max([len(w) for w in all_wordpiece_list])
    all_wordpiece = np.stack(
          [np.pad(a, (0, all_wordpiece_max_len - len(a)), 'constant', constant_values=tokenizer.pad_token_id) for a in all_wordpiece_list])
    all_first_index_max_len = max([len(i) for i in all_first_index_list])
    all_first_index = np.stack(
          [np.pad(a, (0, all_first_index_max_len - len(a)), 'constant', constant_values=0) for a in all_first_index_list])

    # (batch, max_bpe_len)
    input_ids = torch.from_numpy(all_wordpiece)
    # (batch, seq_len)
    first_indices = torch.from_numpy(all_first_index)

    return input_ids, first_indices


def convert_texts_to_ids(tokenizer, texts):

    all_wordpiece_list = []
    all_first_index_list = []

    for text in texts:
        wordpiece_list = []
        # for root token and the first token
        first_index_list = [1,2]
        wordpiece_list = [tokenizer.cls_token] + tokenizer.tokenize(text)
        #print (wordpiece_list)
        for i in range(2, len(wordpiece_list)):
            if wordpiece_list[i][0] == "\u0120" or wordpiece_list[i][0] in string.punctuation or wordpiece_list[i-1][-1] in string.punctuation:
                # add 1 for <s> token
                first_index_list.append(i+1)

        #print (wordpiece_list)
        #print (first_index_list)
        bpe_ids = tokenizer.convert_tokens_to_ids(wordpiece_list)
        #print (bpe_ids)
        bpe_ids = tokenizer.build_inputs_with_special_tokens(bpe_ids)
        #print (bpe_ids)
        all_wordpiece_list.append(bpe_ids)
        all_first_index_list.append(first_index_list)

    all_wordpiece_max_len = max([len(w) for w in all_wordpiece_list])
    all_wordpiece = np.stack(
          [np.pad(a, (0, all_wordpiece_max_len - len(a)), 'constant', constant_values=tokenizer.pad_token_id) for a in all_wordpiece_list])
    all_first_index_max_len = max([len(i) for i in all_first_index_list])
    all_first_index = np.stack(
          [np.pad(a, (0, all_first_index_max_len - len(a)), 'constant', constant_values=0) for a in all_first_index_list])

    # (batch, max_bpe_len)
    input_ids = torch.from_numpy(all_wordpiece)
    # (batch, seq_len)
    first_indices = torch.from_numpy(all_first_index)

    return input_ids, first_indices, all_wordpiece_list, all_first_index_list


def reform(tokens, text):
    toks = []
    offset = 0
    text = text.strip(" ")
    for token in tokens:
        while text[offset:][0] == " ":
            offset += 1
        if text[offset:].startswith(token):
            offset += len(token)
            toks.append(token)
        elif token == "``" and text[offset:].startswith("''"):
            toks.append("''")
            offset += 2
        elif token == "``" and text[offset] == '"':
            toks.append('"')
            offset += 1
        elif token == "''" and text[offset] == '"':
            toks.append('"')
            offset += 1
    assert offset == len(text)
    return toks


def get_valid_wp(wps, idx):
    wp = wps[idx]
    while wp == "\u0120":
        idx += 1
        wp = wps[idx]
    return wp, idx


def get_first_ids(tokenizer, input_ids, type="nltk", debug=False):
    # type (nltk: use nltk word_tokenize to align|rule: use rule to align)

    first_ids_list = []

    for ids in input_ids:
        wps = tokenizer.convert_ids_to_tokens(ids)
        if type == "rule":
            # for root token and the first token
            first_ids = [1,2]
            #print (wps)
            for i in range(3, len(wps)-1):
                if wps[i][0] == "\u0120" or wps[i][0] in string.punctuation or wps[i-1][-1] in string.punctuation:
                    first_ids.append(i)

            first_ids_list.append(first_ids)
            #print ("wps:\n", wps)
            #print ("first_ids:\n", first_ids)
        elif type == "nltk":
            # ignore first two cls token and the last sep token
            wps_valid = []
            for i in range(len(wps)):
                # rm pad & sep tokens
                if wps[i] in [tokenizer.pad_token, tokenizer.sep_token]:
                    break
                wps_valid.append(wps[i])
            wps_valid.append(tokenizer.sep_token)
            wps = wps_valid
            text = "".join(wps[2:-1]).replace("\u0120", " ")
            tokens = nltk.word_tokenize(text)
            if debug:
                print ("text:\n", text)
                print ("tokens:\n", tokens)
                print ("wps:\n", wps)
            tokens = reform(tokens, text)
            # for root token
            first_ids = [1]
            wp_idx = 2
            #wp = wps[wp_idx]
            wp, wp_idx = get_valid_wp(wps, wp_idx)
            i = 0
            token = tokens[i]
            #for token in tokens:
            while i < len(tokens):
                token = tokens[i]
                offset = 0
                if debug:
                    print ("token:{} | wp:{}".format(token[offset:], wp))
                """
                if token.startswith("``") and (wp[0]=='"' or (len(wp)>1 and wp[1]=='"')):
                    token = token.replace("``", '"')
                elif token.startswith("''") and (wp[0]=='"' or (len(wp)>1 and wp[1]=='"')):
                    token = token.replace("\'\'", '"')
                elif token.startswith("`") and (wp[0] == "'" or (len(wp)>1 and wp[1]=='"')):
                    token = token.replace("`", "'")
                """
                if (wp.startswith("\u0120") and token.startswith(wp[1:])) or token.startswith(wp):
                    first_ids.append(wp_idx)
                    offset += len(wp) - 1 if wp.startswith("\u0120") else len(wp)
                    wp_idx += 1
                    #wp = wps[wp_idx]
                    wp, wp_idx = get_valid_wp(wps, wp_idx)
                else:
                    token = token+tokens[i+1]
                    i += 1
                    while (wp.startswith("\u0120") and len(token)<len(wp)-1) or (not wp.startswith("\u0120") and len(token)<len(wp)):
                        token = token+tokens[i+1]
                        i += 1
                    if (wp.startswith("\u0120") and token.startswith(wp[1:])) or token.startswith(wp):
                        first_ids.append(wp_idx)
                        offset += len(wp) - 1 if wp.startswith("\u0120") else len(wp)
                        wp_idx += 1
                        #wp = wps[wp_idx]
                        wp, wp_idx = get_valid_wp(wps, wp_idx)
                    else:
                        print ("Mismatch (start) in {} token:{} | wp:{}:\ntoks:{}\nwp:{}".format(wp_idx, 
                            token[offset:], wp, tokens, wps))
                        exit()
                while offset < len(token):
                    if debug:
                        print ("token:{} | wp:{}".format(token[offset:], wp))
                    if token[offset:].startswith(wp):
                        offset += len(wp)
                        wp_idx += 1
                        #wp = wps[wp_idx]
                        wp, wp_idx = get_valid_wp(wps, wp_idx)
                    else:
                        token = token + tokens[i+1]
                        i += 1
                        while len(token[offset:]) < len(wp):
                            token = token + tokens[i+1]
                            i += 1
                        if token[offset:].startswith(wp):
                            offset += len(wp)
                            wp_idx += 1
                            #wp = wps[wp_idx]
                            wp, wp_idx = get_valid_wp(wps, wp_idx)
                        else:
                            print ("Mismatch (mid) in {} token:{} | wp:{}\ntoks:{}\nwp:{}".format(wp_idx, 
                                token[offset:], wp, tokens, wps))
                            exit()
                i += 1
            if debug:
                print ("tokens:\n", tokens)
                print ("wps:\n", wps)
                print ("first_ids:\n", first_ids)
            first_ids_list.append(first_ids)

    max_len = max([len(i) for i in first_ids_list])
    first_ids = np.stack(
          [np.pad(a, (0, max_len - len(a)), 'constant', constant_values=0) for a in first_ids_list])
    first_ids = torch.from_numpy(first_ids)
    return first_ids, first_ids_list


def split_ids(tokenizer, input_ids):
    sep_id = tokenizer.sep_token_id
    cls_id = tokenizer.cls_token_id
    input_ids_a, input_ids_b = [], []
    for ids in input_ids:
        #print ("ids:\n", ids)
        sep_index = ids.index(sep_id)
        # [CLS] [CLS] ... [SEP] , the second cls is for root
        input_ids_a.append([cls_id]+ids[:sep_index]+[sep_id])
        input_ids_b.append([cls_id,cls_id]+ids[sep_index+2:])
        #print ("ids_a:\n", input_ids_a[-1])
        #print ("ids_b:\n", input_ids_b[-1])

    max_len_a = max([len(w) for w in input_ids_a])
    inputs_a = np.stack(
          [np.pad(a, (0, max_len_a - len(a)), 'constant', constant_values=tokenizer.pad_token_id) for a in input_ids_a])
    max_len_b = max([len(w) for w in input_ids_b])
    inputs_b = np.stack(
          [np.pad(b, (0, max_len_b - len(b)), 'constant', constant_values=tokenizer.pad_token_id) for b in input_ids_b])
    return inputs_a, inputs_b, input_ids_a, input_ids_b


def merge_first_ids(ids, ids_a, first_ids_a, ids_b=None, first_ids_b=None, debug=False):
    pad_id = 0
    sep_id = 2
    first_ids_list = []
    for i in range(len(ids_a)):
        #print (ids_a)
        merge_ids = ids_a[i][1:]
        # minus 1 for removing the root token
        first_ids = [x-1 for x in first_ids_a[i]] + [len(merge_ids)-1]
        if ids_b is not None:
            first_ids += [x-1+len(merge_ids) for x in first_ids_b[i]]
            merge_ids += [sep_id] + ids_b[i][2:]
            first_ids += [len(merge_ids)-1]
        if debug:
            print ("ids_a:\n{}".format(ids_a[i]))
            if ids_b is not None:
                print ("ids_b:\n{}".format(ids_b[i]))
            print ("first_ids:\n",first_ids)
            print ("merge_ids:\n", merge_ids)
            print ("ids:\n", ids[i])
        assert merge_ids == ids[i][:len(merge_ids)]
        
        first_ids_list.append(first_ids)
    return first_ids_list


class Parser(object):
    def __init__(self, model_path, pretrained_lm="roberta", lm_path=None,
                use_pretrained_static=False, batch_size=16, parser_type="ptb"):
        cuda = torch.cuda.is_available()
        device = torch.device('cuda', 0) if cuda else torch.device('cpu')
        self.null_label = "_<PAD>"
        self.word_label = "_<WORD>"
        self.device = device
        self.batch_size = batch_size
        self.parser_type = parser_type

        model_name = os.path.join(model_path, 'model.pt')

        #logger = get_logger("Parsing")
        logger.info("Creating Alphabets")
        alphabet_path = os.path.join(model_path, 'alphabets')
        assert os.path.exists(alphabet_path)
        word_alphabet, char_alphabet, pos_alphabet, rel_alphabet = conllx_data.create_alphabets(alphabet_path, None, 
                                    normalize_digits=False, pos_idx=3)
        self.rel_alphabet = rel_alphabet
        self.parser_label_map = rel_alphabet.instance2index
        if self.null_label not in self.parser_label_map:
            self.parser_label_map[self.null_label] = len(self.parser_label_map)
        if self.word_label not in self.parser_label_map:
            self.parser_label_map[self.word_label] = len(self.parser_label_map)

        num_words = word_alphabet.size()
        num_chars = char_alphabet.size()
        num_pos = pos_alphabet.size()
        num_rels = rel_alphabet.size()

        logger.info("Word Alphabet Size: %d" % num_words)
        logger.info("Character Alphabet Size: %d" % num_chars)
        logger.info("POS Alphabet Size: %d" % num_pos)
        logger.info("Rel Alphabet Size: %d" % num_rels)


        hyps = json.load(open(os.path.join(model_path, 'config.json'), 'r'))

        num_lans = 1
        if pretrained_lm in ['none']:
            self.tokenizer = None 
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(lm_path)


        self.network = BiaffineParser(hyps, 0, num_words, num_chars, num_pos, num_rels,
                               device=device, pretrained_lm=pretrained_lm, lm_path=lm_path,
                               use_pretrained_static=use_pretrained_static, 
                               use_random_static=False, use_elmo=False, elmo_path=None, num_lans=1)


        self.network = self.network.to(device)
        self.network.load_state_dict(torch.load(model_name, map_location=device))

    def predict(self, first_ids, ids):
        heads, rels = [], []
        masks = torch.ones_like(first_ids)
        zeros = torch.zeros_like(first_ids)
        masks = torch.where(first_ids==0, zeros, masks)
        lengths = masks.sum(-1).cpu().numpy()
        ids = ids.to(self.device)
        first_ids = first_ids.to(self.device)
        masks = masks.to(self.device)
        with torch.no_grad():
            self.network.eval()
            heads_pred, rels_pred = self.network.decode(first_ids, None, None, None, mask=masks, 
                bpes=ids, first_idx=first_ids, input_elmo=None, lan_id=None, 
                leading_symbolic=NUM_SYMBOLIC_TAGS)
            for j, length in enumerate(lengths):
                heads.append(heads_pred[j][1:length])
                rels.append(rels_pred[j][1:length])
                #rels.append([self.rel_alphabet.get_instance(r) for r in rels_pred[j][1:length]])
        return heads, rels

    def parse_bpes(self, input_ids, masks, batch_size=None,has_b=False,expand_type="copy",
                    max_length=512, align_type="nltk", return_tensor=False, sep_token_id=2, **kwargs):
        batch_size = batch_size if batch_size is not None else self.batch_size
        
        first_ids_list = []
        heads_a_list, rels_a_list = [], []
        heads_b_list, rels_b_list = [], []
        for i in range(0, len(input_ids), batch_size):
            if i % 1024 == 0:
                print ("%d..."%i, end="")
                sys.stdout.flush()
            inputs = input_ids[i:i+batch_size]
            #print ("inputs:\n", inputs)
            mask = np.array(masks[i:i+batch_size]).sum(-1)
            max_len = max(mask)
            inputs_ = [x[:max_len] for x in inputs]
            #print ("inputs_:\n", inputs_)
            if has_b:
                ids_a, ids_b, input_ids_a, input_ids_b = split_ids(self.tokenizer, inputs_)
                ids_b = torch.from_numpy(ids_b)
            else:
                # add the root token, use as parser input
                ids_a = [[self.tokenizer.cls_token_id]+x for x in inputs_]
                # stay list for align 
                input_ids_a = [[self.tokenizer.cls_token_id]+x for x in inputs_]
                ids_b, input_ids_b, first_ids_b = None, None, None
            ids_a = torch.from_numpy(np.array(ids_a))
            fids_a, first_ids_a = get_first_ids(self.tokenizer, ids_a, align_type)
            heads_a, rels_a = self.predict(fids_a, ids_a)
            #print ("heads_a:\n", heads_a)
            #print ("rels_a:\n", rels_a)
            heads_b, rels_b = None, None
            if ids_b is not None:
                fids_b, first_ids_b = get_first_ids(self.tokenizer, ids_b, align_type)
                heads_b, rels_b = self.predict(fids_b, ids_b)
                #print ("heads_b:\n", heads_b)
                #print ("rels_b:\n", rels_b)
                heads_b_list.extend(heads_b)
                rels_b_list.extend(rels_b)

            first_ids = merge_first_ids(inputs_, input_ids_a, first_ids_a, input_ids_b, first_ids_b)
            first_ids_list.extend(first_ids)
            heads_a_list.extend(heads_a)
            rels_a_list.extend(rels_a)
        heads, rels = self.align_heads(self.tokenizer, first_ids_list, heads_a_list, rels_a_list, heads_b_list, rels_b_list, 
                        max_length=max_length, expand_type=expand_type)
        if return_tensor:
            heads = torch.from_numpy(heads)
            rels = torch.from_numpy(rels)
            mask = torch.from_numpy(np.array(masks))

            batch_size, seq_len = heads.size()

            # (batch, seq_len), seq mask, where at position 0 is 0
            root_mask = torch.arange(seq_len, device=heads.device).gt(0).float().unsqueeze(0) * mask
            zeros = torch.zeros_like(root_mask, device=heads.device)
            root_mask = torch.where(torch.from_numpy(np.array(input_ids))==sep_token_id, zeros, root_mask)
            # (batch, seq_len, seq_len)
            mask_3D = (root_mask.unsqueeze(-1) * mask.unsqueeze(1))
            # (batch, seq_len, seq_len)
            heads_3D = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.int32, device=heads.device)
            heads_3D.scatter_(-1, heads.unsqueeze(-1), 1)
            heads_3D = heads_3D * mask_3D
            # (batch, seq_len, seq_len)
            rels_3D = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.long, device=heads.device)
            rels_3D.scatter_(-1, heads.unsqueeze(-1), rels.unsqueeze(-1))

            heads = heads_3D
            rels = rels_3D

        return heads, rels

    def align_heads(self, tokenizer, first_ids_list, heads_a, rels_a, 
                    heads_b=None, rels_b=None,
                    max_length=None, expand_type="copy", debug=False):
        null_label = self.parser_label_map[self.null_label]
        word_label = self.parser_label_map[self.word_label]

        #all_first_index_list = []
        all_heads_list = []
        all_rels_list = []

        for i in range(len(heads_a)):
            # the i-th example
            if debug:
                print ("first_ids_list:\n", first_ids_list[i])
                print ("heads_a:\n", heads_a[i])
                print ("rels_a:\n", rels_a[i])
                if heads_b:
                    print ("heads_b:\n", heads_b[i])
                    print ("rels_b:\n", rels_b[i])
            # for the <s> token
            expand_head = [0]
            expand_rel = [null_label]
            # -1 for removing root token
            # it has the idx for starting cls_token
            first_ids = first_ids_list[i]
            for j in range(len(heads_a[i])):
            # the j-th word
                head_to_add = first_ids[heads_a[i][j]]
                rel_to_add = rels_a[i][j] if head_to_add < max_length else null_label
                head_to_add = head_to_add if head_to_add < max_length else 0
                expand_head.append(head_to_add)
                expand_rel.append(rel_to_add)
                begin = first_ids[j+1]+1
                end = first_ids[j+2]
                for k in range(begin, end):
                    if expand_type == "word":
                        head_to_add = first_ids[j+1]
                        rel_to_add = word_label if head_to_add < max_length else null_label
                        head_to_add = head_to_add if head_to_add < max_length else 0
                        expand_head.append(first_ids[j+1])
                        expand_rel.append(word_label)
                    elif expand_type == "copy":
                        head_to_add = first_ids[heads_a[i][j]]
                        rel_to_add = rels_a[i][j] if head_to_add < max_length else null_label
                        head_to_add = head_to_add if head_to_add < max_length else 0
                        expand_head.append(head_to_add)
                        expand_rel.append(rel_to_add)
            expand_head.append(0)
            expand_rel.append(null_label)
            
            offset = len(heads_a[i])+2
            # the index of the 2nd sep_token
            #offset = first_ids[first_offset]
            if heads_b:
                # the 2nd sep token
                expand_head.append(0)
                expand_rel.append(null_label)
                for j in range(len(heads_b[i])):
                # the j-th word
                    head_to_add = first_ids[offset+heads_b[i][j]]
                    rel_to_add = rels_b[i][j] if head_to_add < max_length else null_label
                    head_to_add = head_to_add if head_to_add < max_length else 0
                    expand_head.append(head_to_add)
                    expand_rel.append(rel_to_add)
                    begin = first_ids[offset+j+1]+1
                    end = first_ids[offset+j+2]
                    for k in range(begin, end):
                        if expand_type == "word":
                            head_to_add = first_ids[offset+j+1]
                            rel_to_add = word_label if head_to_add < max_length else null_label
                            head_to_add = head_to_add if head_to_add < max_length else 0
                            expand_head.append(head_to_add)
                            expand_rel.append(rel_to_add)
                        elif expand_type == "copy":
                            head_to_add = first_ids[offset+heads_b[i][j]]
                            rel_to_add = rels_b[i][j] if head_to_add < max_length else null_label
                            head_to_add = head_to_add if head_to_add < max_length else 0
                            expand_head.append(head_to_add)
                            expand_rel.append(rel_to_add)
                # the ending sep token
                expand_head.append(0)
                expand_rel.append(null_label)

            if len(expand_head) > max_length:
                expand_head = expand_head[:max_length-1]
                expand_rel = expand_rel[:max_length-1]
                expand_head.append(0)
                expand_rel.append(null_label)
                #print ("#### Warning: expanded head ({}) exceeds max_length ({}) ####".format(len(expand_head), max_length))

            while len(expand_head) < max_length:
                expand_head.append(0)
                expand_rel.append(0)
            if debug:
                print ("expand_head:\n", expand_head)
                print ("expand_rel:\n", expand_rel)
            assert max(expand_head) < max_length
            all_heads_list.append(expand_head)
            all_rels_list.append(expand_rel)
        heads = np.array(all_heads_list)
        rels = np.array(all_rels_list)
        return heads, rels

    def parse(self, sentences, batch_size=None,type="wp"):
        batch_size = batch_size if batch_size is not None else self.batch_size
        heads, rels, sents = [], [], []
        inputs, firsts = [], []
        for i in range(0, len(sentences), batch_size):
            if type == "wp":
                input_ids, first_indices, wp_ids, first_ids = convert_texts_to_ids(self.tokenizer, sentences[i:i+batch_size])
                inputs.extend(wp_ids)
                firsts.extend(first_ids)
            else:
                sents_ = [[ROOT]+nltk.word_tokenize(s) for s in sentences[i:i+batch_size]]
                sents.extend(sents_)
                input_ids, first_indices = convert_tokens_to_ids(self.tokenizer, sents_)
                inputs.extend([input_ids[j] for j in range(len(sentences[i:i+batch_size]))])
                firsts.extend([first_indices[j] for j in range(len(sentences[i:i+batch_size]))])
            masks = torch.ones_like(first_indices)
            zeros = torch.zeros_like(first_indices)
            masks = torch.where(first_indices==0, zeros, masks)
            lengths = masks.sum(-1).cpu().numpy()
            input_ids = input_ids.to(self.device)
            first_indices = first_indices.to(self.device)
            masks = masks.to(self.device)
            #print ("input_ids:\n", input_ids)
            #print ("masks:\n", masks)
            #print ("first_idx:\n", first_indices)
            #print ("lengths:\n", lengths)
            with torch.no_grad():
                self.network.eval()
                heads_pred, rels_pred = self.network.decode(first_indices, None, None, None, mask=masks, 
                    bpes=input_ids, first_idx=first_indices, input_elmo=None, lan_id=None, 
                    leading_symbolic=NUM_SYMBOLIC_TAGS)
                for j, length in enumerate(lengths):
                    heads.append(heads_pred[j][1:length])
                    rels.append(rels_pred[j][1:length])
                    #rels.append([self.rel_alphabet.get_instance(r) for r in rels_pred[j][1:length]])
        if type != "wp":
            for i in range(len(sents)):
                # remove ROOT token
                sents[i] = sents[i][1:]
                #heads[i] = heads[i] - 1

        return sents, heads, rels, inputs, firsts

if __name__ == "__main__":
    #model_path = "/users2/yxwang/work/experiments/adv/parsers/saves/ptb-biaf-roberta-v0"
    #lm_path = "/users2/yxwang/work/data/models/roberta-base"

    model_path = "/home/alex/work/codes/NeuroNLP2/experiments/models/parsing/robust"
    lm_path = "roberta-base"
    parser = Parser(model_path,  lm_path=lm_path)
    sents = ["I often do part-time jobs.",
             "Hi, my name is alexandar!",
             "What's your name?",
             "Nice to meet you today.",
             "Let's have breakerage togehter!"
            ]
    sents, heads, rels, inputs, firsts = parser.parse(sents, batch_size=3, type="wp")
    print ("sents:\n", sents)
    print ("heads:\n", heads)
    print ("rels:\n", rels)
