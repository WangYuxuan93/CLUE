import os
import sys
import gc
import json
import nltk
import jieba

#current_path = os.path.dirname(os.path.realpath(__file__))
#root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
#sys.path.append(root_path)

import time
import argparse
import math
import numpy as np
import torch
import random
import string

from .io import conllx_data
from .models.sdp_biaffine_parser import SDPBiaffineParser
from transformers import AutoTokenizer

import logging
#logger = logging.get_logger(__name__)
logger = logging.getLogger(__name__)

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
        elif type in ["nltk", "jieba"]:
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
            if type == "nltk":
                tokens = nltk.word_tokenize(text)
            else:
                tokens = jieba.cut(text)
            #if debug:
            #    print ("text:\n", text)
            #    print ("tokens:\n", tokens)
            #    print ("wps:\n", wps)
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
                #if debug:
                #    print ("token:{} | wp:{}".format(token[offset:], wp))
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
                    #if debug:
                    #    print ("token:{} | wp:{}".format(token[offset:], wp))
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


def split_ids(tokenizer, input_ids, debug=False):
    sep_id = tokenizer.sep_token_id
    cls_id = tokenizer.cls_token_id
    input_ids_list_a, input_ids_list_b = [], []

    # input_ids: [CLS] ... [SEP] ... [SEP]
    for ids in input_ids:
        sep_index = ids.index(sep_id)
        # [CLS] [CLS] ... [SEP] , the second cls is for root
        input_ids_list_a.append([cls_id]+ids[:sep_index]+[sep_id])
        input_ids_list_b.append([cls_id,cls_id]+ids[sep_index+1:])
        if debug:
            print ("ids:\n", ids)
            print ("ids_a:\n", input_ids_list_a[-1])
            print ("ids_b:\n", input_ids_list_b[-1])

    max_len_a = max([len(w) for w in input_ids_list_a])
    inputs_a = np.stack(
          [np.pad(a, (0, max_len_a - len(a)), 'constant', constant_values=tokenizer.pad_token_id) for a in input_ids_list_a])
    max_len_b = max([len(w) for w in input_ids_list_b])
    inputs_b = np.stack(
          [np.pad(b, (0, max_len_b - len(b)), 'constant', constant_values=tokenizer.pad_token_id) for b in input_ids_list_b])
    return inputs_a, inputs_b, input_ids_list_a, input_ids_list_b


def merge_first_ids(tokenizer, ids, ids_a, first_ids_a, ids_b=None, first_ids_b=None, debug=False):
    sep_id = tokenizer.sep_token_id
    first_ids_list = []
    for i in range(len(ids_a)):
        #print (ids_a)
        merge_ids = ids_a[i][1:]
        # minus 1 for removing the root token
        first_ids = [x-1 for x in first_ids_a[i]] + [len(merge_ids)-1]
        if ids_b is not None:
            # x -1 for rm the [CLS] of ids_b, offset = len(merge_ids)-1
            first_ids += [x-1+len(merge_ids)-1 for x in first_ids_b[i][1:]]
            # only 1 [SEP] for chinese bert/roberta
            merge_ids += ids_b[i][2:]
            first_ids += [len(merge_ids)-1]
        if debug:
            print ("ids_a:\n{}".format(ids_a[i]))
            print ("first_ids_a:\n",first_ids_a)
            if ids_b is not None:
                print ("ids_b:\n{}".format(ids_b[i]))
                print ("first_ids_b:\n",first_ids_b)
            print ("first_ids:\n",first_ids)
            print ("merge_ids:\n", merge_ids)
            print ("ids:\n", ids[i])
        assert merge_ids == ids[i][:len(merge_ids)]
        
        first_ids_list.append(first_ids)
    return first_ids_list


def first_ids_to_map(first_ids, lengths, debug=False):
    assert len(first_ids) == len(lengths)
    wid2wpid_list = []
    for fids, l in zip(first_ids, lengths):
        wid2wpid = {}
        wpid2wid = {}
        # first id starts from 1 (exclude cls token)
        # ends l-1 (exclu/de sep token)
        for i in range(l):
            if i in fids:
                wid2wpid[len(wid2wpid)] = [i]
                wpid2wid[i] = len(wid2wpid) - 1
            else:
                wid2wpid[len(wid2wpid) - 1].append(i)
                wpid2wid[i] = len(wid2wpid) - 1
        wid2wpid_list.append(wid2wpid)
    if debug:
        print ("first_ids:\n", first_ids)
        print ("lengths:\n", lengths)
        print ("wid2wpid_list:\n", wid2wpid_list)
    return wid2wpid_list


class SDPParser(object):
    def __init__(self, model_path, pretrained_lm="roberta", lm_path=None,
                use_pretrained_static=False, batch_size=16, parser_type="dm"):
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
                                    normalize_digits=False, pos_idx=3, task_type="sdp")
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


        self.network = SDPBiaffineParser(hyps, 0, num_words, num_chars, num_pos, num_rels,
                               device=device, pretrained_lm=pretrained_lm, lm_path=lm_path,
                               use_pretrained_static=use_pretrained_static, 
                               use_random_static=False, use_elmo=False, elmo_path=None, num_lans=1)


        self.network = self.network.to(device)
        self.network.load_state_dict(torch.load(model_name, map_location=device))

    def predict(self, first_ids, ids, debug=False):
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
            rels_pred = rels_pred * heads_pred
            #if debug:
            #    print ("ids:\n", ids)
            #    print ("first_ids:\n", first_ids)
            #    print ("heads_pred:\n", heads_pred)
            #    print ("rels_pred:\n", rels_pred)
            for j, length in enumerate(lengths):
                if debug:
                    print ("len:\n", length)
                    print ("heads_pred:\n", heads_pred[j])
                    print ("rels_pred:\n", rels_pred[j])
                heads.append(heads_pred[j][:length,:length])
                rels.append(rels_pred[j][:length,:length])
                #rels.append([self.rel_alphabet.get_instance(r) for r in rels_pred[j][1:length]])
        return heads, rels

    def parse_bpes(self, input_ids, masks, batch_size=None, has_b=False, has_c=False, expand_type="copy",
                    max_length=512, align_type="jieba", return_tensor=True, **kwargs):
        batch_size = batch_size if batch_size is not None else self.batch_size
        
        first_ids_list = []
        heads_a_list, rels_a_list = [], []
        heads_b_list, rels_b_list = [], []
        lengths = []
        for i in range(0, len(input_ids), batch_size):
            if i % 1024 == 0:
                print ("%d..."%i, end="")
                sys.stdout.flush()
            inputs = input_ids[i:i+batch_size]
            #print ("inputs:\n", inputs)
            mask = np.array(masks[i:i+batch_size]).sum(-1)
            lengths.extend(mask)
            max_len = max(mask)
            inputs_ = [x[:max_len] for x in inputs]
            #print ("inputs_:\n", inputs_)
            if has_b:
                ids_a, ids_b, input_ids_list_a, input_ids_list_b = split_ids(self.tokenizer, inputs_)
                ids_b = torch.from_numpy(ids_b)
            else:
                # add the root token, use as parser input
                ids_a = [[self.tokenizer.cls_token_id]+x for x in inputs_]
                # stay list for align 
                input_ids_list_a = [[self.tokenizer.cls_token_id]+x for x in inputs_]
                ids_b, input_ids_list_b, first_ids_b = None, None, None
            ids_a = torch.from_numpy(np.array(ids_a))
            fids_a, first_ids_a = get_first_ids(self.tokenizer, ids_a, align_type)
            heads_a, rels_a = self.predict(fids_a, ids_a)
            #wid2wpid_list_a = first_ids_to_map(first_ids_a, [len(x) for x in input_ids_list_a])
            #print ("heads_a:\n", heads_a)
            #print ("rels_a:\n", rels_a)
            heads_b, rels_b = None, None
            if ids_b is not None:
                fids_b, first_ids_b = get_first_ids(self.tokenizer, ids_b, align_type)
                heads_b, rels_b = self.predict(fids_b, ids_b)
                #wid2wpid_list_b = first_ids_to_map(first_ids_b, [len(x) for x in input_ids_list_b])
                #print ("heads_b:\n", heads_b)
                #print ("rels_b:\n", rels_b)
                heads_b_list.extend(heads_b)
                rels_b_list.extend(rels_b)

            first_ids = merge_first_ids(self.tokenizer, inputs_, input_ids_list_a, first_ids_a, input_ids_list_b, first_ids_b)
            first_ids_list.extend(first_ids)
            heads_a_list.extend(heads_a)
            rels_a_list.extend(rels_a)
        print ("")
        heads, rels = self.align_heads(self.tokenizer, first_ids_list, lengths, heads_a_list, rels_a_list, heads_b_list, rels_b_list, 
                        max_length=max_length, expand_type=expand_type)

        return heads, rels

    def align_heads(self, tokenizer, first_ids_list, lengths, heads_a, rels_a, 
                    heads_b=None, rels_b=None,
                    max_length=None, expand_type="copy", debug=False):
        null_label = self.parser_label_map[self.null_label]
        word_label = self.parser_label_map[self.word_label]

        heads_list = []
        rels_list = []

        wid2wpid_list = first_ids_to_map(first_ids_list, lengths)
        #print ("first_ids_list:\n", first_ids_list)
        #print ("wid2wpid_list:\n", wid2wpid_list)
        for i in range(len(heads_a)):
            # the i-th example
            wid2wpid = wid2wpid_list[i]
            first_ids = first_ids_list[i]
            if debug:
                print ("first_ids:\n", first_ids_list[i])
                print ("wid2wpid:\n",  wid2wpid_list[i])
                print ("heads_a:\n", heads_a[i])
                print ("rels_a:\n", rels_a[i])
                if heads_b:
                    print ("heads_b:\n", heads_b[i])
                    print ("rels_b:\n", rels_b[i])
            
            heads = torch.zeros(max_length, max_length, dtype=torch.long)
            rels = torch.zeros(max_length, max_length, dtype=torch.long)
            arc_indices = torch.nonzero(heads_a[i], as_tuple=False).detach().cpu().numpy()
            for x,y in arc_indices:
                label = rels_a[i][x][y]
                head_id = first_ids[y]
                mod_ids = wid2wpid[x]
                for mod_id in mod_ids:
                    # ignore out of range arcs
                    if mod_id < max_length and head_id < max_length:
                        heads[mod_id][head_id] = 1
                        rels[mod_id][head_id] = label

            if heads_b:
                # here only 1 [SEP] in between, offset is (len_a-1) +1
                offset = list(heads_a[i].size())[0] #+ 1
                arc_indices = torch.nonzero(heads_b[i], as_tuple=False).detach().cpu().numpy()
                for x,y in arc_indices:
                    label = rels_b[i][x][y]
                    x = offset + x
                    y = offset + y
                    head_id = first_ids[y]
                    mod_ids = wid2wpid[x]
                    for mod_id in mod_ids:
                        # ignore out of range arcs
                        if mod_id < max_length and head_id < max_length:
                            heads[mod_id][head_id] = 1
                            rels[mod_id][head_id] = label
                if debug:
                    torch.set_printoptions(profile="full")
                    print ("offset:",offset)
                    print ("heads (end):\n", heads)
                    print ("rels (end):\n", rels)
            heads_list.append(heads)
            rels_list.append(rels)

        heads = torch.stack(heads_list, dim=0)
        rels = torch.stack(rels_list, dim=0)

        if debug:
            print ("heads:\n", heads)
            print ("rels:\n", rels)

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

    model_path = "/home/alex/work/codes/NeuroNLP2/experiments/models/parsing/sdp_biaffine"
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
