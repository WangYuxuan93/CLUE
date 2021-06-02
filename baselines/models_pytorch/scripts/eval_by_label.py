import argparse

def load(filename, debug=False):
  dataset = []
  with open(filename, "r") as f:
    data = f.read().strip().split("\n\n")
    for (i, sent) in enumerate(data):
      sent = [line.split("\t") for line in sent.strip().split("\n")]
      #print (sent)
      words = [line[1] for line in sent]
      #heads = [int(line[8]) for line in sent]
      #rels = [line[10] for line in sent]
      #pred_heads = [int(line[9]) for line in sent]
      #pred_rels = [line[11] for line in sent]
      pred_senses = [line[13].split('.')[-1] if line[13] != '_' and line[12] == 'Y' else '_' for line in sent]
      pred_ids = []
      for j, line in enumerate(sent):
        if line[12] == 'Y':
            assert line[13] != '-'
            pred_ids.append(j)
      arg_labels = []
      for j in range(len(pred_ids)):
        arg_labels.append([line[14+j] if line[14+j] != '_' else '_' for line in sent])
      if debug:
        print ("words: ", words)
        print ("pred_ids: ", pred_ids)
        print ("sense: ", pred_senses)
        print ("arg: ", arg_labels)
      dataset.append({"words":words, "pred_ids":pred_ids, "sense":pred_senses, "arg":arg_labels})
  return dataset

def eval_prf(gold_data, sys_data):
  assert len(gold_data) == len(sys_data)
  n_gold, n_pred, n_corr, n_arc = {}, {}, {}, {}
  for gold_sent, sys_sent in zip(gold_data, sys_data):
    for i in range(len(gold_sent["arg"])):
      gold_labels = gold_sent["arg"][i]
      sys_labels = sys_sent["arg"][i]
      gold_pred_id = gold_sent["pred_ids"][i]
      sys_pred_id = sys_sent["pred_ids"][i]
      assert gold_pred_id == sys_pred_id
      for j, (gold_label, sys_label) in enumerate(zip(gold_labels, sys_labels)):
        if gold_label != '_':
          if gold_label not in n_gold:
            n_gold[gold_label] = 0
          n_gold[gold_label] += 1
        if sys_label != '_':
          if sys_label not in n_pred:
            n_pred[sys_label] = 0
          n_pred[sys_label] += 1
        if gold_label != '_' and sys_label != '_' and gold_label == sys_label:
          if gold_label not in n_corr:
            n_corr[gold_label] = 0
          n_corr[gold_label] += 1
  
  n_p, n_r, n_f = {}, {}, {}
  label_list = sorted(n_gold.items(), key=lambda x:x[1], reverse=True)
  print ("Label count in gold:", label_list)
  for tup in label_list:
    if tup[1] < 10: break
    l = tup[0]
    n_p[l] = float(n_corr[l]) / n_pred[l] if n_pred[l] > 0 else 0
    n_r[l] = float(n_corr[l]) / n_gold[l] if n_gold[l] > 0 else 0
    n_f[l] = 2*n_p[l]*n_r[l] / (n_p[l]+n_r[l]) if n_p[l]+n_r[l] > 0 else 0
    print ("Arg label={}, gold={}, pred={}, corr={}, P={:.4f}, R={:.4f}, F={:.4f}".format(
                    l, n_gold[l], n_pred[l], n_corr[l], n_p[l], n_r[l], n_f[l]))

  tot_gold = sum(n_gold.values())
  tot_pred = sum(n_pred.values())
  tot_corr = sum(n_corr.values())
  p = float(tot_corr) / tot_pred
  r = float(tot_corr) / tot_gold
  f = 2*p*r/(p+r)
  print ("Overall gold={}, pred={}, corr={}, P={:.4f}, R={:.4f}, F={:.4f}".format(tot_gold, tot_pred, tot_corr, p, r, f))

  #print ("precision:", n_p)
  #print ("recall:", n_r)
  #print ("f1:",n_f)

parser = argparse.ArgumentParser()

parser.add_argument("-g", default=None, type=str, required=True, help="gold file.")
parser.add_argument("-s", default=None, type=str, required=True, help="system file.")
#parser.add_argument("--type", default="sent", type=str, choices=["sent","arc"], 
#    help="type of distance. (sent: split by the length of sentence, arc: split by the length of arc)")
#parser.add_argument("--splits", default="10:20:30:40", type=str, help="split interval")
args = parser.parse_args()

#splits = [int(x) for x in args.splits.strip().split(':')]+[100]
gold_data = load(args.g)
print ("gold sents:", len(gold_data))
sys_data = load(args.s)
print ("sys sents:", len(sys_data))
eval_prf(gold_data, sys_data)#, args.type, splits)
