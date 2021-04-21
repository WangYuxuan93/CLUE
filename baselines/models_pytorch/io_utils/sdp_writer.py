
def write_sdp_conllu(pred_heads, pred_rels, examples, output_file, label_map, debug=True):
	with open(output_file, "w") as writer:
		sid = 0
		for pred, example in zip(pred_heads, examples):
			predict_rels = pred_rels[sid]
			words = example.words
			upos_tags = example.upos_tags
			xpos_tags = example.xpos_tags
			source_heads = example.source_heads
			source_rels = example.source_rels
			output = []
			if debug:
				print ("pred:\n", pred)
			for j, word in enumerate(words):
				items = ['_'] * 10
				items[0] = str(j+1)
				items[1] = word
				items[2] = word
				items[3] = upos_tags[j]
				items[4] = xpos_tags[j]
				head_info = []
				for k in range(len(pred)):
					if pred[j+1][k] == 1:
						label = label_map[predict_rels[j+1][k]]
						head_info.append(str(k)+":"+label)
				items[8] = "|".join(head_info)
				output.append('\t'.join(items))
			sid += 1
			writer.write('\n'.join(output)+'\n\n')