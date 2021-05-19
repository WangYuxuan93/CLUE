
def write_conll09_predicate_sense(predict_labels, examples, output_file):
	with open(output_file, "w") as writer:
		for pred, example in zip(predict_labels, examples):
			pred_ids = example.pred_ids
			words = example.words
			tags = example.pos_tags
			heads = example.gold_heads
			rels = example.gold_rels
			output = []
			i = 0
			for j, word in enumerate(words):
				items = ['_'] * 14
				items[0] = str(j+1)
				items[1] = word
				items[2] = word
				items[3] = word
				items[4] = tags[j]
				items[5] = tags[j]
				items[8] = str(heads[j])
				items[9] = str(heads[j])
				items[10] = rels[j]
				items[11] = rels[j]
				if j in pred_ids:
					items[-2] = 'Y'
					items[-1] = pred[i]
					i += 1
				rel_items = ['_' for _ in range(len(pred_ids))]
				output.append('\t'.join(items+rel_items))
			writer.write('\n'.join(output)+'\n\n')


def write_conll09_argument_label(predict_labels, examples, output_file):
	with open(output_file, "w") as writer:
		previous_sid = None
		words_list = []
		tags_list = []
		heads_list = []
		rels_list = []
		labels_list = []
		predicates_list = []

		prev_words = None
		prev_tags = None
		prev_heads = None
		prev_rels = None
		predicates = []
		labels = []
		for pred, example in zip(predict_labels, examples):
			#example.show()
			if example.sid != previous_sid:
				previous_sid = example.sid
				words_list.append(prev_words)
				tags_list.append(prev_tags)
				heads_list.append(prev_heads)
				rels_list.append(prev_rels)
				predicates_list.append(predicates)
				labels_list.append(labels)
				# start collecting data for a new sentence
				prev_words = example.tokens_a
				prev_tags = example.pos_tags
				prev_heads = example.gold_heads
				prev_rels = example.gold_rels
				predicates = [example.pred_id]
				labels = [pred]
			else:
				predicates.append(example.pred_id)
				labels.append(pred)
		# add the last one
		words_list.append(prev_words)
		tags_list.append(prev_tags)
		heads_list.append(prev_heads)
		rels_list.append(prev_rels)
		predicates_list.append(predicates)
		labels_list.append(labels)
		# rm the first empty data
		words_list = words_list[1:]
		tags_list = tags_list[1:]
		heads_list = heads_list[1:]
		rels_list = rels_list[1:]
		predicates_list = predicates_list[1:]
		labels_list = labels_list[1:]

		#print ("words_list:\n", words_list)
		#print ("predicates_list:\n", predicates_list)
		#print ("labels_list:\n", labels_list)

		for i in range(len(words_list)):
			pred_ids = predicates_list[i]
			words = words_list[i]
			tags = tags_list[i]
			heads = heads_list[i]
			rels = rels_list[i]
			labels = labels_list[i]
			output = []
			#print ("labels: ", labels)
			# there is not real predicate
			if len(labels[0]) == 0:
				pred_ids = []
			for j, word in enumerate(words):
				items = ['_'] * 14
				items[0] = str(j+1)
				items[1] = word
				items[2] = word
				items[3] = word
				items[4] = tags[j]
				items[5] = tags[j]
				items[8] = str(heads[j])
				items[9] = str(heads[j])
				items[10] = rels[j]
				items[11] = rels[j]
				if j in pred_ids:
					items[-2] = 'Y'
					items[-1] = word+'.01'
				if len(labels[0]) > 0:
					rel_items = [label[j] if label[j] not in ['O','<PAD>'] else '_' for label in labels]
					output.append('\t'.join(items+rel_items))
				else:
					output.append('\t'.join(items))
			writer.write('\n'.join(output)+'\n\n')


def write_conll09_end2end(preds, golds, label_map, examples, output_file, debug=False):
	with open(output_file, "w") as writer:
		for i, pred in enumerate(preds):
			if debug:
				print ("pred:\n", pred)
				print ("gold:\n", golds[i])
			example = examples[i]

			pred_ids = [x-1 for x in example.pred_ids]
			words = example.words[1:]
			tags = example.pos_tags[1:]
			heads = [h-1 for h in example.gold_heads[1:]]
			rels = example.gold_rels[1:]
			output = []

			num_prds = len(pred_ids)
			# omit the first <ROOT> token
			for j, word in enumerate(words):
				items = ['_'] * 14
				items[0] = str(j+1)
				items[1] = word
				items[2] = word
				items[3] = word
				items[4] = tags[j]
				items[5] = tags[j]
				items[8] = str(heads[j])
				items[9] = str(heads[j])
				items[10] = rels[j]
				items[11] = rels[j]
				if j in pred_ids:
					items[-2] = 'Y'
					prd_label = label_map[pred[j+1,0]]
					assert prd_label.startswith('prd')
					items[-1] = prd_label.split(':')[1]
				rel_items = ['_' for _ in range(num_prds)]
				for k, prd_id in enumerate(pred_ids):
					arg_label = label_map[pred[j+1,prd_id+1]]
					if arg_label != 'O':
						assert arg_label.startswith('arg')
						rel_items[k] = arg_label.split(':')[1]

				output.append('\t'.join(items+rel_items))
			writer.write('\n'.join(output)+'\n\n')