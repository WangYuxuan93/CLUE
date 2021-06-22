
def write_re(preds, examples, output_file):
	with open(output_file, "w") as writer:
		output = []
		for pred, example in zip(preds, examples):
			text = example.text
			items = [example.ent1, example.ent2, pred, example.text]
			output.append("\t".join(items))
		writer.write("\n".join(output))