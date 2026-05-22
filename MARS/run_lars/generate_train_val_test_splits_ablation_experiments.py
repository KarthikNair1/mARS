import argparse
import csv
import math
import os
import pickle
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional


DEFAULT_INPUT_SPLIT_PKL = "/gpfs/data/mayerhoeferlab/projects/myeloma/datafiles_for_mars/splits/train_val_test_split.pkl"
DEFAULT_OUTPUT_DIR = "/gpfs/data/mayerhoeferlab/projects/myeloma/datafiles_for_mars/splits/ablation_experiments"
DEFAULT_TRAIN_DF_CSV = "/gpfs/data/mayerhoeferlab/projects/myeloma/datafiles_for_mars/train_df.csv"
DEFAULT_SEED = 2025
DEFAULT_NUM_LEVELS = 8
DEFAULT_MIN_TRAIN_EXAMPLES = 1
DEFAULT_LABEL_COL = "PET_jco"
DEFAULT_ACCESSION_COL = "accession"


def _as_list(items: Any, split_name: str) -> List[Any]:
	if isinstance(items, list):
		return items
	if hasattr(items, "tolist"):
		out = items.tolist()
		if isinstance(out, list):
			return out
	raise ValueError(f"Split '{split_name}' is not list-like. Got type: {type(items)}")


def load_base_split(split_pkl_path: str) -> Dict[str, List[Any]]:
	with open(split_pkl_path, "rb") as f:
		split = pickle.load(f)

	if not isinstance(split, dict):
		raise ValueError(f"Expected split pickle to contain a dict, got: {type(split)}")

	missing = {k for k in ["train", "val", "test"] if k not in split}
	if missing:
		raise ValueError(f"Missing required keys in split file: {sorted(missing)}")

	return {
		"train": _as_list(split["train"], "train"),
		"val": _as_list(split["val"], "val"),
		"test": _as_list(split["test"], "test"),
	}


def load_labels_from_csv(train_df_csv: str, accession_col: str, label_col: str) -> Dict[str, Any]:
	label_map: Dict[str, Any] = {}
	with open(train_df_csv, "r", newline="") as f:
		reader = csv.DictReader(f)
		if accession_col not in reader.fieldnames or label_col not in reader.fieldnames:
			raise ValueError(
				f"CSV must contain columns '{accession_col}' and '{label_col}'. Found: {reader.fieldnames}"
			)

		for row in reader:
			accession = str(row[accession_col])
			label = row[label_col]
			if label is None or str(label).strip() == "":
				continue
			label_map[accession] = label

	if not label_map:
		raise ValueError(f"No labels were loaded from {train_df_csv} using column '{label_col}'.")
	return label_map


def build_balanced_train_order(train_ids: List[Any], label_map: Dict[str, Any], seed: int) -> List[Any]:
	class_to_ids: Dict[str, List[Any]] = defaultdict(list)
	missing_labels = []

	for accession in train_ids:
		acc_key = str(accession)
		if acc_key not in label_map:
			missing_labels.append(accession)
			continue
		class_to_ids[str(label_map[acc_key])].append(accession)

	if missing_labels:
		raise ValueError(
			f"{len(missing_labels)} training accessions are missing labels in the training CSV. "
			f"Example missing accession: {missing_labels[0]}"
		)

	rng = random.Random(seed)
	for label in class_to_ids:
		rng.shuffle(class_to_ids[label])

	labels_sorted = sorted(class_to_ids.keys())
	ordered: List[Any] = []
	idx = 0
	while len(ordered) < len(train_ids):
		label = labels_sorted[idx % len(labels_sorted)]
		bucket = class_to_ids[label]
		if bucket:
			ordered.append(bucket.pop())
		idx += 1

	return ordered


def build_proportional_train_order(train_ids: List[Any], label_map: Dict[str, Any], seed: int) -> List[Any]:
	class_to_ids: Dict[str, List[Any]] = defaultdict(list)
	missing_labels = []

	for accession in train_ids:
		acc_key = str(accession)
		if acc_key not in label_map:
			missing_labels.append(accession)
			continue
		class_to_ids[str(label_map[acc_key])].append(accession)

	if missing_labels:
		raise ValueError(
			f"{len(missing_labels)} training accessions are missing labels in the training CSV. "
			f"Example missing accession: {missing_labels[0]}"
		)

	rng = random.Random(seed)
	for label in class_to_ids:
		rng.shuffle(class_to_ids[label])

	labels_sorted = sorted(class_to_ids.keys())
	total = sum(len(class_to_ids[label]) for label in labels_sorted)
	target_props = {label: len(class_to_ids[label]) / total for label in labels_sorted}
	selected_counts = {label: 0 for label in labels_sorted}

	ordered: List[Any] = []
	for _ in range(total):
		next_pos = len(ordered) + 1
		best_label = None
		best_deficit = None

		# Pick the class currently most under target proportion for this prefix.
		for label in labels_sorted:
			if not class_to_ids[label]:
				continue
			expected = target_props[label] * next_pos
			deficit = expected - selected_counts[label]
			if best_deficit is None or deficit > best_deficit:
				best_deficit = deficit
				best_label = label

		if best_label is None:
			break

		ordered.append(class_to_ids[best_label].pop())
		selected_counts[best_label] += 1

	return ordered


def generate_default_train_sizes(total_train: int, min_examples: int, num_levels: int) -> List[int]:
	if total_train <= 0:
		raise ValueError("Training split is empty.")
	if min_examples < 1:
		raise ValueError("min_examples must be >= 1")
	if num_levels < 1:
		raise ValueError("num_levels must be >= 1")

	min_examples = min(min_examples, total_train)
	if num_levels == 1:
		return [total_train]

	raw_sizes = []
	for i in range(num_levels):
		t = i / (num_levels - 1)
		size = int(round(min_examples * math.exp(math.log(total_train / min_examples) * t)))
		raw_sizes.append(max(min_examples, min(total_train, size)))

	raw_sizes[0] = min_examples
	raw_sizes[-1] = total_train

	unique_sorted = sorted(set(raw_sizes))
	if len(unique_sorted) >= num_levels:
		return unique_sorted[: num_levels - 1] + [total_train]

	needed = num_levels - len(unique_sorted)
	candidates = [x for x in range(min_examples, total_train + 1) if x not in set(unique_sorted)]
	if not candidates:
		return unique_sorted

	step = max(1, len(candidates) // (needed + 1))
	additions = []
	pos = step
	while len(additions) < needed and pos < len(candidates):
		additions.append(candidates[pos])
		pos += step

	while len(additions) < needed:
		additions.append(candidates[-1 - len(additions)])

	final_sizes = sorted(set(unique_sorted + additions))
	if final_sizes[-1] != total_train:
		final_sizes[-1] = total_train
	return final_sizes[:num_levels]


def fractions_to_train_sizes(
	fractions: List[float], total_train: int, min_examples: int
) -> List[int]:
	if total_train <= 0:
		raise ValueError("Training split is empty.")
	if min_examples < 1:
		raise ValueError("min_examples must be >= 1")

	train_sizes = []
	for frac in fractions:
		if frac <= 0 or frac > 1:
			raise ValueError(f"Each fraction must be in (0, 1]. Got: {frac}")
		n = int(round(frac * total_train))
		n = max(min_examples, min(total_train, n))
		train_sizes.append(n)

	return sorted(set(train_sizes))


def make_train_subset(train_ids_ordered: List[Any], num_examples: int) -> List[Any]:
	if num_examples < 1:
		raise ValueError(f"num_examples must be >= 1, got: {num_examples}")
	n_total = len(train_ids_ordered)
	n_keep = min(num_examples, n_total)
	return train_ids_ordered[:n_keep]


def save_split(split_dict: Dict[str, List[Any]], out_path: str) -> None:
	with open(out_path, "wb") as f:
		pickle.dump(split_dict, f)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_split_pkl", type=str, default=DEFAULT_INPUT_SPLIT_PKL)
	parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--train_df_csv", type=str, default=DEFAULT_TRAIN_DF_CSV)
	parser.add_argument("--label_col", type=str, default=DEFAULT_LABEL_COL)
	parser.add_argument("--accession_col", type=str, default=DEFAULT_ACCESSION_COL)
	parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
	parser.add_argument(
		"--stratification_mode",
		type=str,
		choices=["proportional", "balanced"],
		default="proportional",
		help="How to order training IDs before taking prefixes. 'proportional' preserves overall label ratios; 'balanced' enforces near-equal class mixing.",
	)
	parser.add_argument(
		"--train_sizes",
		type=int,
		nargs="+",
		default=None,
		help="Absolute number of train examples to retain at each level.",
	)
	parser.add_argument(
		"--fractions",
		type=float,
		nargs="+",
		default=None,
		help="Fractions of the train split to retain (e.g., 0.01 0.05 0.1 0.25 0.5 0.75 1.0).",
	)
	parser.add_argument("--num_levels", type=int, default=DEFAULT_NUM_LEVELS)
	parser.add_argument("--min_train_examples", type=int, default=DEFAULT_MIN_TRAIN_EXAMPLES)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	os.makedirs(args.output_dir, exist_ok=True)

	base_split = load_base_split(args.input_split_pkl)
	train_ids = base_split["train"]
	val_ids = base_split["val"]
	test_ids = base_split["test"]

	print(f"Loaded base split from: {args.input_split_pkl}")
	print(f"Base sizes -> train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")

	label_map = load_labels_from_csv(args.train_df_csv, args.accession_col, args.label_col)
	if args.stratification_mode == "balanced":
		train_ids_ordered = build_balanced_train_order(train_ids, label_map, args.seed)
	else:
		train_ids_ordered = build_proportional_train_order(train_ids, label_map, args.seed)

	overall_label_counts = Counter(str(label_map[str(x)]) for x in train_ids_ordered)
	overall_total = len(train_ids_ordered)
	overall_label_props = ";".join(
		[f"{k}:{v}/{overall_total}={v/overall_total:.4f}" for k, v in sorted(overall_label_counts.items())]
	)
	print(
		f"Stratification mode: {args.stratification_mode}. Overall train label distribution: {overall_label_props}"
	)

	if args.train_sizes is not None and args.fractions is not None:
		raise ValueError("Use either --train_sizes or --fractions, not both.")

	if args.fractions is not None:
		train_sizes = fractions_to_train_sizes(
			fractions=args.fractions,
			total_train=len(train_ids_ordered),
			min_examples=args.min_train_examples,
		)
		print(f"Using train fractions: {sorted(set(args.fractions))}")
	elif args.train_sizes is None:
		train_sizes = generate_default_train_sizes(
			total_train=len(train_ids_ordered),
			min_examples=args.min_train_examples,
			num_levels=args.num_levels,
		)
	else:
		train_sizes = sorted(set(args.train_sizes))
		if train_sizes[0] < args.min_train_examples:
			raise ValueError(
				f"Minimum train size must be >= {args.min_train_examples}. Got {train_sizes[0]}"
			)

	print(f"Using train sizes: {train_sizes}")

	requested_fractions = sorted(set(args.fractions)) if args.fractions is not None else None
	manifest = []
	for num_examples in train_sizes:
		train_subset = make_train_subset(train_ids_ordered, num_examples)
		label_counts = Counter(str(label_map[str(x)]) for x in train_subset)
		label_counts_str = ";".join([f"{k}:{v}" for k, v in sorted(label_counts.items())])
		actual_fraction = len(train_subset) / len(train_ids_ordered)

		split_out = {
			"train": train_subset,
			"val": val_ids,
			"test": test_ids,
		}

		out_name = f"train_val_test_split_trainN_{len(train_subset)}.pkl"
		out_path = os.path.join(args.output_dir, out_name)
		save_split(split_out, out_path)

		manifest.append(
			(
				num_examples,
				f"{actual_fraction:.6f}",
				len(train_subset),
				len(val_ids),
				len(test_ids),
				label_counts_str,
				out_name,
			)
		)
		print(
			f"Saved {out_name}: train={len(train_subset)} ({actual_fraction:.4f}), val={len(val_ids)}, test={len(test_ids)}, labels={label_counts_str}"
		)

	manifest_path = os.path.join(args.output_dir, "manifest.tsv")
	with open(manifest_path, "w") as f:
		if requested_fractions is not None:
			f.write(f"requested_fractions\t{','.join(str(x) for x in requested_fractions)}\n")
		f.write(
			"requested_train_examples\tactual_train_fraction\ttrain_size\tval_size\ttest_size\ttrain_label_counts\tfilename\n"
		)
		for row in manifest:
			f.write("\t".join([str(x) for x in row]) + "\n")

	print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
	main()
