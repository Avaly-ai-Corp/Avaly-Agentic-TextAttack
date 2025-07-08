from datasets import ClassLabel, Value
from textattack.datasets import Dataset as TADataset
from pathlib import Path
import datasets
import os


class DatasetPrep:
    def __init__(self, ds, model_id2label):
        self.ds = ds
        self.model_id2label = model_id2label
        self.column_map = {col.lower(): col for col in self.ds.column_names}

    def _get_column_name(self, variations, default=None):
        for var in variations:
            if var.lower() in self.column_map:
                return self.column_map[var.lower()]
        return default

    def _refresh_column_map(self):
        # Refresh column map to ensure it's up to date after renaming columns
        self.column_map = {c.lower(): c for c in self.ds.column_names}

    def unify_text(self):

        # List of possible text column names. Can be modified to include more or fewer column names.
        TEXT_GUESSES = (
            "sentence",
            "sentence1",
            "review",
            "content",
            "text1",
            "text2",
            "text",
            "tweet",
            "tweets",
        )

        # Check for paired columns first
        # If both columns- Sentence1 and Sentence2 exist, combine them into a single text column
        if all(col.lower() in self.column_map for col in ["sentence1", "sentence2"]):
            s1_col = self.column_map["sentence1"]
            s2_col = self.column_map["sentence2"]
            self.ds = self.ds.map(
                lambda e: {"text": e[s1_col] + " [SEP] " + e[s2_col]},
                remove_columns=[s1_col, s2_col],
            )
            print(
                "\n\nCombining 'sentence1' and 'sentence2' into 'text' column and using 'text' as the text column\n\n"
            )
            self._refresh_column_map()

        # If both columns- Text1 and Text2 exist, combine them into a single text column
        elif all(col.lower() in self.column_map for col in ["text1", "text2"]):
            t1_col = self.column_map["text1"]
            t2_col = self.column_map["text2"]
            self.ds = self.ds.map(
                lambda e: {"text": e[t1_col] + " [SEP] " + e[t2_col]},
                remove_columns=[t1_col, t2_col],
            )
            print(
                "\n\nCombining 'text1' and 'text2' into 'text' column and using 'text' as the text column\n\n"
            )
            self._refresh_column_map()
        else:
            # Try to find a single text column
            text_col = self._get_column_name(TEXT_GUESSES)
            if text_col and text_col != "text":
                self.ds = self.ds.rename_column(text_col, "text")
                self._refresh_column_map()
                print(
                    f"\n\nRenaming {text_col} column to 'text' and using 'text' as the text column\n\n"
                )
            elif not text_col:
                raise ValueError(f"No text column found in {self.ds.column_names}")

    def unify_labels(self, label_name):

        # Get label features and values
        feat = self.ds.features[label_name]

        # If the label is a ClassLabel and has names, convert the label to an integer
        if isinstance(feat, ClassLabel) and feat.names:
            ds_order = list(feat.names)
            # Write changes to dataset
            self.ds = self.ds.map(
                lambda x: {label_name: int(x[label_name])},
                desc="Ensuring ClassLabel values are ints",
            )
        else:
            first = self.ds[label_name][0]
            if isinstance(first, str):

                # Example: self.ds[label_name] = ["negative", "neutral", "positive", "negative", "neutral", "positive", "positive"]
                # unique_labels = ["negative", "neutral", "positive"]
                unique_labels = sorted(set(self.ds[label_name]))

                # Example: label_to_idx = {"negative": 0, "neutral": 1, "positive": 2}
                label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

                # Example: self.ds[label_name] = [0, 1, 2, 0, 1, 2, 2]
                self.ds = self.ds.map(
                    lambda x: {label_name: label_to_idx[x[label_name]]},
                    desc="Mapping string labels → ids",
                )
            else:
                # Convert to int64 and handle negative values
                # Example: self.ds[label_name] = [-1, 1, 0, 0, -1, 1, -1, 0, 1, 0, 0, -1, 0]
                self.ds = self.ds.cast_column(label_name, Value("int64"))

                # Get unique labels and sort them to maintain consistent order
                # Example: unique_labels = [-1, 0, 1]
                unique_labels = sorted(set(self.ds[label_name]))

                # Create a mapping from original labels to 0-based indices
                # Example: if we have unique_labels = [-1, 0, 1], label_to_idx = {-1:0, 0:1, 1:2}
                label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

                # Map the labels to 0-based indices
                # This will replace all original labels such as -1, 0, 1 with 0, 1, 2
                self.ds = self.ds.map(
                    lambda x: {label_name: label_to_idx[x[label_name]]},
                    desc="Converting labels to 0-based indices",
                )

            ds_order = [str(lbl) for lbl in unique_labels]

        num_ds_labels = len(ds_order)

        # Example: model_order = ["Negative", "Positive"]
        model_keys = sorted(self.model_id2label.keys())
        model_order = [self.model_id2label[k] for k in model_keys]
        print("\n\nModel Labels Order", model_order, "\n\n")
        print("\n\nDataset Labels Order", ds_order, "\n\n")

        if num_ds_labels == len(model_keys) and all(
            str(idx) == ds_order[idx]
            or ds_order[idx].lower() == model_order[idx].lower()
            for idx in range(num_ds_labels)
        ):
            # If number of labels match, and labels match, we can use the index as the label mapping between the dataset and the model
            label_map = {i: model_keys[i] for i in range(num_ds_labels)}
        else:
            # Example: ds_order_lower = ["negative", "positive"]
            ds_order_lower = [str(lbl).lower() for lbl in ds_order]
            # Example: model_order_lower = ["negative", "positive"]
            model_order_lower = [str(lbl).lower() for lbl in model_order]
            # Example: inv = {"negative": 0, "positive": 1}
            inv = {lbl: idx for idx, lbl in enumerate(model_order_lower)}

            # Map dataset labels to model labels
            label_map = {}
            for i, ds_lbl in enumerate(ds_order_lower):
                if ds_lbl in inv:
                    label_map[i] = model_keys[inv[ds_lbl]]
                else:
                    # Try partial matching
                    for full_lbl, idx in inv.items():
                        if full_lbl.startswith(ds_lbl):
                            label_map[i] = model_keys[idx]
                            break
                    else:
                        raise ValueError(
                            f"Could not map dataset label '{ds_order[i]}' to any model label. "
                            f"Dataset labels: {ds_order}, Model labels: {model_order}"
                        )

        # Rename label column to 'label' for consistency
        if label_name != "label":
            self.ds = self.ds.rename_column(label_name, "label")
            self._refresh_column_map()

        print("\n\nDataset -> Model Label Mapping (after harmonization):", label_map)
        print("\n\nDataset -> Model Label Mapping (before harmonization):")
        for ds_label, model_label in label_map.items():
            print(f'Dataset {ds_label} -> Model "{self.model_id2label[model_label]}"')
        print("\n")
        return label_map

    def export(self, cache_dir: Path, label_map=None):
        # Verify required columns exist
        if "text" not in self.ds.column_names or "label" not in self.ds.column_names:
            raise ValueError("Dataset must have 'text' and 'label' columns")

        # Create pairs with proper label mapping
        pairs = []
        for ex in self.ds:
            text = ex["text"]
            label = int(ex["label"])
            # Apply label mapping if provided
            if label_map is not None:
                try:
                    label = label_map[label]
                except KeyError as e:
                    raise ValueError(
                        f"Label {label} not found in label map. Available labels: {list(label_map.keys())}"
                    ) from e
            pairs.append((text, label))

        model_order = list(self.model_id2label.values())
        # Write dataset to file. Textattack will use this file as dataset for attack. This is the harmonized dataset.
        path = cache_dir / "dataset.py"
        path.write_text(
            "from textattack.datasets import Dataset\n"
            f"dataset = Dataset({pairs!r}, "
            f"label_map={label_map!r}, "
            f"label_names={model_order!r})\n",
            encoding="utf-8",
        )
        return path


def prepare_dataset(dataset, model, label_name, dataset_type):
    try:
        if dataset_type == "local":
            args = {}
            p = Path(dataset)
            if p.suffix.lower() in [".csv", ".tsv"]:
                ds_loader = "csv"
                if p.suffix.lower() == ".tsv":
                    args["delimiter"] = "\t"
            elif p.suffix.lower() in [".json", ".jsonl"]:
                ds_loader = "json"
            elif p.suffix.lower() in [".parquet"]:
                ds_loader = "parquet"
            args["data_files"] = dataset
            ds = datasets.load_dataset(ds_loader, **args)
        else:
            # Load full dataset
            ds = datasets.load_dataset(
                *dataset if not isinstance(dataset, str) else [dataset],
                trust_remote_code=os.getenv("HF_ALLOW_CODE_EVAL", "false").lower()
                == "true",
            )

        # Get first available split and limit to n samples. This is to prevent memory issues.
        for split in ["train", "validation", "test"]:
            if split in ds:
                n = min(5000, len(ds[split]))
                ds = ds[split]

                # ***
                # IMPORTANT: REMOVE IN PRODUCTION TO USE FULL DATASET
                # This will limit the dataset to 5000 samples.
                ds = ds.select(range(n))  # <--- REMOVE IN PRODUCTION
                # IMPORTANT: REMOVE IN PRODUCTION TO USE FULL DATASET
                # ***

                break
        else:
            raise ValueError(
                f"Could not load dataset {dataset} from any available split"
            )

        prep = DatasetPrep(ds, model.config.id2label)
        prep.unify_text()
        try:
            label_map = prep.unify_labels(label_name)
        except Exception as e:
            raise ValueError(f"Failed to unify labels: {str(e)}")
        return prep.export(Path.home() / ".cache" / "textattack", label_map)

    except Exception as e:
        raise ValueError(f"Failed to prepare dataset: {str(e)}")
