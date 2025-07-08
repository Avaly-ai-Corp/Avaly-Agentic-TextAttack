import datetime
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
import os
from . import schema_harmonizer
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoConfig
from datasets import load_dataset, ClassLabel, Value
from huggingface_hub import login, HfApi

HF_TOKEN = os.getenv("HF_TOKEN") or ""
HF_TOKEN = HF_TOKEN.strip().strip("'\"")
if HF_TOKEN:
    login(HF_TOKEN)
    print("\n\nLogging in to Hugging Face Hub using token...")
    api = HfApi()
    hf_info = api.whoami()
    print(f"\nLogged in to Hugging Face Hub as {hf_info['name']}\n\n")
else:
    print(
        "\n\nNo Hugging Face token provided. Will not be able to access private models.\n\n"
    )

# Create cache directory
CACHE_DIR = Path.home() / ".cache" / "textattack"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Finding:
    model_from_hf: str
    dataset_from_hf: str
    recipe: str
    model_from_local: str
    dataset_from_local: str

    n: int = 0  # number of samples
    query_budget: int = 500

    def _check_model_config(self):

        # If model is from HF, load config from HF
        if self.model_from_hf:
            try:
                cfg = AutoConfig.from_pretrained(
                    self.model_from_hf,
                    trust_remote_code=os.getenv("HF_ALLOW_CODE_EVAL", "false").lower()
                    == "true",
                )
            except Exception as e:
                if "not a valid model" in str(e):
                    return {
                        "error": f"Model {self.model_from_hf} is not a valid model on the Hugging Face Hub."
                    }
                return {"error": f"Could not load config for {self.model_from_hf}: {e}"}
        # If model is from local, load config from local model directory
        else:
            try:
                cfg = AutoConfig.from_pretrained(
                    self.model_from_local,
                    trust_remote_code=os.getenv("HF_ALLOW_CODE_EVAL", "false").lower()
                    == "true",
                )
            except Exception as e:
                return {
                    "error": f"Could not load config file from path {self.model_from_local}: {e}"
                }

        # Check if model is a sequence-classification model using the config file
        if not any(
            a.endswith("ForSequenceClassification") for a in cfg.architectures or []
        ):
            return {
                "error": f"Model {self.model_from_hf if self.model_from_hf else self.model_from_local} is not a sequence-classification model: architectures={cfg.architectures})."
            }
        return {"success": True, "config": cfg}

    def _check_and_load_dataset_preview(self):
        ds_preview = None
        dataset_exists = False
        if self.dataset_from_local:
            # Check the extension of the dataset file and load the dataset using the appropriate loader
            # Limit the number of samples to 200 for each split to avoid memory issues
            # Assert the file extension to CSV, TSV, JSON, JSONL, or Parquet
            args = {}
            p = Path(self.dataset_from_local)
            if p.suffix.lower() in [".csv", ".tsv"]:
                ds_loader = "csv"
                if p.suffix.lower() == ".tsv":
                    args["delimiter"] = "\t"
            elif p.suffix.lower() in [".json", ".jsonl"]:
                ds_loader = "json"
            elif p.suffix.lower() in [".parquet"]:
                ds_loader = "parquet"
            args["data_files"] = self.dataset_from_local
            for split in ["validation[:200]", "test[:200]", "train[:200]"]:
                args["split"] = split
                try:
                    ds_preview = load_dataset(ds_loader, **args)
                    dataset_exists = True
                    break
                except Exception as e:
                    continue
        else:
            # If dataset is from HF, load the dataset using the load_dataset function
            # Limit the number of samples to 200 for each split to avoid memory issues
            for split in ["train[:200]", "validation[:200]", "test[:200]"]:
                try:
                    ds_preview = load_dataset(
                        self.dataset_from_hf,
                        split=split,
                        trust_remote_code=os.getenv(
                            "HF_ALLOW_CODE_EVAL", "false"
                        ).lower()
                        == "true",
                    )
                    dataset_exists = True
                    break
                except (FileNotFoundError, ValueError) as e:
                    if "doesn't exist on the Hub" in str(e):
                        return {
                            "error": f"Dataset '{self.dataset_from_hf}' doesn't exist on the Hugging Face Hub or cannot be accessed.",
                        }
                    elif "not found" in str(e):
                        return {
                            "error": f"Dataset '{self.dataset_from_hf}' not found on the Hugging Face Hub.",
                        }
                    elif "Config name is missing" in str(e):
                        return {
                            "error": f"Dataset '{self.dataset_from_hf}' config name is missing.",
                        }
                    else:
                        print(f"{e}")
                        continue

        if not dataset_exists:
            return {
                "error": f"Could not load dataset {self.dataset_from_hf} from any available split.",
            }
        # If dataset is loaded successfully, return the dataset
        return {"success": True, "dataset": ds_preview}

    def _check_dataset_label_compatibility_with_model(self, ds_preview, model_config):
        # Check if the dataset labels are compatible with the model labels
        label_variations = [
            "label",
            "labels",
            "label_name",
            "label_names",
            "label_id",
            "label_ids",
            "label_text",
            "label_texts",
            "label_value",
            "label_values",
            "intent",
            "intents",
            "sentiment",
            "rating",
        ]
        # Create case-insensitive mapping of column names
        column_map = {col.lower(): col for col in ds_preview.features}

        for label_variation in label_variations:
            if label_variation.lower() in column_map:
                label_name = column_map[label_variation.lower()]
                break
        else:
            return {
                "error": f"Dataset {self.dataset_from_hf if self.dataset_from_hf else self.dataset_from_local} : Could not detect 'label' column. Tried: {label_variations}",
            }

        label_feat = ds_preview.features[label_name]
        # Check if the label is a ClassLabel and has names
        if isinstance(label_feat, ClassLabel) and label_feat.names:
            ds_num = len(label_feat.names)
            ds_order = label_feat.names
        else:
            # If the label is not a ClassLabel, check if it is a Value
            if not isinstance(label_feat, Value):
                # If the label is not a Value, cast it to a Value
                ds_preview = ds_preview.cast_column(label_name, Value("int64"))

            ds_order = sorted(set(ds_preview[label_name]))
            ds_num = len(set(ds_preview[label_name]))

        # Check if the number of labels in the dataset is equal to or less than the number of labels in the model
        if ds_num > model_config.num_labels:
            return {
                "error": (
                    f"Label-count mismatch:\n"
                    f"  Model {self.model_from_hf if self.model_from_hf else self.model_from_local} expects {model_config.num_labels} labels: {list(model_config.id2label.values())}\n"
                    f"  Dataset {self.dataset_from_hf if self.dataset_from_hf else self.dataset_from_local} provides {ds_num} labels: {ds_order}\n"
                    f"  Number of labels in the dataset must be equal to or less than the number of labels in the model."
                )
            }

        TEXT_GUESSES = (
            "sentence",
            "sentence1",
            "review",
            "content",
            "text1",
            "text2",
            "tweet",
            "tweets",
        )
        if not "text" in ds_preview.features:
            for c in TEXT_GUESSES:
                if c.lower() in column_map:
                    break
            else:
                return {
                    "error": f"Dataset {self.dataset_from_hf if self.dataset_from_hf else self.dataset_from_local} : Could not detect 'text' column. Tried: 'text', {TEXT_GUESSES}",
                }
        return {"success": True, "label_name": label_name}

    def _check_local_model(self):
        # Check if local model directory exists and contains required files
        # Assert that the directory contains a config.json file
        # Assert that the directory contains a pytorch_model.bin or model.safetensors file
        # Assert that the directory contains a tokenizer_config.json file and either a vocab.txt and merges.txt file, a tokenizer.json file, or a spiece.model file
        if not os.path.isdir(self.model_from_local):
            return {
                "error": f"For local model, you must provide a path to a directory containing a config.json file along with the model."
            }
        elif not os.path.isfile(os.path.join(self.model_from_local, "config.json")):
            return {
                "error": f"For local model, the path must contain a config.json file."
            }
        else:
            if os.path.isfile(os.path.join(self.model_from_local, "pytorch_model.bin")):
                weight_file_type = "pytorch_model.bin"
            elif os.path.isfile(
                os.path.join(self.model_from_local, "model.safetensors")
            ):
                weight_file_type = "model.safetensors"
            else:
                return {
                    "error": f"For local model, the path must contain a pytorch_model.bin or model.safetensors file."
                }
        tok_cfg = os.path.isfile(
            os.path.join(self.model_from_local, "tokenizer_config.json")
        ) or os.path.isfile(os.path.join(self.model_from_local, "tokenizer.json"))
        has_vocab = os.path.isfile(os.path.join(self.model_from_local, "vocab.txt"))
        has_tokjson = os.path.isfile(
            os.path.join(self.model_from_local, "tokenizer.json")
        )
        has_sp = os.path.isfile(os.path.join(self.model_from_local, "spiece.model"))

        if not tok_cfg or not (has_vocab or has_tokjson or has_sp):
            return {
                "error": f"For local model, the path must contain a tokenizer_config.json file and either a vocab.txt and merges.txt file, a tokenizer.json file, or a spiece.model file."
            }
        # The weight_file_type is currently not used, but it is returned for potential future use
        return {"success": True, "weight_file_type": weight_file_type}

    def _check_local_dataset(self):
        # Check if local dataset file exists and is a valid file type from it's extension
        # Assert that the file extension is one of the following: .csv, .json, .jsonl, .tsv, .parquet
        if not os.path.isfile(self.dataset_from_local):
            return {"error": f"For local dataset, you must provide a path to a file."}
        p = Path(self.dataset_from_local)
        if not p.suffix.lower() in [".csv", ".json", ".jsonl", ".tsv", ".parquet"]:
            return {
                "error": f"For local dataset, the file must be a CSV, JSON, JSONL, TSV, or Parquet file."
            }
        return {"success": True}

    def _validate_model_and_dataset(self):
        # Validate local model
        if self.model_from_local:
            local_model_check = self._check_local_model()
            if local_model_check.get("error"):
                return local_model_check

        # Validate local dataset
        if self.dataset_from_local:
            local_dataset_check = self._check_local_dataset()
            if local_dataset_check.get("error"):
                return local_dataset_check

        # Validate model config
        model_config = self._check_model_config()
        if model_config.get("error"):
            return model_config

        # Check dataset and load preview
        ds_result = self._check_and_load_dataset_preview()
        if ds_result.get("error"):
            return ds_result
        ds_preview = ds_result["dataset"]

        # Check label compatibility between dataset and model
        label_check = self._check_dataset_label_compatibility_with_model(
            ds_preview, model_config["config"]
        )
        if label_check.get("error"):
            return label_check

        # If all checks pass, return the components
        return {
            "success": True,
            "model_config": model_config["config"],
            "dataset": ds_preview,
            "label_name": label_check["label_name"],
            "weight_file_type": (
                local_model_check["weight_file_type"] if self.model_from_local else None
            ),
        }

    def _process_attack_results(
        self, attack_summary_file, attack_details_file, attack_file_prefix, timestamp
    ):
        with open(attack_summary_file, "r") as f:
            # Load attack summary file
            data_summary = json.load(f)
            # Extract attack results from summary
            n_success = data_summary["Attack Results"]["Number of successful attacks:"]
            n_fail = data_summary["Attack Results"]["Number of failed attacks:"]
            n_skipped = data_summary["Attack Results"]["Number of skipped attacks:"]
            n_total = n_success + n_fail + n_skipped
            asr = n_success / n_total
            # Determine severity based on ASR
            severity = "high" if asr >= 0.6 else "medium" if asr >= 0.3 else "low"
            # Filter attack summary to include only relevant results
            filtered_data_summary = {
                "Attack Summary": {
                    "Attack Success Rate (ASR) Percentage": f"{asr * 100:.2f}%",
                    "Severity": severity,
                    "Number of successful attacks:": n_success,
                    "Number of failed attacks:": n_fail,
                    "Number of skipped attacks:": n_skipped,
                    "Model": self.model_from_hf if self.model_from_hf else self.model_from_local,
                    "Dataset": self.dataset_from_hf if self.dataset_from_hf else self.dataset_from_local,
                    "Recipe": self.recipe,
                    "Number of examples": self.n,
                    "Timestamp": timestamp,
                }
            }

            print("\n\n", filtered_data_summary, "\n\n")

        with open(attack_details_file, "r") as f:
            # Load attack details file
            data_details = pd.read_csv(f)
            print(data_details)
        # Convert attack details to JSON
        csv_to_json = data_details.to_json(orient="records")
        # Combine attack summary and details
        full_data_summary = {
            **filtered_data_summary,
            "attack_details": json.loads(csv_to_json),
        }
        # Write combined attack summary to file
        file_name = attack_file_prefix + "_attack_summary.json"
        with open(file_name, "w") as f:
            json.dump(full_data_summary, f)
        # Remove earlier two files
        if os.path.isfile(attack_details_file):
            os.remove(attack_details_file)
        if os.path.isfile(attack_summary_file):
            os.remove(attack_summary_file)
        # Return filtered and summarized attack results
        return full_data_summary

    def run(self):
        if self.model_from_local:
            self.model_from_hf = None
        if self.dataset_from_local:
            self.dataset_from_hf = None
        # Generate unique file prefix for attack results
        timestamp = datetime.datetime.now().astimezone().isoformat().split(".")[0]
        attack_file_prefix = os.path.join(
            "./",
            "REPORT_" + timestamp,
        )
        # Generate file names for attack details and summary
        attack_details_file = attack_file_prefix + "_attack_details.csv"
        attack_summary_file = attack_file_prefix + "_attack_short_summary.json"

        # Check if model and dataset are provided
        if not (self.model_from_hf or self.model_from_local):
            return {"error": "No model provided"}

        if not (self.dataset_from_hf or self.dataset_from_local):
            return {"error": "No dataset provided"}

        # Print parameters for debugging
        print("Parameters received:", "\n\n", "***", "\n\n")
        print("Model from HF:", self.model_from_hf, "\n\n")
        print("Dataset from HF:", self.dataset_from_hf, "\n\n")
        print("Model from Local:", self.model_from_local, "\n\n")
        print("Dataset from Local:", self.dataset_from_local, "\n\n")
        print("Recipe:", self.recipe, "\n\n")
        print("Query budget:", self.query_budget, "\n\n")
        print("Number of samples:", self.n, "\n\n")
        print("***", "\n\n")

        # Validate model and dataset
        validation_result = self._validate_model_and_dataset()
        if validation_result.get("error"):
            return validation_result

        # If validation passes, prepare dataset for attack. Call schema_harmonizer.prepare_dataset()
        try:
            dataset = schema_harmonizer.prepare_dataset(
                (
                    self.dataset_from_hf
                    if self.dataset_from_hf
                    else self.dataset_from_local
                ),
                AutoModelForSequenceClassification.from_pretrained(
                    self.model_from_hf if self.model_from_hf else self.model_from_local,
                    trust_remote_code=os.getenv("HF_ALLOW_CODE_EVAL", "false").lower()
                    == "true",
                ),
                label_name=validation_result["label_name"],
                dataset_type="local" if self.dataset_from_local else "hf",
            )
        except Exception as e:
            if "Could not map dataset" in str(e):
                return {
                    "error": f"The dataset labels are not compatible with the model labels.",
                    "stderr": str(e),
                }
            return {
                "error": f"An error occurred while harmonizing the dataset: {str(e)}"
            }

        # Set environment variable for model name. This is used by the TextAttack wrapper script when it is called via subprocess.
        os.environ["MODEL_NAME"] = (
            self.model_from_hf if self.model_from_hf else self.model_from_local
        )

        cmd = [
            "textattack",
            "attack",
            "--model-from-file",
            "./multi_tool_agent/app/hf_wrapper.py",
            "--dataset-from-file",
            str(dataset),
            "--recipe",
            self.recipe,
            "--num-examples",
            str(self.n),
            "--log-summary-to-json",
            attack_summary_file,
            "--log-to-csv",
            attack_details_file,
            "--query-budget",
            str(self.query_budget),
        ]
        print("\n\nCommand: ", cmd, "\n\n")
        try:
            # Capture both stdout and stderr to inspect output
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        # Handle errors from TextAttack process
        except subprocess.CalledProcessError as e:
            error_msg = "TextAttack process failed"
            if "out of memory" in e.stderr.strip().lower():
                error_msg = "Out of memory."
            if e.returncode == -9:
                error_msg = "Memory limit exceeded. Subprocess killed by OS."
            if "Could not map dataset" in e.stderr.strip().lower():
                error_msg = "The dataset labels do not match the model labels."
            # Send error back to streamlit
            return {
                "error": error_msg,
                "returncode": e.returncode,
                "stderr": e.stderr.strip(),
                "stdout": e.stdout.strip(),
            }
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}

        return self._process_attack_results(
            attack_summary_file, attack_details_file, attack_file_prefix, timestamp
        )
