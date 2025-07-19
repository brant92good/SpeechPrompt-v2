import os
from pathlib import Path

import torchaudio
from tqdm import tqdm

from FSD_asvspoof.dataset import ASVSpoofDataset, get_dataloader
from preprocessor_base import PreprocessorBase


class Preprocessor(PreprocessorBase):
    def __init__(self, global_config, dataset_config):
        super().__init__(global_config, dataset_config)
        self.get_label()

    def get_label(self):
        self.label_dict = {}

        if self.datarc["custom_validation"]:
            meta_data = Path(self.datarc["custom_validation_protocal_path"])
            with open(meta_data, "r") as f:
                lines = f.readlines()
                for line in lines:
                    self.label_dict[f'{line.split(" ")[1]}.flac'] = line.split(" ")[4].replace("\n", "")
        else:
            for split in ["train", "dev", "eval"]:
                if split == "train":
                    meta_data = Path(
                        self.datarc["root_path"], "ASVspoof2019_LA_cm_protocols", f"ASVspoof2019.LA.cm.{split}.trn.txt"
                    )
                else:
                    meta_data = Path(
                        self.datarc["root_path"], "ASVspoof2019_LA_cm_protocols", f"ASVspoof2019.LA.cm.{split}.trl.txt"
                    )
                with open(meta_data, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        self.label_dict[f'{line.split(" ")[1]}.flac'] = line.split(" ")[4].replace("\n", "")

    def generate_manifest(self):
        os.makedirs(
            os.path.join(self.datarc["output_path"], "manifest"),
            exist_ok=True,
        )

        mapping = {"train": "train", "valid": "dev", "test": "eval"}
        for split in ["train", "valid", "test"]:
            if split == "train":
                meta_data = Path(
                    self.datarc["root_path"],
                    "ASVspoof2019_LA_cm_protocols",
                    f"ASVspoof2019.LA.cm.{mapping[split]}.trn.txt",
                )
            else:
                meta_data = Path(
                    self.datarc["root_path"],
                    "ASVspoof2019_LA_cm_protocols",
                    f"ASVspoof2019.LA.cm.{mapping[split]}.trl.txt",
                )
            dataset = ASVSpoofDataset(
                meta_data=meta_data,
                root_path=Path(self.datarc["root_path"], f"ASVspoof2019_LA_{mapping[split]}", "flac"),
            )
            dataloader = get_dataloader(
                dataset=dataset,
                batch_size=self.datarc["batch_size"],
                num_workers=self.datarc["num_workers"],
                collate_fn=dataset.collate_fn,
            )

            with open(Path(self.datarc["output_path"], "manifest", f"{split}.manifest"), "w") as f:
                root_path = self.datarc["root_path"]
                f.write(f"{root_path}\n")
                for wavs, labels, audio_pathes in tqdm(dataloader, desc=split):
                    for wav, label, audio_path in zip(wavs, labels, audio_pathes):
                        if not audio_path.exists():
                            torchaudio.save(audio_path, wav.unsqueeze(0), 16000)

                        relative_path = audio_path.relative_to(self.datarc["root_path"])
                        f.write(f"{relative_path}\t{str(len(wav))}\n")

    def generate_manifest_valid(self):
        if not self.datarc["custom_validation"] or self.datarc["custom_validation"] == "False":
            print("[INFO] Custom validation is not enabled in dataset config. Skipping manifest generation for validation.")
            return
        if not self.datarc["custom_validation_protocal_path"] or not self.datarc["custom_validation_audio_path"]:
            print("[ERROR] Custom validation path is not specified. Please set the path in dataset config.")
            return
        os.makedirs(
            os.path.join(self.datarc["output_path"], "manifest"), #modify later
            exist_ok=True,
        )
        meta_data = Path(self.datarc["custom_validation_protocal_path"])
        dataset = ASVSpoofDataset(
            meta_data=meta_data,
            root_path=Path(self.datarc["custom_validation_audio_path"]),
        )
        dataloader = get_dataloader(
            dataset=dataset,
            batch_size=self.datarc["batch_size"],
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn,
        )
        with open(Path(self.datarc["output_path"], "manifest", "test.manifest"), "w") as f: #modify later
            root_path = self.datarc["root_path"]
            f.write(f"{root_path}\n")
            for wavs, labels, audio_pathes in tqdm(dataloader, desc="test"):
                for wav, label, audio_path in zip(wavs, labels, audio_pathes):
                    if not audio_path.exists():
                        torchaudio.save(audio_path, wav.unsqueeze(0), 16000)
                    relative_path = audio_path.relative_to(root_path)
                    f.write(f"{relative_path}\t{str(len(wav))}\n")

    def generate_manifest_asv21(self):
        os.makedirs(
            os.path.join(self.datarc["output_path"], "manifest"),
            exist_ok=True,
        )

        
        meta_data_dirty = Path(
            "/home/brant/Projects/clean-SP-v2/storage/temp/asvspoof2021/keys/LA/CM/trial_metadata.txt"
        )
        meta_data = Path(
            "/home/brant/Projects/clean-SP-v2/storage/temp/asvspoof2021/keys/LA/CM/converted_to_2019_format.txt"
        )
        with open(meta_data_dirty, 'r') as fin, open(meta_data, 'w') as fout:
            for line in fin:
                parts = line.strip().split()
                if len(parts) >= 6:
                    col1 = parts[0]
                    col2 = parts[1]
                    col4 = parts[4]
                    if col4 == 'bonafide':
                        col4 = '-'
                    col5 = parts[5]
                    fout.write(f"{col1} {col2} - {col4} {col5}\n")

        dataset = ASVSpoofDataset(
            meta_data=meta_data,
            root_path=Path(self.datarc["root_path"], f"ASVspoof2021_LA_eval", "flac"),
        )
        dataloader = get_dataloader(
            dataset=dataset,
            batch_size=self.datarc["batch_size"],
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn,
        )

        with open(Path(self.datarc["output_path"], "manifest", f"test.manifest"), "w") as f:
            root_path = self.datarc["root_path"]
            f.write(f"{root_path}\n")
            for wavs, labels, audio_pathes in tqdm(dataloader, desc="test"):
                for wav, label, audio_path in zip(wavs, labels, audio_pathes):
                    if not audio_path.exists():
                        torchaudio.save(audio_path, wav.unsqueeze(0), 16000)

                    relative_path = audio_path.relative_to(self.datarc["root_path"])
                    f.write(f"{relative_path}\t{str(len(wav))}\n")

    def get_class(self, file_name):
        class_name = self.label_dict[file_name.split("/")[-1]]
        return class_name
