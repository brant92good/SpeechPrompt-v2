import os
from pathlib import Path

import torchaudio
from tqdm import tqdm

from preprocessor_base import PreprocessorBase
from FSD_asvspoof.dataset import get_dataloader




class Preprocessor(PreprocessorBase):
    def __init__(self, global_config, dataset_config):
        super().__init__(global_config, dataset_config)
        self.get_label()
        self.metadata_inthewild()

    def get_label(self):
        """Load CSV file and build {filename: label} dict"""
        import csv
        self.label_dict = {}

        csv_path = Path(self.datarc["metadata_csv"])
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # assume CSV has: filename,label
                fname = Path(row["filename"]).name
                self.label_dict[fname] = row["label"]

    def generate_manifest(self):
        """Generate manifest from InTheWild CSV metadata"""
        os.makedirs(Path(self.datarc["output_path"], "manifest"), exist_ok=True)

        import csv
        for split in ["train", "valid", "test"]:
            csv_path = Path(self.datarc[f"{split}_csv"])
            if not csv_path.exists():
                print(f"[WARN] Missing {split} CSV: {csv_path}, skipping...")
                continue

            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                wavs, labels, audio_paths = [], [], []

                for row in reader:
                    audio_path = Path(self.datarc["root_path"], row["filename"])
                    label = row["label"]
                    wav, sr = torchaudio.load(audio_path)

                    if sr != 16000:
                        wav = torchaudio.functional.resample(wav, sr, 16000)

                    wavs.append(wav.squeeze(0))
                    labels.append(label)
                    audio_paths.append(audio_path)

                # save manifest
                manifest_path = Path(self.datarc["output_path"], "manifest", f"{split}.manifest")
                with open(manifest_path, "w") as fout:
                    root_path = self.datarc["root_path"]
                    fout.write(f"{root_path}\n")
                    for wav, label, audio_path in zip(wavs, labels, audio_paths):
                        if not audio_path.exists():
                            torchaudio.save(audio_path, wav.unsqueeze(0), 16000)
                        relative_path = audio_path.relative_to(root_path)
                        fout.write(f"{relative_path}\t{len(wav)}\n")

    def get_class(self, file_name):
        """Return label from file_name"""
        fname = Path(file_name).name
        return self.label_dict[fname]
