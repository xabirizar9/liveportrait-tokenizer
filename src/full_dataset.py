import csv
import torch
import torchaudio

from src import MotionDataset
from pathlib import Path

class SNACMotionTextDataset(MotionDataset):
    def __init__(self, data_path: str, split: str = 'train', val_split: float = 0.2, seed: int = 42, 
                 compute_stats: bool = True, num_threads: int = 8, device: str = 'cuda'):
        super().__init__(data_path, split, val_split, seed, compute_stats, num_threads, device)
        
        self.audio_dir = Path(data_path) / "audio"
        self.text_dir = Path(data_path) / "transcripts"
        self.audio_paths = []
        self.text_paths = []

        for pickle_path in self.pickle_paths:
            audio_path = self.audio_dir / f"{pickle_path.stem}.wav"
            self.audio_paths.append(audio_path)
            text_path = self.text_dir / f"{pickle_path.stem}.csv"
            self.text_paths.append(text_path)

    def _resample_audio(self, audio, sample_rate):
        if sample_rate != 24000:
            resampler = torchaudio.transforms.Resample(sample_rate, 24000)
            audio = resampler(audio)
        return audio
        
    def __getitem__(self, idx):
        motion_item = super().__getitem__(idx)
        audio_path = self.audio_paths[idx]
        audio, sample_rate = torchaudio.load(audio_path)
        audio = self._resample_audio(audio, sample_rate)
        text_path = self.text_paths[idx]

        words = []
        with open(text_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                words.append(row['word'])
        text = ' '.join(words)
        # convert to mono
        audio = audio.mean(dim=0, keepdim=True)
        item = {
            "audio": audio,
            "motion": motion_item,
            "text": text
        }
        return item
