import torch
import torchaudio
from torch.utils.data import Dataset
import pandas as pd

class SirenDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            sample_rate: int = 16000,
            transforms = None
    ):

        super().__init__()

        self.df = df
        self.sample_rate = sample_rate
        self.transforms = transforms

        self.class_labels = ['noise', 'siren']
        self.sample_duration = 1.0

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index: int):
        cur_sample = self.df.iloc[index]
        

        audio = self._load_audio(cur_sample['file_path'])

        if audio.shape[1] < self.sample_duration:
            repetitions = int(self.sample_duration / audio.shape[1] + 1)
            audio = audio.repeat(1, repetitions)

        onset = int(float(torch.rand(1,)) * (audio.shape[1] - self.sample_duration * self.sample_rate))
        offset = int(onset + self.sample_duration * self.sample_rate)

        audio = audio[:,onset:offset]
        
        label = torch.as_tensor(self.class_labels.index(cur_sample['file_class']), dtype=torch.long)

        if self.transforms is not None:
            audio = self.transforms(audio)
        
        return audio, label
    
    def _load_audio(self, file_path: str):
        audio, sample_rate = torchaudio.load(file_path)

        # Convert to mono if needed
        if len(audio) > 1:
            audio = audio[0]
            audio = audio[None, :]
        
        # Resample to target sample rate
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            audio = resampler(audio)

        # Apply peak normalization to shrink the amplitude range between 0 and 1
        norm_factor = torch.max(torch.abs(audio))
        if norm_factor > 0:
            audio /= norm_factor
        
        return audio