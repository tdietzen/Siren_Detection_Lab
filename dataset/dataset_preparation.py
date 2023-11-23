import torchaudio
import random
random.seed(4703)
from pathlib import Path
import pandas as pd

classes = ['noise', 'siren']
root_path = Path('dataset')
segment_len = 1

file_list = []
for cl in classes:
    for file_path in Path.joinpath(root_path, cl).rglob('*.wav'):
        if 'ambulance' in str(file_path):
            file_class = 'siren'
            split_prob = random.random()
            if split_prob < 0.8:
                split = 'train'
            elif split_prob < 0.9:
                split = 'validation'
            else:
                split = 'test'
        else:
            file_class = 'noise'
            split_prob = random.random()
            if split_prob < 0.8:
                split = 'train'
            elif split_prob < 0.9:
                split = 'validation'
            else:
                split = 'test'
        
        audio, sr = torchaudio.load(str(file_path))
        file_duration = float(round(len(audio[0]) / sr, 2))
        
        if segment_len == 'full':
            if(file_duration > 1):
                
                file_list.append({
                    'file_path': str(file_path),
                    'file_class': file_class,
                    'file_duration': file_duration,
                    'split': split
                })
            df_name = 'df_nature_split.csv'
        else:
            audio_start = 0.
            while audio_start < file_duration - segment_len:
                file_list.append({
                    'file_path': str(file_path),
                    'file_class': file_class,
                    'file_duration': file_duration,
                    'split': split,
                    'audio_start': audio_start,
                    'audio_end': audio_start + segment_len
                })  
                audio_start += segment_len
            df_name = 'df_nature_split_'+str(segment_len)+'s_segments.csv'
df = pd.DataFrame(file_list)

df.to_csv(Path.joinpath(root_path, df_name))