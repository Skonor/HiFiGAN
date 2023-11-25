import gdown
import shutil
import os
from pathlib import Path


URL_LINKS = {
    'DeepSpeech2_clean': 'https://drive.google.com/uc?id=16N0g78RAdXeJkRFmPwzk2BstvMA1jB0A',
    'DeepSpeech2_clean_other': 'https://drive.google.com/uc?id=1lHzfCwiyUfod0wrgsmIQfigNo7oHfNmg',
    'DeepSpeech2_finetuned': 'https://drive.google.com/uc?id=1bQ1zGhUrOMYq4yzoquiMZ2nIYuzKOtDC'
}

def main():
    dir = Path(__file__).absolute().resolve().parent.parent
    for name in URL_LINKS:
        checkpoint_dir = dir / 'saved' / 'models' / 'checkpoints' / name
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        zip_pth = checkpoint_dir / (name + '.zip')
        model_pth = checkpoint_dir / 'model_weights.pth'
        if not model_pth.exists():
            gdown.download(URL_LINKS[name], str(zip_pth), quiet=False)
            shutil.unpack_archive(str(zip_pth), str(checkpoint_dir), "zip")
            os.remove(zip_pth)

if __name__ == "__main__":
    main()