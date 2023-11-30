import argparse
import json
import os
import glob
from pathlib import Path

import torch
from tqdm import tqdm

import src.model as module_model
from src.utils import ROOT_PATH
from src.utils.parse_config import ConfigParser
import torchaudio
from src.utils.melspec import MelSpectrogram, MelSpectrogramConfig

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, test_data_folder):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model architecture
    generator = config.init_obj(config["arch"]["generator"], module_model)
    logger.info(generator)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["generator_state_dict"]
    if config["n_gpu"] > 1:
        generator = torch.nn.DataParallel(generator)
    generator.load_state_dict(state_dict)

    # prepare model for testing
    generator = generator.to(device)
    generator.eval()

    mel = MelSpectrogram(MelSpectrogramConfig).to(device)

    save_path = test_data_folder.parent / (test_data_folder.name + '_generated')

    save_path.mkdir(exist_ok=True, parents=True)

    for file_name in glob.glob(os.path.join(test_data_folder, '*.wav')):
        audio_wave, sr = torchaudio.load(file_name)
        audio_wave = audio_wave[0:1, :]
        with torch.no_grad():
            spectrogram = mel(audio_wave.to(device))
            gen_audio = generator(spectrogram=spectrogram)["gen_audio"]
        gen_audio = gen_audio.cpu()
        save_path_file = save_path / (Path(file_name).stem + '_generated.wav')
        torchaudio.save(save_path_file, gen_audio, sr)



if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # if `--test-data-folder` was provided, set it as a default test set
    if args.test_data_folder is not None:
        test_data_folder = Path(args.test_data_folder).absolute().resolve()
        assert test_data_folder.exists()


    main(config, test_data_folder)
