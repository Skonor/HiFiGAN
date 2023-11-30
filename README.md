
## Overview

This repo contains HiFiGAN training framework on LJSpeech.

## Installation guide

```shell
pip install -r ./requirements.txt
```

To load checkpoints run:
```shell
python scripts/load_chheckpoints.py
```


## Training
To reproduce training do the following (All training was done on kaggle with LJspeech dataset)

Train model for 140 epochs on random crops of size 8192

```shell
python train.py -c src/configs/HiFiGan/train140.json
```

## Generation

For testing audio generation from mels do the following:

1. (Optional) Load checkpoint from training:
```shell
python scripts/load_checkpoints.py
```
This will create HiFi_140 folder in saved/models/checkpoints contaning model weigths file and training config

2. Run test.py (for test-other use librispeech_other.json config instead of librispeech_clean.json):
```shell
python test.py -r saved/models/checkpoints/HiFi_140/model_weights.pth -t test_data
```

This will create a new directory with the same name and suffix 'generated', containing generated wavs from mels. (For the 3 test wavs this directory already exists). 

You can use this script with any data_path to a directory, containing .wavs with -t option.

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.
