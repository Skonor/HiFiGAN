
## Overview

This repo contains framework for ASR training as well as implemented training and evaluation procedure for DeepSpeech2 on librispeech dataset.

Repo contains following features:

1. Beamsearch and Beamsearch with language model
2. Noise, Pitch Shift, Gain and TimeStretch augmentations (as well as their random versions)


## Installation guide

```shell
pip install -r ./requirements.txt
```

To load LM model for beamsearch run:
 ```shell
python scripts/load_lm.py
 ```

To load checkpoints run:
```shell
python scripts/load_chheckpoints.py
```


## Training
To reproduce training do the following (All training was done on kaggle with librispeech dataset: [insert link])

1. Train DeepSpeech2 on librispeech clean100 and clean360 for 80 epochs (len epoch = 100 steps)

```shell
python train.py -c src/configs/DeepSpeech2_configs/baseline_clean360.json
```

2. Finetune model for 50 epochs on train-other-500:
```
python train.py -c src_configs/DeepSpeech2_configs/finetune_other.json -f saved/checkpoints/DeepSpeech2_clean/model_weights.pth
```

(Here I used checkpoint loaded by load_checkpoints.py script. You can change -f path to your local weights destination)

## Evaluation

For evaluating models on librispeech test-clean and test-other do th following:

1. Load LM for beamsearch:
 ```shell
python scripts/load_lm.py
 ```

2. (Optional) Load checkpoint from training:
```shell
python scripts/load_chheckpoints.py
```
This will create DeepSpeech2 in saved/models/checkpoints contaning model weigths file and training config

You can skip this step if you are using you own model

3. Run test.py (for test-other use librispeech_other.json config instead of librispeech_clean.json):
```shell
python test.py -b 32 -c src/configs/test_configs/DeepSpeech2/librispeech_clean.json -r saved/checkpoints/DeepSpeech2_finetuned/model_weights.pth
```

This will create output.json file containing argmax, beamsearch (beam_size=10) and LM beamsearch (beam_size=500) predictions

4. Run evaluation.py:
```shell
python evaluation.py -o output.json
```
This will print out WER and CER metrics for each of prediction methods


## Results

For DeepSpeech2 model trained on clean part of the librispeech we get the following results:

| Method | test-clean CER| test-clean WER | test-other CER | test-other WER |
|--------|---------------|----------------|----------------|----------------|
| Argmax |        8.43   |     26.64      |     24.86      |     57.48      |
| Beamsearch |  8.23     |     25.90      |     24.35      |     56.37      |
| LM Beamsearch |  5.80  |     14.45      |     21.12      |     39.35      |

And after finetuning on other:

| Method | test-clean CER| test-clean WER | test-other CER | test-other WER |
|--------|---------------|----------------|----------------|----------------|
| Argmax |        7.87   |     24.75      |     18.75      |     46.08      |
| Beamsearch |  7.63     |     23.82      |     18.25      |     44.66      |
| LM Beamsearch |  5.41  |     13.30      |     15.01      |     29.18      |

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.
