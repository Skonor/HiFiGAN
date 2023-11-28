import random
from pathlib import Path
from random import shuffle


import torchaudio
import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from src.base import BaseTrainer
from src.logger.utils import plot_spectrogram_to_buf
from src.utils import inf_loop, MetricTracker
from src.utils.melspec import MelSpectrogram, MelSpectrogramConfig

from src.utils import ROOT_PATH


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            generator,
            discriminator,
            criterion_G,
            criterion_D,
            optimizer_G,
            optimizer_D,
            metrics,
            config,
            device,
            dataloaders,
            lr_scheduler_G=None,
            lr_scheduler_D=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(generator, discriminator, criterion_G, criterion_D, optimizer_G, 
                         optimizer_D, metrics, config, device, lr_scheduler_G, lr_scheduler_D)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        self.mel = MelSpectrogram(MelSpectrogramConfig()).to(self.device)

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler_G = lr_scheduler_G
        self.lr_scheduler_D = lr_scheduler_D
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "loss_G", "loss_D", "mel_loss", "fm_loss", "adv_loss_G",
            *[m.name for m in self.metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "mel_loss", *[m.name for m in self.metrics], writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["spectrogram", "audio"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm_G(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.generator.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _clip_grad_norm_D(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.discriminator.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.generator.train()
        self.discriminator.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch, generated_batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.generator.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    for p in self.discriminator.parameters():
                        if p.grad is not None:
                            del p.grad 
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss G: {:.6f}; Loss D: {:.6f}".format(
                        epoch, self._progress(batch_idx), generated_batch["loss_G"].item(), generated_batch["loss_D"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate G", self.lr_scheduler_G.get_last_lr()[0]
                )
                self.writer.add_scalar(
                    "learning rate D", self.lr_scheduler_D.get_last_lr()[0]
                )
                self._log_predictions(**batch, **generated_batch)
                #self._log_spectrogram(batch["spectrogram"])
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch - 1:
                break
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        self._log_test_audio()

        if self.lr_scheduler_D is not None:
            self.lr_scheduler_D.step()
        if self.lr_scheduler_G is not None:
            self.lr_scheduler_G.step()

        return log

    def _detach_batch(self, batch):
        detached_batch = {}
        for key in batch:
            detached_batch[key] = batch[key].detach()
        return detached_batch

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)

        generated_batch = self.generator(**batch)

        self.optimizer_D.zero_grad()

        discriminated_batch = self.discriminator(**batch, **self._detach_batch(generated_batch))

        disc_losses = self.criterion_D(**discriminated_batch)

        loss_D = disc_losses["loss_D"]

        loss_D.backward()

        self._clip_grad_norm_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()

        generated_batch["gen_spectrogram"] = self.mel(generated_batch["gen_audio"])
        discriminated_batch = self.discriminator(**batch, **generated_batch)

        generator_losses = self.criterion_G(**batch, **generated_batch, **discriminated_batch)

        loss_G = generator_losses["loss_G"]

        loss_G.backward()

        self._clip_grad_norm_G()
        self.optimizer_G.step()

        metrics.update("loss_D", loss_D.item())
        for loss in ["loss_G", "mel_loss", "fm_loss", "adv_loss_G"]:
            metrics.update(loss, generator_losses[loss])

        for met in self.metrics:
            metrics.update(met.name, met(**batch, **generated_batch, **discriminated_batch))

        generated_batch["loss_G"] = loss_G
        generated_batch["loss_D"] = loss_D
        return batch, generated_batch

    def process_batch_eval(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)

        generated_batch = self.generator(**batch)
        generated_batch["gen_melsepc"] = self.mel(generated_batch["gen_audio"])
        mel_loss  = F.l1_loss(batch["spectrogram"], generated_batch["gen_melsepc"])
        metrics.update("mel_loss", mel_loss.item())

        for met in self.metrics:
            metrics.update(met.name, met(**batch, **generated_batch))

        return batch, generated_batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.generator.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
             for batch_idx, batch in enumerate(tqdm(dataloader, desc="train", total=self.len_epoch)):
                batch, generated_batch = self.process_batch_eval(batch, self.evaluation_metrics)

        self.writer.set_step(epoch * self.len_epoch, part)
        self._log_scalars(self.evaluation_metrics)
        self._log_predictions(**batch, **generated_batch)
        #self._log_spectrogram(batch["spectrogram"])

        return self.evaluation_metrics.result()
    
    def _log_test_audio(self):
        self.generator.eval()
        test_path = ROOT_PATH / "test_data"
        rows = {}
        for file_name in ["audio_1.wav", "audio_2.wav", "audio_3.wav"]:
            audio_path = test_path / file_name
            audio_wave, sr = torchaudio.load(str(audio_path))
            audio_wave = audio_wave[0:1, :]
            spectrogram = self.mel(audio_wave.to(self.device))
            gen_audio = self.generator(spectrogram=spectrogram)["gen_audio"]
            
            rows[audio_path.name] = {
                "gen_audio": self.writer.wandb.Audio(gen_audio.squeeze().detach().cpu().numpy(), sample_rate=sr),
            }
        
        self.writer.add_table("generated_test", pd.DataFrame.from_dict(rows, orient="index"))


    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
            self,
            gen_audio,
            audio,
            audio_path,
            examples_to_log=5,
            *args,
            **kwargs,
    ):
        
        if self.writer is None:
            return

        tuples = list(zip(audio.cpu(), gen_audio.cpu(), audio_path))
        shuffle(tuples)
        rows = {}
        for audio, gen_audio, audio_path in tuples[:examples_to_log]:

            rows[Path(audio_path).name] = {
                "orig_audio": self.writer.wandb.Audio(audio_path), 
                "cutted_audio": self.writer.wandb.Audio(audio.squeeze().detach().numpy(), sample_rate=22050),
                "generated_audio": self.writer.wandb.Audio(gen_audio.squeeze().detach().numpy(), sample_rate=22050),
            }
        self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, model, norm_type=2):
        parameters = model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
