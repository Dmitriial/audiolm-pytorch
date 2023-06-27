from pathlib import Path

import torch
import argparse
from audiolm_pytorch import EncodecWrapper
from audiolm_pytorch import SoundStream, SoundStreamTrainer


def train(wav_folder: Path):
    encodec = EncodecWrapper()
    # Now you can use the encodec variable in the same way you'd use the soundstream variables below.

    soundstream = SoundStream(
        codebook_size=1024,
        rq_num_quantizers=8,
        rq_groups=2,
        # this paper proposes using multi-headed residual vector quantization - https://arxiv.org/abs/2305.02765
        attn_window_size=128,  # local attention receptive field at bottleneck
        attn_depth=2
        # 2 local attention transformer blocks - the soundstream folks were not experts with attention, so i took the liberty to add some. encodec went with lstms, but attention should be better
    )

    trainer = SoundStreamTrainer(
        soundstream,
        folder=str(wav_folder),
        batch_size=4,
        grad_accum_every=8,  # effective batch size of 32
        data_max_length_seconds=2,  # train on 2 second audio
        num_train_steps=1_000_000
    ).cuda()

    trainer.train()

    # after a lot of training, you can test the autoencoding as so

    audio = torch.randn(10080).cuda()
    recons = soundstream(audio, return_recons_only=True)  # (1, 10080) - 1 channel


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_folder", dest="wav_folder", type=Path, required=True,
                        help="Path to the folder with wav files (!)")

    opt = parser.parse_args()
    wav_folder: Path = opt.wav_folder
    assert wav_folder.exists() and wav_folder.is_dir(), f"Must be exists: {wav_folder}!"

    train(wav_folder=wav_folder)


if __name__ == '__main__':
    run()
