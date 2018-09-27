import tensorflow as tf
import numpy as np
import os
import io

from utils import audio
from hparams import hparams

import librosa.display
import matplotlib.pyplot as plt

### Python implementation of the Griffin-Lim algorithm.

def griffin_lim(spectrogram, n_iter=hparams.griffin_lim_iters, n_fft=(hparams.num_freq - 1) * 2,
                            win=int(hparams.frame_length_ms / 1000 * hparams.sample_rate),
                            hop=int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)):

    spectrogram = tf.cast(tf.transpose(spectrogram), dtype=tf.complex64)
    approx_S = tf.identity(spectrogram)
    for i in range(n_iter + 1):
        approx_S = tf.expand_dims(approx_S, 0)
        inversed = tf.contrib.signal.inverse_stft(approx_S, win, hop, n_fft)
        approx_X = tf.squeeze(inversed, 0)
        if i < n_iter:
            est = tf.contrib.signal.stft(approx_X, win, hop, n_fft, pad_end=False)
            phase = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
            approx_S = spectrogram * phase

    return tf.real(approx_X)

def invert_spectrogram(spectrogram):
    S = _db_to_amp(_denormalize(spectrogram) + hparams.ref_level_db)  # Convert back to linear
    return _inv_preemphasis(griffin_lim(S ** 1.5))  # Reconstruct phase

# TensorFlow implementations of utility functions from utils/audio.py
def _denormalize(S):
    return (tf.clip_by_value(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db


def _db_to_amp(x):
    return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)


def _inv_preemphasis(x):
    N = tf.shape(x)[0]
    i = tf.constant(0)
    W = tf.zeros(shape=tf.shape(x), dtype=tf.float32)

    def condition(i, y):
        return tf.less(i, N)

    def body(i, y):
        tmp = tf.slice(x, [0], [i + 1])
        tmp = tf.concat([tf.zeros([N - i - 1]), tmp], -1)
        y = hparams.preemphasis * y + tmp
        i = tf.add(i, 1)
        return [i, y]

    final = tf.while_loop(condition, body, [i, W])

    y = final[1]

    return y

# File processing
if __name__ == '__main__':
    print("\nDue to dependence on open-source libraries, warning messages may appear.")

    print("Loading .wav files...")
    filenames = os.listdir("input")
    filenames = [f for f in filenames if f.endswith('.wav')]
    wavs = ["input/"+f for f in filenames]
    outputs_tf = ["output/rep."+f for f in filenames]
    wavs = [audio.load_wav(wav_path, hparams.sample_rate) for wav_path in wavs]

    print("Replicating .wav files...")
    spectrograms = [audio.spectrogram(wav).astype(np.float32) for wav in wavs]
    samples = [invert_spectrogram(spec) for spec in spectrograms]

    with tf.Session() as sess:
        samples = [sess.run(sample) for sample in samples]

    for gen, output in zip(samples, outputs_tf):
        out = io.BytesIO()
        audio.save_wav(gen, out)

        with open(output, "wb") as f:
            f.write(out.getvalue())

    print("Saving waveforms and spectrograms...")
    samples = [gen/hparams.byte_norm for gen in samples]
    gen_spectrograms = [audio.spectrogram(gen) for gen in samples]
    fig, ax = plt.subplots()
    plt.draw()
    for f, wav, gen in zip(filenames, wavs, samples):
        librosa.display.waveplot(wav, sr=hparams.sample_rate)
        ax = plt.gca()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        plt.draw()
        plt.savefig("figures/wave."+f+".png")
        plt.gcf().clear()

        librosa.display.waveplot(gen, sr=hparams.sample_rate)
        ax = plt.gca()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        plt.draw()
        plt.savefig("figures/wave.rep."+f+".png")
        plt.gcf().clear()

    for f, spec, gen in zip(filenames, spectrograms, gen_spectrograms):
        librosa.display.specshow(spec, sr=hparams.sample_rate, x_axis='time', y_axis='log')
        ax = plt.gca()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Log of Frequency (Hz)')
        plt.draw()
        plt.savefig("figures/spec."+f+".png")
        plt.gcf().clear()

        librosa.display.specshow(gen, sr=hparams.sample_rate, x_axis='time', y_axis='log')
        ax = plt.gca()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Log of Frequency (Hz)')
        plt.draw()
        plt.savefig("figures/spec.rep."+f+".png")
        plt.gcf().clear()


    print("Program complete.")

