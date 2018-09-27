# Timbre-Evolution
Genetic algorithm for generating timbres from audio samples

This project details a method for generating new audio waveforms given a set of input audio samples. The magnitudes of the short-time Fourier transforms of the inputs are decomposed by non-negative matrix factorization, and the resulting components can be placed in linear combinations to generate spectrograms. A genetic algorithm is implemented to stochastically train the weights of these linear combinations, and output audio samples can be estimated from the generated spectrograms via the Griffin-Lim method.\
A full explanation can be found in report.pdf

Dependencies (latest versions used as of 03/11/18)
>Python3 (3.6)\
>TensorFlow (1.6)\
>Librosa (0.6)\
>scipy (0.19)\
>numpy (1.14)

Audio samples are provided in ./samples\
Pre-generated results are saved in ./samples/results\
Hyperparameters can be adjusted in ./hparams.py

griffin_lim.py

Griffin-Lim method for replicating audio samples.

Run\
Once the dependencies have been installed, the file can be run in terminal using the following command:

>python griffin_lim.py

Audio samples are read from the folder ./input (must be .wav format)\
Spectrograms and waveforms are written to ./figures\
Replicated audio files are written to ./output

genetic_algo_NMF.py

A genetic algorithm for creating new timbres.

Run\
The file is executed in the same manner:

>python genetic_algo_NMF.py

Audio samples are read from the folder ./input (must be .wav format)\
NMF decomposition figures are written to ./figures\
New audio files are written to ./output