# Timbre-Evolution
Genetic algorithm for generating timbres from audio samples

This project details a method for generating new audio waveforms given a set of input audio samples. The magnitudes of the short-time Fourier transforms of the inputs are decomposed by non-negative matrix factorization, and the resulting components can be placed in linear combinations to generate spectrograms. A genetic algorithm is implemented to stochastically train the weights of these linear combinations, and output audio samples can be estimated from the generated spectrograms via the Griffin-Lim method.
