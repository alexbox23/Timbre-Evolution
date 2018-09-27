import scipy,sklearn
import numpy as np
import os
import io

import random

from utils import audio
from hparams import hparams

import librosa,librosa.display
import matplotlib.pyplot as plt

### Procedural generation of audio samples using NMF decomposition

class genetic_algorithm():
    # Container class for evolutionary iterations.

    class organism():
        # Class representation of a single generated audio sample.

        def __init__(self):
            # self.genes format of a single element:
            #   [((species, spectrum_index), amplitude), ((species, spectrum_index), amplitude), ...]
            self.genes = [list() for i in range(hparams.num_genes)]
            self.species = -1
            self.waveform = []
            self.fitness = 0

        def insert_gene(self, index, spectra, amplitudes):
            self.genes[index] = list(zip(spectra, amplitudes))

        def store_waveform(self, waveform):
            self.waveform = waveform


    def __init__(self, wavs):
        # self.gene_pool format of a single key-value pair:
        #   species: [(spectrum, temp_activation), (spectrum, temp_activation), ...]
        self.originals = wavs
        self.spectrograms = [librosa.stft(wav) for wav in wavs]
        self.gene_pool = self.extract_genes(self.spectrograms)
        self.curr_generation = self.init_generation()

    def extract_genes(self, spectrograms):
        ret = {}
        for i in range(len(spectrograms)):
            X = np.absolute(spectrograms[i])
            W, H = librosa.decompose.decompose(X, n_components=hparams.num_genes, sort=True)
            genes = []
            for n in range(hparams.num_genes):
                w = W[:,n]
                h = H[n]
                genes.append((w, h))
            ret[i] = genes
        return ret

    def new_organism(self):
        # Each temporal gene initialized with original spectrum and random spectrum
        # random new spectral gene has 75-99% total amplitude
        gene_randomness = (3 + random.random())/4
        spawn = self.organism()
        spawn.species = random.randint(0, len(self.originals)-1)
        for index in range(hparams.num_genes):
            rand_s = random.randint(0, len(self.originals)-1)
            rand_w = random.randint(0, hparams.num_genes-1)
            amplitude = (3 + random.random())/4
            spawn.insert_gene(index, [(spawn.species, index), (rand_s, rand_w)], [1 - amplitude, amplitude])

        waveform = self.generate_waveform(spawn)
        spawn.store_waveform(waveform)
        return spawn

    def init_generation(self):
        new_generation = []
        while len(new_generation) < hparams.gen_size:
            new_generation.append(self.new_organism())
        return new_generation

    def next_generation(self, curr_gen):
        # half of each generation survives
        num_survivors = int(hparams.gen_size/2)

        for org in self.curr_generation:
            fit_a = int(100*self.fitness_MSE(org))
            fit_b = int(self.fitness_coherence(org))
            org.fitness = (fit_b, fit_a)
            # print(org.species, org.fitness)

        self.curr_generation = sorted(self.curr_generation, key=lambda org:org.fitness)
        survivors = self.curr_generation[-1*num_survivors:]

        new_generation = []
        # fraction of new generation not spawned from survivors starts at 1/4
        # begins to decrease halfway through the simulation
        room_for_new = min(int(num_survivors/2), int(num_survivors*2*(1 - curr_gen/hparams.num_generations)))
        cycle = 0
        while len(new_generation) < (hparams.gen_size - room_for_new):
            progenitor = survivors[cycle%num_survivors]
            cycle += 1
            spawn = self.organism()
            spawn.species = progenitor.species
            for index in range(hparams.num_genes):
                # Crossover mutation: copy gene from a peer
                if random.random() < hparams.crossover_rate:
                    rand_s = random.randint(0, num_survivors-1)
                    new_gene = survivors[rand_s].genes[index]
                    spawn.genes[index] = new_gene.copy()

                spectra = []
                amplitudes = []
                for s, amp in progenitor.genes[index]:
                    spectra.append(s)
                    amplitudes.append(amp)

                # Expansion mutation: append a new gene
                if random.random() < hparams.expand_gene_rate:
                    rand_s = random.randint(0, len(self.originals)-1)
                    rand_w = random.randint(0, hparams.num_genes-1)
                    amp = random.random()*hparams.mutate_amplitude
                    spectra.append((rand_s, rand_w))
                    for i in range(len(amplitudes)):
                        amplitudes[i] -= (amp / len(amplitudes))
                    amplitudes.append(amp)

                # Normal mutation: redistribute gene amplitudes
                rand_a = random.randint(0, len(amplitudes)-1)
                amp = random.random()*hparams.mutate_amplitude
                for i in range(len(amplitudes)):
                    if i == rand_a:
                        amplitudes[i] += amp
                    else:
                        amplitudes[i] -= (amp / len(amplitudes))

                # Re-normalize gene amplitudes
                to_remove = []
                for i in range(len(amplitudes)):
                    if amplitudes[i] < 0:
                        to_remove.append(i)
                    elif amplitudes[i] > 1:
                        amplitudes[i] = 1
                for i in sorted(to_remove, reverse=True):
                    amplitudes.pop(i)
                    spectra.pop(i)

                spawn.insert_gene(index, spectra, amplitudes)

            waveform = self.generate_waveform(spawn)
            spawn.store_waveform(waveform)
            new_generation.append(spawn)

        while len(new_generation) < hparams.gen_size:
            new_generation.append(self.new_organism())

        return new_generation

    def fitness_MSE(self, org):
        freq, S = scipy.signal.welch(org.waveform)
        MSE = np.zeros(len(S))
        
        for waveform in self.originals:
            freq, Sx = scipy.signal.welch(waveform)
            for f in range(len(S)):
                MSE[f] += (S[f] - Sx[f]) ** 2
        MSE /= len(self.originals)

        return -1 * sum(MSE)

    def fitness_coherence(self, org):
        avg_C = []
        for waveform in self.originals:
            freq, Cxy = scipy.signal.coherence(org.waveform, waveform)
            avg_C.append(Cxy)
        avg_C = [sum(i) for i in zip(*avg_C)]

        return sum(avg_C)

    def generate_waveform(self, org):
        W = []
        H = []
        phi = []
        for index in range(len(org.genes)):
            gene = org.genes[index]
            spectra = []
            for s, amplitude in gene:
                species, num = s
                spectrum = self.gene_pool[species][num][0]
                spectra.append(amplitude * spectrum)
            temp_activation = self.gene_pool[org.species][index][1]
            phase = np.angle(self.spectrograms[org.species])

            W.append(sum(spectra))
            H.append(temp_activation)
            phi.append(phase)

        max_length = max(len(h) for h in H)
        components = []
        for w, h, p in zip(W, H, phi):
            if len(h) < max_length:
                h = np.pad(h, (0, max_length-len(h)), 'minimum')
            if len(p) < max_length:
                p = np.pad(p, (0, max_length-len(p)), 'minimum')

            Y = scipy.outer(w,h)*np.exp(1j*phase)
            y = librosa.istft(Y)
            components.append(y)

        return sum(components)

    def iterate(self):
        print("Running genetic algorithm...")
        for i in range(hparams.num_generations):
            self.curr_generation = self.next_generation(i)
            print("Completed generation "+str(i+1)+".")


# File processing
if __name__ == '__main__':
    random.seed(2018)
    print("\nDue to dependence on open-source libraries, warning messages may appear.")

    print("Loading .wav files...")
    filenames = os.listdir("input")
    filenames = [f for f in filenames if f.endswith('.wav')]
    wavs = []
    for f in filenames:
        wav = audio.load_wav("input/"+f, hparams.sample_rate)
        wavs.append(wav)

    species_names = filenames
    GA = genetic_algorithm(wavs)
    GA.iterate()

    print("Saving gene spectrograms...")
    fig = plt.subplots()
    plt.draw()
    for i in range(len(GA.originals)):
        genes = GA.gene_pool[i]
        for n in range(len(genes)):
            w, h = genes[n]
            plt.subplot(len(genes), 2, 2*n+1)
            spectrum = np.log10(np.maximum(1e-5,w))
            plt.plot(spectrum)
            plt.ylim(-5, spectrum.max())
            plt.xlim(0, len(spectrum))
            plt.subplot(len(genes), 2, 2*n+2)
            plt.plot(h)
            plt.ylim(0, h.max())
            plt.xlim(0, len(h))
        plt.draw()
        plt.savefig("figures/genes_"+species_names[i]+".png")
        plt.gcf().clear()

    print("Storing final generation...")
    filenames = []
    samples = []
    group_by_species = {}
    for i in range(len(GA.originals)):
        group_by_species[i] = [org for org in GA.curr_generation if org.species==i]
    for i in range(len(GA.originals)):
        peers = group_by_species[i]
        for j in range(len(peers)):
            filenames.append("GA."+species_names[i][:-4]+"_"+str(j)+".wav")
            samples.append(peers[j].waveform)

    for gen, output in zip(samples, filenames):
        out = io.BytesIO()
        audio.save_wav(gen, out)

        with open("output/"+output, "wb") as f:
            f.write(out.getvalue())

    print("Program complete.")
