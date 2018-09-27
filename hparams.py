import tensorflow as tf

### Utility code adapted from https://github.com/keithito/tacotron

# Default hyperparameters:
hparams = tf.contrib.training.HParams(
    byte_norm=32767,
    num_freq=2048+1,
    sample_rate=44100,
    frame_length_ms=50,
    frame_shift_ms=12.5,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,

    griffin_lim_iters=50,

    num_generations=8,
    num_genes=4,
    gen_size=16,
    crossover_rate=0.1,
    expand_gene_rate=0.25,
    mutate_amplitude=0.1,
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
