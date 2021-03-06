"""Compute mean and std for feature normalizer, and save to file."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import argparse
import functools

sys.path.insert(0, os.path.dirname(__file__) + "/../")
from data_utils.normalizer import FeatureNormalizer
from data_utils.audio_featurizer import AudioFeaturizer
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('num_samples',      int,    2000,    "# of samples to for statistics.")
add_arg('specgram_type',    str,
        'linear',
        "Audio feature type. Options: linear, mfcc.",
        choices=['linear', 'mfcc'])
add_arg('manifest_path',    str,
        'data/aishell/manifest.train',
        "Filepath of manifest to compute normalizer's mean and stddev.")
add_arg('output_path',    str,
        'data/aishell/mean_std.npz',
        "Filepath of write mean and stddev to (.npz).")
# yapf: disable
args = parser.parse_args()


def main():
    print_arguments(args)

    audio_featurizer = AudioFeaturizer(specgram_type=args.specgram_type)

    def augment_and_featurize(audio_segment):
        return audio_featurizer.featurize(audio_segment)

    normalizer = FeatureNormalizer(
        mean_std_filepath=None,
        manifest_path=args.manifest_path,
        featurize_func=augment_and_featurize,
        num_samples=args.num_samples)
    normalizer.write_to_file(args.output_path)


if __name__ == '__main__':
    main()
