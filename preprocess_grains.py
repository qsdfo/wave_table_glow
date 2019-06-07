import argparse
import json
import os

import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

import osc_gen_modif.dsp as dsp
from osc_gen_modif.dsp import slice_cycles
from osc_gen_modif.sig import SigGen

"""
Long grain introduces click at the change of grain.
- use olap ?


"""


class PreprocessorGrains:
    def __init__(self, sampling_rate, window_size, output_dir):
        self.sampling_rate = sampling_rate

        # FFT parameters
        assert ((window_size & (window_size - 1)) == 0), "window_size is not a power of 2"
        self.window_size = window_size  #  Used in fft when training the model later
        self.hop_length = int(window_size // 4)

        #
        self.output_dir = output_dir

        # #  Allowed f0 for grains
        # #  Determined so that the window_size is a multiple of a complete cycle's length
        # self.allowed_f0s = []
        # allowed_f0 = 1
        # while allowed_f0 <= self.window_size:
        #     self.allowed_f0s.append(allowed_f0)
        #     allowed_f0 *= 2

        return

    def preprocess_file(self, filepath):
        print(f'# {filepath}')
        filename = os.path.splitext(os.path.basename(filepath))[0]
        ############################################################
        #  Read file
        y, _ = librosa.core.load(filepath, sr=self.sampling_rate, mono=True)
        #  Remove leading and trailing silences
        y, _ = librosa.effects.trim(y)

        #  Just a test
        write(f'dump/{filename}.wav', self.sampling_rate, y)

        ###########################################################
        # Note: tried osc_gen
        #  but it's a bit too coarse for using.
        #  for instance a unique frequency is used
        # sig_gen = SigGen(num_points=self.window_size)
        # cycles = slice_cycles(y, 50, self.sampling_rate)
        # waves = [sig_gen.arb(c) for c in cycles]

        ###########################################################
        # Detect successive zeros corresponding to cycling points
        #  Instantaneous frequencies
        inst_freq, _ = librosa.core.piptrack(y, sr=self.sampling_rate,
                                             n_fft=self.window_size,
                                             hop_length=self.hop_length)

        #  start_frame is the next zero
        zero_crossings_left = np.where(np.abs(np.diff(np.sign(y))) > 0)[0]

        # Browse waveform, extracting segment of size window_length
        start_frame = 0
        counter = 0
        while start_frame < len(y):
            #  We want after the zero crossing
            index_zero_crossing = np.min(np.where(zero_crossings_left >= start_frame))
            start_frame = zero_crossings_left[index_zero_crossing] + 1

            # Get instantaneous f0
            segment_index = int(start_frame / self.hop_length)
            detected_inst_freq = np.where(inst_freq[:, segment_index] > 0)
            non_zeros_freq = inst_freq[detected_inst_freq, segment_index]
            f0 = np.min(non_zeros_freq)

            ################################################
            # How to compute end_frame ?
            #  Long grains, finding zeros
            # end_frame = self.find_next_stiching_frame(start_frame, self.window_size, zero_crossings_left, f0)

            #  Short grains (one cycle), on zeros ?
            frames_per_cycle = int(self.sampling_rate // f0)
            end_frame = self.find_next_stiching_frame(start_frame, frames_per_cycle, zero_crossings_left, f0)

            #  Short grains, bourrin
            #  In fact, still need to use frames_per_cycles.
            #  If we use an arbitrary value (like 441 ~ 100Hz @ 44100), artifacts are introduced
            # end_frame = start_frame + frames_per_cycle + 1
            ################################################

            # Extract segment
            segment = y[start_frame:end_frame]

            # Interpolate segment
            def arb(data, num_points):
                """ Generate an arbitrary wave cycle. The provided data will be
                interpolated, if possible, to occupy the correct number of samples for
                a single cycle at our reference frequency and then normalized and
                scaled as appropriate.

                @param data seq : A sequence of samples representing a single cycle
                    of a wave
                """

                interp_y = data
                num = interp_y.size
                interp_x = np.linspace(0, num, num=num)
                interp_xx = np.linspace(0, num, num=num_points)
                interp_yy = np.interp(interp_xx, interp_x, interp_y)
                return interp_yy

            interpolated_segment = arb(segment, self.window_size)

            # Normalize waveform
            norm_interp_seg = 0.999969 * librosa.util.normalize(interpolated_segment, axis=0)
            norm_segment = 0.999969 * librosa.util.normalize(segment, axis=0)

            #  Generate a waveform from the grain at different pitches
            #  Here test if it works to use non perfectly cut waveforms
            wave_to_write = np.concatenate([norm_interp_seg for _ in range(1000)])
            write(f'dump/{filename}_{counter}_interp.wav', self.sampling_rate, wave_to_write)
            #  If we repitch shift the low frequency grain, what happens ?
            shifted_norm_interp_seg = librosa.effects.pitch_shift(wave_to_write, sr=self.sampling_rate, n_steps=42)
            write(f'dump/{filename}_{counter}_pitchShifted.wav', self.sampling_rate, shifted_norm_interp_seg)
            # short grains
            wave_to_write = np.concatenate([norm_segment for _ in range(1000)])
            write(f'dump/{filename}_{counter}.wav', self.sampling_rate, wave_to_write)

            #  Compute next starting frame
            start_frame = self.find_next_stiching_frame(start_frame, self.hop_length, zero_crossings_left, f0)
            counter += 1

        return

    def find_next_stiching_frame(self, start_frame, target_length, zero_crossings_left, f0):
        # Find next stiching point (possibly not at self.window_size frames from start point)
        # which is close to a multiple of Tn0 = fs / f0
        frames_per_cycle = self.sampling_rate // f0
        num_cycles = target_length // frames_per_cycle
        end_frame_approx = start_frame + num_cycles * frames_per_cycle
        #  Take next zero_crossing frame
        end_frame = zero_crossings_left[np.where(end_frame_approx <= zero_crossings_left)[0][0]] + 1
        return end_frame

    def process_database(self, filename):
        with open(filename, encoding='utf-8') as f:
            files = f.readlines()

        for file in files:
            file = file.rstrip()
            self.preprocess_file(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str,
                        help='txt list containing the files to process')
    parser.add_argument('-c', '--config', type=str,
                        help='JSON config')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='path to output directory')
    args = parser.parse_args()

    # FFT parameters
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    data_config = config["data_config"]
    window_size = data_config["segment_length"]
    sampling_rate = data_config["sampling_rate"]

    preprocessor = PreprocessorGrains(sampling_rate=sampling_rate, window_size=window_size, output_dir=args.output_dir)
    preprocessor.process_database(args.filename)
