import json
import shutil
import os
import pathlib
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn.functional as F_general
import torchaudio
import torchaudio.functional as F
from PIL import Image
from tqdm import tqdm
import datetime

pd.set_option('future.no_silent_downcasting', True)
random.seed(42)

cm = plt.get_cmap('jet')


class YOLODataset:
        def __init__(self, config):
            # Spectrogram settings
            self.duration = config['duration']
            self.overlap = config['overlap']  # overlap of the chunks in %
            self.desired_fs = config['desired_fs']
            self.channel = config['channel']
            self.log = config['log']
            self.color = config['color']

            # Folders
            self.wavs_folder = pathlib.Path(config['wavs_folder'])
            self.dataset_folder = pathlib.Path(config['dataset_folder'])
            self.images_folder = self.dataset_folder.joinpath('images')
            self.labels_folder = self.dataset_folder.joinpath('labels')

            self.annotations_file = config['annotations_file']

            self.nfft = config['nfft']
            self.win_len = config['win_len']
            self.hop_length = int(self.win_len / config['hop_ratio'])
            self.win_overlap = self.win_len - self.hop_length

            if 'minimum_freq' in config.keys():
                self.F_MIN = config['minimum_freq']
            else:
                self.F_MIN = 0

            self.split_folders = False

            self.blocksize = int(self.duration * self.desired_fs)

            self.config = config

        def __setitem__(self, key, value):
            if key in self.config.keys():
                self.config[key] = value
            self.__dict__[key] = value

        def save_config(self, config_path):
            with open(config_path, 'w') as f:
                json.dump(self.config, f)

        def create_spectrograms(self, overwrite=False, save_image=True, model=None, labels_path=None):
            # First, create all the images
            if self.split_folders:
                folders_list = []
                for f in self.wavs_folder.glob('*'):
                    if f.is_dir():
                        folders_list.append(f)
                        if save_image:
                            if not self.images_folder.joinpath(f.name).exists():
                                os.mkdir(self.images_folder.joinpath(f.name))
                        if model is not None:
                            if not labels_path.joinpath(f.name).exists():
                                os.mkdir(labels_path.joinpath(f.name))
            else:
                folders_list = [self.wavs_folder]

            for folder_n, folder_path in tqdm(enumerate(folders_list), total=len(folders_list)):
                print('Spectrograms from folder %s/%s: %s' % (folder_n, len(folders_list), folder_path))
                for wav_path in tqdm(list(folder_path.glob('*.wav')), total=len(list(folder_path.glob('*.wav')))):
                    waveform_info = torchaudio.info(wav_path)
                    i = 0.0
                    while (i * self.duration + self.duration / 2) < (
                            waveform_info.num_frames / waveform_info.sample_rate):
                        img_name = wav_path.name.replace('.wav', '_%s.png' % i)
                        if self.split_folders:
                            img_path = self.images_folder.joinpath(folder_path.name, img_name)
                        else:
                            img_path = self.images_folder.joinpath(img_name)

                        if overwrite or (not img_path.exists()):
                            start_chunk = int(i * self.blocksize)
                            start_chunk_s = start_chunk / self.desired_fs
                            if waveform_info.sample_rate > self.desired_fs:
                                start_chunk_old_fs = int(start_chunk_s * waveform_info.sample_rate)
                                blocksize_old_fs = int(self.duration * waveform_info.sample_rate)
                                chunk_old_fs, fs = torchaudio.load(wav_path,
                                                                   normalize=True,
                                                                   frame_offset=start_chunk_old_fs,
                                                                   num_frames=blocksize_old_fs)
                                chunk = F.resample(waveform=chunk_old_fs[0, :], orig_freq=fs, new_freq=self.desired_fs)
                            else:
                                chunk, fs = torchaudio.load(wav_path, normalize=True, frame_offset=start_chunk,
                                                            num_frames=self.blocksize)
                                chunk = chunk[0, :]

                            if len(chunk) < self.blocksize:
                                chunk = F_general.pad(chunk, (0, self.blocksize - len(chunk)))

                            img, f = self.create_chunk_spectrogram(chunk)

                            if model is not None:
                                results = model(source=np.ascontiguousarray(np.flipud(img)[:, :, ::-1]),
                                                project=str(self.dataset_folder),
                                                name='predictions',
                                                save=False, show=False, save_conf=True, save_txt=False, conf=0.1,
                                                save_crop=False, agnostic_nms=False, stream=False, verbose=False,
                                                imgsz=640, exist_ok=True)
                                for r in results:
                                    label_name = img_name.replace('.png', '.txt')
                                    if self.split_folders:
                                        r.save_txt(labels_path.joinpath(folder_path.name, label_name), save_conf=True)
                                    else:
                                        r.save_txt(labels_path.joinpath(label_name), save_conf=True)

                            if save_image:
                                if self.log:
                                    fig, ax = plt.subplots()
                                    ax.pcolormesh(img[:, :, ::-1])
                                    ax.set_yscale('symlog')
                                    plt.axis('off')
                                    plt.ylim(bottom=3)
                                    plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
                                else:
                                    Image.fromarray(np.flipud(img)).save(img_path)
                            plt.close()
                        i += self.overlap

        def create_chunk_spectrogram(self, chunk):
            f, t, sxx = scipy.signal.spectrogram(chunk, fs=self.desired_fs, window=('hann'),
                                                 nperseg=self.win_len,
                                                 noverlap=self.win_overlap, nfft=self.nfft,
                                                 detrend=False,
                                                 return_onesided=True, scaling='density', axis=-1,
                                                 mode='magnitude')
            sxx = sxx[f > self.F_MIN, :]
            sxx = 10 * np.log10(sxx)
            per_min = np.percentile(sxx.flatten(), 1)
            per_max = np.percentile(sxx.flatten(), 99)
            sxx = (sxx - per_min) / (per_max - per_min)
            sxx[sxx < 0] = 0
            sxx[sxx > 1] = 1
            sxx = cm(sxx)  # convert to color

            img = np.array(sxx[:, :, :3] * 255, dtype=np.uint8)
            return img, f

        def convert_raven_annotations_to_yolo(self, labels_to_exclude=None, values_to_replace=0):
            """

            :param annotations_file:
            :param labels_to_exclude: list
            :param values_to_replace: should be a dict with the name of the Tag as a key and an int as the value, for the
            yolo classes
            :return:
            """
            f_bandwidth = (self.desired_fs / 2) - self.F_MIN
            for selections_path, selections in self.load_relevant_selection_table(labels_to_exclude=labels_to_exclude):
                print(selections_path)
                selections = selections.loc[selections['ShipClass'] != '0']
                selections.ShipClass = selections.ShipClass.replace({'A': 0, 'B': 0, '?': 0}).astype(int)
                selections.loc[selections['Low Freq (Hz)'] < self.F_MIN, 'Low Freq (Hz)'] = self.F_MIN
                selections['height'] = (selections['High Freq (Hz)'] - selections['Low Freq (Hz)']) / f_bandwidth

                # The y is from the TOP!
                selections['y'] = 1 - (selections['High Freq (Hz)'] / f_bandwidth)

                pbar = tqdm(total=len(selections['Begin File'].unique()))

                for wav_name, wav_selections in selections.groupby('Begin File'):
                    if os.path.isdir(self.wavs_folder):
                        wav_file_path = self.wavs_folder.joinpath(wav_name)
                    else:
                        wav_file_path = self.wavs_folder

                    waveform_info = torchaudio.info(wav_file_path)
                    fs = waveform_info.sample_rate

                    if 'Beg File Samp (samples)' not in wav_selections.columns:
                        wav_selections['Beg File Samp (samples)'] = wav_selections['Begin Time (s)'] * fs
                    if 'End File Samp (samples)' not in wav_selections.columns:
                        wav_selections['End File Samp (samples)'] = wav_selections['End Time (s)'] * fs

                    # Re-compute the samples to match the new sampling rate
                    wav_selections['End File Samp (samples)'] = wav_selections[
                                                                    'End File Samp (samples)'] / fs * self.desired_fs
                    wav_selections['Beg File Samp (samples)'] = wav_selections[
                                                                    'Beg File Samp (samples)'] / fs * self.desired_fs

                    wav_selections.loc[
                        wav_selections['Beg File Samp (samples)'] > wav_selections['End File Samp (samples)'],
                        'End File Samp (samples)'] = waveform_info.num_frames

                    i = 0.0
                    while (i * self.duration + self.duration / 2) < (
                            waveform_info.num_frames / waveform_info.sample_rate):
                        start_sample = int(i * self.blocksize)

                        start_mask = (wav_selections['Beg File Samp (samples)'] >= start_sample) & (wav_selections[
                                                                                                        'Beg File Samp (samples)'] <= (
                                                                                                                start_sample + self.blocksize))
                        end_mask = (wav_selections['End File Samp (samples)'] >= start_sample) & (wav_selections[
                                                                                                      'End File Samp (samples)'] <= (
                                                                                                              start_sample + self.blocksize))
                        chunk_selection = wav_selections.loc[start_mask | end_mask]

                        chunk_selection = chunk_selection.assign(
                            start_x=((chunk_selection['Beg File Samp (samples)'] - i * self.blocksize) / self.blocksize).clip(lower=0, upper=1).values)

                        chunk_selection = chunk_selection.assign(
                            end_x=((chunk_selection['End File Samp (samples)'] - i * self.blocksize) / self.blocksize).clip(lower=0, upper=1).values)

                        chunk_selection = chunk_selection.assign(
                            width=(chunk_selection['end_x'] - chunk_selection['start_x']).values)

                        # Save the chunk detections so that they are with the yolo format
                        # <class > < x > < y > < width > < height >
                        chunk_selection['x'] = (chunk_selection['start_x'] + chunk_selection['width'] / 2)
                        chunk_selection['y'] = (chunk_selection['y'] + chunk_selection['height'] / 2)

                        chunk_selection[[
                            'ShipClass',
                            'x',
                            'y',
                            'width',
                            'height']].to_csv(self.labels_folder.joinpath(wav_name.replace('.wav', '_%s.txt' % i)),
                                              header=None, index=None, sep=' ', mode='w')
                        # Add the station if the image adds it as well!
                        i += self.overlap
                        pbar.update(1)
                pbar.close()

        def convert_raven_annotations_to_df(self, labels_to_exclude=None, values_to_replace=0):
            total_selections = pd.DataFrame()
            f_bandwidth = (self.desired_fs / 2) - self.F_MIN
            for _, selections in self.load_relevant_selection_table(labels_to_exclude=labels_to_exclude):
                selections['height'] = (selections['High Freq (Hz)'] - selections['Low Freq (Hz)']) / f_bandwidth
                # compute the width in pixels
                selections['width'] = ((selections['End Time (s)'] - selections['Begin Time (s)']) / self.duration)

                # The y is from the TOP!
                selections['y'] = 1 - (selections['High Freq (Hz)'] / f_bandwidth) + selections['height'] / 2

                selections['wav'] = np.nan
                selections['wav_name'] = selections['Begin File']
                selections['duration'] = selections.width * self.duration
                selections['min_freq'] = 1 - (selections.y + selections.height / 2)
                selections['max_freq'] = 1 - (selections.y - selections.height / 2)

                if isinstance(values_to_replace, dict):
                    selections = selections.replace(values_to_replace)
                else:
                    selections['Tags'] = 0
                total_selections = pd.concat([total_selections, selections])

            return total_selections

        def load_relevant_selection_table(self, labels_to_exclude=None):
            annotations_file = pathlib.Path(self.annotations_file)
            if annotations_file.is_dir():
                selections_list = list(annotations_file.glob('*.txt'))
            else:
                selections_list = [annotations_file]
            for selection_table_path in selections_list:
                print('Annotations table %s' % selection_table_path.name)
                selections = pd.read_table(selection_table_path)
                if 'Tags' in selections.columns:
                    if labels_to_exclude is not None:
                        selections = selections.loc[~selections.Tags.isin(labels_to_exclude)]

                # Filter the selections
                selections = selections.loc[selections['Low Freq (Hz)'] < (self.desired_fs / 2)]
                selections = selections.loc[selections.View == 'Spectrogram 1']

                selections.loc[
                    selections['High Freq (Hz)'] > (self.desired_fs / 2), 'High Freq (Hz)'] = self.desired_fs / 2

                yield selection_table_path, selections


if __name__ == '__main__':
    config_path = './images_config.json'
    f = open(config_path)
    config = json.load(f)

    ds = YOLODataset(config)
    ds.create_spectrograms(overwrite=False)
    ds.convert_raven_annotations_to_yolo()