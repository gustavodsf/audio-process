import json
import math
import os
from json import JSONEncoder
from math import floor
from multiprocessing.dummy import Pool as ThreadPool
from os.path import join

import librosa
import numpy as np

from extract_features import ExtractFeatures


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class PreProcess():

    def __init__(self, config):
        """[Constructor get config json so that is possible to run the pre-process]

        Args:
            config ([dict]): [parameter used to adjust the preprocess]
        """
        self.config = config
        self.samples_per_segment = self.config["pre_process"]["sample_rate"] * self.config["pre_process"]["segment_duration"]
        self.data = {
            "label": ["music", "not_music"],
            "data": []
        }


    def run(self):
        """[run the pre-process algorithm over the entire sample and save it, we use a pool of threads to run it]
        """        
        all_files = self.get_list_of_files()
        pool = ThreadPool(self.config["pre_process"]["number_of_threads"])

        pool.map(self.features_for_music, all_files['music'])
        pool.map(self.features_for_speech,  all_files['speech'])
        pool.close() 
        pool.join()

        with open(self.config["pre_process"]["output"], "w") as fp:
            json.dump(self.data, fp, cls=NumpyArrayEncoder)

    def get_list_of_files(self):
        """[create two lists one for music other for speech,  with all files inside de sample directory.]

        Returns:
            [dictionary]: [two keys one for speech and other for music]
        """        
        music_files = [join(root,f) for root,dirs,files in os.walk(self.config["pre_process"]["music_path"]) for f in files]
        music_files = list(filter(lambda x: '.wav' in x, music_files))
        
        speech_files = [join(root,f) for root,dirs,files in os.walk(self.config["pre_process"]["speech_path"]) for f in files]
        speech_files = list(filter(lambda x: '.wav' in x, speech_files))
        return {
            "music": music_files,
            "speech": speech_files
        }


    def generate_features(self,file_path, label):
        """[convert audio file using mfcc feature extration]
        Args:
            file_path ([string]): [path of the sample file]
            label ([int]): [0 if is music and 1 if is speech]
        """

        signal, sample_rate = librosa.load(file_path, sr=self.config["pre_process"]["sample_rate"])
        num_segments = floor(len(signal)/self.samples_per_segment)
        extractFeatures = ExtractFeatures()
        num_mfcc_vectors_per_segment = math.ceil(self.samples_per_segment / self.config["pre_process"]["hop_length"])
        
        # process all segments of audio file
        for d in range(num_segments):
            # calculate start and finish sample for current segment
            start = self.samples_per_segment * d
            finish = start + self.samples_per_segment

            #mfcc = extractFeatures.generate_mfcc(self.config,signal[start:finish])
            centroid = extractFeatures.generate_spectral_centroid(self.config,signal[start:finish])
            zcr = extractFeatures.generate_zero_crossing_rate(self.config,signal[start:finish])
            mel_spec = extractFeatures.generate_mel_spectogram(self.config,signal[start:finish])
            onset = extractFeatures.geneate_onset_strength(self.config,signal[start:finish])
            rms = extractFeatures.generate_rms(self.config,signal[start:finish])
            self.data["data"].append({
                # "mfcc": mfcc,
                "centroid": centroid,
                "zcr": zcr,
                "mel_spec": mel_spec,
                "onset": onset,
                "rms": rms,
                "label": label
            })

    def features_for_music(self, file_path):
        """[calculate the mfcc for all files inside the directory, but run only for music]

        Args:
            file_path ([str]): [path that contains all music samples]
        """        
        self.generate_features(file_path, 0)
        print("File: {}, Type:{}".format(file_path, "music"))

    def features_for_speech(self, file_path):
        """[calculate the mfcc for all files inside the directory, but run only for speech]

        Args:
            file_path ([str]): [path that contains all speech samples]
        """        
        self.generate_features(file_path, 1)
        print("File: {}, Type:{}".format(file_path, "not_music"))
