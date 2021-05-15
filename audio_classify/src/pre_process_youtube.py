import json
import math
import os
from json import JSONEncoder
from math import floor
from multiprocessing.dummy import Pool as ThreadPool
from os import path

import librosa
import numpy as np

from extract_features import ExtractFeatures


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class PreProcessYoutube():

  def __init__(self, config):
    self.config = config
    self.samples_per_segment = self.config["pre_process"]["sample_rate"] * self.config["pre_process"]["segment_duration"]
    self.data = {
        "label": ["music", "not_music"],
        "data": []
    }

  def run(self):
    """[run the pre-process algorithm over the entire sample and save it]
    """
    youtube_files = self.get_list_of_files()
    file_name = self.open_class_file()
    class_name = self.open_class_youtube()
    file_name = self.join_file_with_types(youtube_files, file_name)
    file_name = self.join_class_with_file(file_name, class_name)

    pool = ThreadPool(self.config["pre_process"]["number_of_threads"])
    pool.map(self.generate_features, file_name,)
    pool.close() 
    pool.join()
    
    with open(self.config["youtube"]["output"], "w") as fp:
        json.dump(self.data, fp, cls=NumpyArrayEncoder)


  def get_list_of_files(self):
    """[create a list with the files download from the audioset youtube. ]

      Returns:
          [dictionary]: [two keys one for speech and other for music]
    """    
    youtube_files = [path.join(root,f) for root,dirs,files in os.walk(self.config["youtube"]["path"]+"audio/") for f in files]
    youtube_files = list(filter(lambda x: '.wav' in x, youtube_files))
    return youtube_files

  def open_class_file(self):
    file_name_dict = {}
    path = self.config["youtube"]["path"]
    with open(path+'balanced_train_segments.csv', 'r') as f:
      file_names = f.readlines()
      file_names = file_names[3:]
    for fname in file_names:
      fname = fname.rstrip().replace('\"','').split(',')
      file_name_dict[fname[0]] = fname
    return file_name_dict

  def open_class_youtube(self):
    class_name_dict = {}
    path = self.config["youtube"]["path"]
    with open(path+'class_labels_indices.csv', 'r') as f:
      class_names = f.readlines()
      class_names = class_names[1:]
    for cname in class_names:
      cname = cname.rstrip().split(',')[1:]
      class_name_dict[cname[0]] = cname
    return class_name_dict

  def join_file_with_types(self, youtube_files, file_name):
    for yfile in youtube_files:
      sidx  = yfile.rindex("/") + 1
      fidx  = yfile.rindex("_")
      fname = yfile[sidx:fidx]
      if fname in file_name:
        file_name[fname].append(yfile)
    return file_name

  def join_class_with_file(self, file_name, class_name):
    new_list = []
    for key in file_name.keys():
      file = {
        "name": key,
        "path": file_name[key][-1],
        "class": ""
      }
      for ckey in class_name.keys():
        if ckey in file_name[key]:
          if file["class"] == "":
            file["class"] = class_name[ckey][2]
          elif file["class"] == "Music":
            file["class"] = class_name[ckey][2]
      new_list.append(file)
    return new_list

  def generate_features(self,file_dict):
      """[convert audio file using mfcc feature extration]
      Args:
          file_path ([string]): [path of the sample file]
          label ([int]): [0 if is music and 1 if is speech]
      """
      label = 1
      if file_dict["class"] == "Music":
        label = 0
      if path.exists(file_dict["path"]):
        print("File: {}, Type:{}".format(file_dict["path"], file_dict["class"]))
        signal, sample_rate = librosa.load(file_dict["path"], sr=self.config["pre_process"]["sample_rate"])
        num_segments = floor(len(signal)/self.samples_per_segment)
        extractFeatures = ExtractFeatures()
        #num_mfcc_vectors_per_segment = math.ceil(self.samples_per_segment / self.config["pre_process"]["hop_length"])
        
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
      else:
        print("Erro: {}, Type:{}".format(file_dict["path"], file_dict["class"]))
