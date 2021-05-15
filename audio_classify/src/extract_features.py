import librosa
from mel_features import log_mel_spectrogram
import numpy as np

class ExtractFeatures():
  def generate_mfcc(self, config, filtered_signal):
    mfcc = librosa.feature.mfcc(filtered_signal, 
                                sr = config["pre_process"]["sample_rate"], 
                                n_mfcc=config["pre_process"]["num_mfcc"], 
                                n_fft=config["pre_process"]["n_fft"], 
                                hop_length=config["pre_process"]["hop_length"])
    return mfcc

  def generate_rms(self, config, filtered_signal):
    rms = librosa.feature.rms(y=filtered_signal, 
                              frame_length= config["pre_process"]["n_fft"], 
                              hop_length= config["pre_process"]["hop_length"], 
                              center=True)
    return rms

  def generate_spectral_centroid(self, config, filtered_signal):
    centroid = librosa.feature.spectral_centroid(filtered_signal+0.01, 
                                                  sr = config["pre_process"]["sample_rate"])
    return centroid

  def generate_zero_crossing_rate(self, config, filtered_signal):
    zcr = librosa.feature.zero_crossing_rate(y=filtered_signal + 0.01)
    return zcr


  def geneate_onset_strength(self, config, filtered_signal):
    onset = librosa.onset.onset_strength(y=filtered_signal, 
                                         sr= config["pre_process"]["sample_rate"])
    
    onset = onset[np.newaxis, ...]
    return onset

  def generate_mel_spectogram(self, config, filtered_signal):
    '''
    mel = librosa.feature.melspectrogram(y=filtered_signal, 
                                          sr = config["pre_process"]["sample_rate"], 
                                          n_mels = config["pre_process"]["n_mels"], 
                                          fmax=10000 , 
                                          n_fft = config["pre_process"]["n_fft"], 
                                          hop_length= config["pre_process"]["hop_length"])
    '''
    mel = log_mel_spectrogram(filtered_signal, audio_sample_rate=config["pre_process"]["sample_rate"], log_offset=0.01)
    return mel

  def generate_rec_matrix(self, config, filtered_signal):
    chroma = librosa.feature.chroma_cqt(y=filtered_signal, 
                                        sr=config["pre_process"]["sample_rate"], 
                                        hop_length=config["pre_process"]["hop_length"])
    chroma_stack = librosa.feature.stack_memory(chroma, n_steps=10, delay=3)
    rec_matrix = librosa.segment.recurrence_matrix(chroma_stack, metric="cosine",  mode='affinity')
    return rec_matrix

  def geneate_tempogram(self, config, filtered_signal):
    oenv = librosa.onset.onset_strength(y=filtered_signal,
                                        sr= config["pre_process"]["sample_rate"],
                                        hop_length= config["pre_process"]["hop_length"])
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, 
                                          sr= config["pre_process"]["sample_rate"],
                                          hop_length= config["pre_process"]["hop_length"])
    return tempogram
