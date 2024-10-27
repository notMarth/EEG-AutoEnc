#imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pyflac
import scipy.io.wavfile as wav
import scipy.signal as sig
from sklearn.preprocessing import StandardScaler
import mne
import numpy as np

"""helper functions for testing_script.py and models. Used for data modification/
preprocessing before use in models"""

def load_eeg(filename: str) -> mne.io.Raw:
    '''Load raw eeg data from file and grab event dictionary. Return data, events,
    and event labelling as a tuple'''
    
    raw = mne.io.read_raw_brainvision(filename, preload=True)
    events, event_dict = mne.events_from_annotations(raw)

    return (raw, events, event_dict)

def load_audio(filename: str) -> np.ndarray:
    '''Load audio data in. Takes filename for audio file as argument. Return
    audio and its original sample rate'''

    audio, aud_samp_rate = pyflac.FileDecoder(filename, "temp.wav").process()

    return (audio, aud_samp_rate)

def split_events(X, events, sample_rate, bound=(10, 10)) -> np.ndarray:
    '''Split data based on sample-points. Bound should be a tuple of the bound
    around each event (eg. -100ms before and +250ms after each event should be
    given to the function as (0.1, 0.25)).'''
    new_X = []
    
    #events is a list of sample points
    for event in events:
        new_X.append(X[:,event - (int(sample_rate*bound[0])):event + (int(sample_rate*bound[1]))])

    return np.array(new_X)

def train_test_val_split(X: np.ndarray, Y: np.ndarray, train_size, test_size, rand) -> tuple:
    '''Takes X and Y data and splits them. Both sets are split identically (that is,
      element 1 of the split X set will correspond to element 1 of the Y set). Assumes
      all data are the same size. Data is expected to be in the shape 
      (n_samples, n_channels, n_sample_points) for eeg and (n_samples, n_audio_channels, n_sample_points)
      for audio. Both X and Y have the same number of samples. Outputs 6-tuples'''

    labels = list(range(len(X)))
    labels_train, labels_test = train_test_split(labels, train_size=train_size, test_size=test_size, random_state=rand)
    labels_test, labels_val = train_test_split(labels_test, train_size=0.5, test_size=0.5, random_state=rand)
        
    return X[labels_train], Y[labels_train], X[labels_test], Y[labels_test], X[labels_val], Y[labels_val]



#generate mask for empty portions of data at end of songs
#stimulus should be given as 1-D
#threshold gives the maximum amplitude of the audio envelope to be considered as
#"no audio"
#minimum gives the number of sample points to be under this threshold for the
#current section of the song to be considered the end
def mask(stimulus, threshold, minimum):
    n_samples = len(stimulus)
    song_mask = np.ones(n_samples)
    min_num = minimum
    thresh = threshold
    zeros = 0
    num=0
    
    for sample in range(n_samples):
        if np.abs(stimulus[sample]) <= thresh:
            zeros += 1
            if zeros == min_num:
                song_mask[sample-min_num-1:n_samples] = np.zeros(n_samples - (sample - min_num-1))
                zeros=0
                num+=1
                break
        else:
            zeros = 0

    return song_mask