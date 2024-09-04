#HELPER FUNCTIONS
#================

#imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pyflac
import scipy.io.wavfile as wav
import scipy.signal as sig
from sklearn.preprocessing import StandardScaler
import mne
import numpy as np

""" helper functions for testing_script.py. Used for data modification/
preprocessing before use.

"""

def load_eeg(filename: str, sample_rate: int) -> mne.io.Raw:
	raw = mne.io.read_raw_brainvision(filename, preload=True)
	events, event_dict = mne.events_from_annotations(raw)
	#start of songs in sample numbers
	song_starts = np.array(events)[events[:,2] == 10001][2:,0]
	press_starts = []
	press_starts = events[2:,0]

	return raw

def load_audio(filename: str, sample_rate: int, number_samples: int) -> np.ndarray:
	
	scaler = StandardScaler()

	audio, aud_samp_rate = pyflac.FileDecoder(filename, "temp.wav").process()
	#aud_rate, audio = wav.read("temp.wav")
	dtype = audio.dtype

	#Resample audio to EEG sample rate and get audio envelope
	return sig.resample(audio, number_samples)


def split_events(X, Y, events, sample_rate, bound):

	new_X = []
	new_Y = []

	for event in events:
		new_X.append(X[:,event - (sample_rate*bound):event + (sample_rate*bound)])
		new_Y.append(Y[:,event - (sample_rate*bound):event + (sample_rate*bound)])

	return new_X, new_Y

def train_test_val_split(eeg, audio, train_size, test_size, samp_rate, rand, mode=None, num_segments=1000, seconds=10):

	scaler = StandardScaler()

	split_eeg = eeg

	#split data into num_segments segments
	times = np.linspace(seconds*samp_rate, eeg.shape[1], num_segments, dtype = int)
	#each segment has a bound of "seconds" seconds before and after the segment
	split_audio, split_eeg = split_events(audio, split_eeg, times, samp_rate, seconds)

	labels = [f'song{i}' for i in range(1, len(split_audio))]
	labels_train, labels_test = train_test_split(labels, train_size=train_size, test_size=test_size, random_state=rand)
	labels_test, labels_val = train_test_split(labels_test, train_size=0.5, test_size=0.5, random_state=rand)

	#X and Y are dictionaries so that the ordering of the corresponding segments can
	#be maintained
	X = {}
	Y = {}
	for i in range(1,len(split_audio)):
		X[f'song{i}'] = split_audio[i]

		Y[f'song{i}'] = split_eeg[i]

	size = Y['song3'].T.shape

	X_test, X_train, X_val, Y_test, Y_train, Y_val = [],[],[],[],[],[]

	for i in labels_train:
		if(Y[i].T.shape == size):
			if len(X[i].shape) == 1:
				Y_train.append(X[i])
			else:
				Y_train.append(X[i])

			X_train.append(Y[i].T)

	for i in labels_val:
		if(Y[i].T.shape == size):
			if len(X[i].shape) == 1:
				Y_val.append(X[i])
			else:
				Y_val.append(X[i])

			X_val.append(Y[i].T)

	for i in labels_test:
		if(Y[i].T.shape == size):
			if len(X[i].shape) == 1:
				Y_test.append(X[i])
			else:
				Y_test.append(X[i])
				
			X_test.append(Y[i].T)

	X_train = np.array(X_train)
	Y_train = np.array(Y_train)
	X_test = np.array(X_test)
	Y_test = np.array(Y_test)
	X_val = np.array(X_val)
	Y_val = np.array(Y_val)

	return X_train, Y_train, X_test, Y_test, X_val, Y_val



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


#split input data into epochs
#give to function as 2d array at the minimum where each column is a sample point
#benchmarks should be a list of sample points
def split(data, benchmarks):
	n_samples = data.shape[-1]
	songs = []
	prev = 0
	for song in benchmarks:
		songs.append(data[:,prev:song])
		prev = song

	return songs
