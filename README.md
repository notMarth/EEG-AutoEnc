# TRF and EEG Autoencoder Project
Clone this repo to your machine and have fun :)
The models directory has the autoencoder and the TRF models (WIP models included) while the TRFNotebooks have Jupyter notebooks that were used for testing the TRF in the early stages of the project (here for archival/historical purposes).
The figs directory is where figures should be generated

# How to Use:
## Required Libraries
I recommend using the provided conda environment file (env.yml), but if you would like to install the libraries yourself, the ones needed are as follows:
  - Numpy
  - Scipy
  - matplotlib
  - Scikit-learn
  - Tensorflow
  - mne
  - pyflac
The following are optional (required for running non-necessary portions of the repo):
  - mtrf

## Data
Note that the EEG and audio data are not included in the repo as they are too large (like 20-30 gigs). Add them loose/individually to a folder called "Data" in the project directory. Please do not try to commit them, they are very large and git will get very angry with you. By default the subject used is subject 28, but you can change this in the testing script
## Run and Compare Models
To run models, run "python testing_script.py MODEL_NAME [MODEL_NAME...] NUM_EPOCHS"
