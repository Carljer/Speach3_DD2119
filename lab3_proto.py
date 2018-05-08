import numpy as np
from lab3_tools import *
from proto import mfcc
from prondict import prondict
from proto2 import concatHMMs, viterbi
from sklearn.mixture import log_multivariate_normal_density
import matplotlib.pyplot as plt
from tools2 import log_multivariate_normal_density_diag



np.set_printoptions(threshold=np.nan)

def words2phones(wordList, pronDict, addSilence=True, addShortPause=False):
    phonelist=['sil']
    for word in wordList:
        for element in prondict[word]:
            phonelist.append(element)
    phonelist.append('sil')
    return phonelist
    """ word2phones: converts word level to phone level transcription adding silence

    Args:
       wordList: list of word symbols
       pronDict: pronunciation dictionary. The keys correspond to words in wordList
       addSilence: if True, add initial and final silence
       addShortPause: if True, add short pause model "sp" at end of each word
    Output:
       list of phone symbols
    """

def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
    utteranceHMM=concatHMMs(phoneHMMs,phoneTrans)
    example=np.load('lab3_example.npz')['example'].item()
    emmision=log_multivariate_normal_density_diag(lmfcc,utteranceHMM['means'],utteranceHMM['covars'])
    #print(emmision[19:22,:])
    #print(emmision[])
    u=example['utteranceHMM']

    vitpath,vitmax=viterbi(example['obsloglik'],np.log(u['startprob']),np.log(u['transmat']))
    #print("example ",vitpath)
    vitpath,vitmax=viterbi(emmision,np.log(utteranceHMM['startprob']),np.log(utteranceHMM['transmat']))

    #print(vitpath,vitmax)
    #print(emmision[19:22,:])
    return vitpath,vitmax
    """ forcedAlignmen: aligns a phonetic transcription at the state level

    Args:
       lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
              computed the same way as for the training of phoneHMMs
       phoneHMMs: set of phonetic Gaussian HMM models
       phoneTrans: list of phonetic symbols to be aligned including initial and
                   final silence

    Returns:
       list of strings in the form phoneme_index specifying, for each time step
       the state from phoneHMMs corresponding to the viterbi path.
    """

def hmmLoop(hmmmodels, namelist=None):
    """ Combines HMM models in a loop

    Args:
       hmmmodels: list of dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to combine, if None,
                 all the models in hmmmodels are used

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models
       stateMap: map between states in combinedhmm and states in the
                 input models.

    Examples:
       phoneLoop = hmmLoop(phoneHMMs)
       wordLoop = hmmLoop(wordHMMs, ['o', 'z', '1', '2', '3'])
    """


def main():
    phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()
    phones = sorted(phoneHMMs.keys())
    nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
    stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]
    filename = 'tidigits/disc_4.1.1/tidigits/train/man/nw/z43a.wav'
    samples, samplingrate = loadAudio(filename)
    lmfcc = mfcc(samples)
    wordTrans = list(path2info(filename)[2])
    phoneTrans=words2phones(wordTrans,prondict)

start=0
if start==1:
    main()
## Get states
example=np.load('lab3_example.npz')['example'].item()
phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()
phones = sorted(phoneHMMs.keys())
nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]

##Get lmfcc
filename = 'tidigits/disc_4.1.1/tidigits/train/man/nw/z43a.wav'
samples, samplingrate = loadAudio(filename)
lmfcc = mfcc(samples)

## World level transcription
wordTrans = list(path2info(filename)[2])
phoneTrans=words2phones(wordTrans,prondict)
#print(plist)
phoneTrans=phoneTrans[1:-1]
##Concatenate HMMS
utteranceHMM=concatHMMs(phoneHMMs,phoneTrans)

##Statetrans
stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans
                  for stateid in range(nstates[phone])]







vit,v=forcedAlignment(lmfcc,phoneHMMs,phoneTrans)
viterbiStateTrans=[]
for i in vit[1]:
    viterbiStateTrans.append(stateTrans[i])
#seq=frames2trans(viterbiStateTrans, outfilename='z43a.lab')
