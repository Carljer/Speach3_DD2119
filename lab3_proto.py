import numpy as np
from lab3_tools import *
from proto import mfcc
import os
from prondict import prondict
from proto2 import concatHMMs, viterbi
from sklearn.mixture import log_multivariate_normal_density
import matplotlib.pyplot as plt
from tools2 import log_multivariate_normal_density_diag
import random
from sklearn.preprocessing import StandardScaler


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





def getfeatures():
    traindata = []
    for root, dirs, files in os.walk('tidigits/disc_4.1.1/tidigits/train'):
        for file in files:
            if file.endswith('.wav'):
                filename = os.path.join(root, file)
                print(filename)
                samples, samplingrate = loadAudio(filename)
                lmfcc,mspec = mfcc(samples)
                ## World level transcription
                wordTrans = list(path2info(filename)[2])
                phoneTrans1=words2phones(wordTrans,prondict)
                phoneTrans=phoneTrans1[1:-1]
                ##Concatenate HMMS
                utteranceHMM=concatHMMs(phoneHMMs,phoneTrans1)
                ##Statetrans
                stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans1
                                  for stateid in range(nstates[phone])]

                vit,v=forcedAlignment(lmfcc,phoneHMMs,phoneTrans)
                viterbiStateTrans=[]
                for i in vit[1]:
                    viterbiStateTrans.append(stateTrans[i])
                targets=[]
                for r in viterbiStateTrans:
                    targets.append(stateList.index(r))
                    #print(type(stateList.index(r)))

                #...your code for feature extraction and forced alignment
                traindata.append({'filename': filename, 'lmfcc': lmfcc,
                                  'mspec': mspec, 'targets': targets})


    np.savez('traindata.npz', traindata=traindata)


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


def sortmanwoman(alldata):
    dman={}
    dwoman={}
    nr_data=alldata.shape[0]
    for i in range(nr_data):
        filename=alldata[i]['filename']
        if 'woman' in filename:
            speaker=filename[41:43]+'_w'
            if speaker in dwoman:
                dwoman[speaker].append(alldata[i])
            else:
                dwoman[speaker]=[alldata[i]]
        else:
            speaker=filename[39:41]+'_m'
            if speaker in dman:
                dman[speaker].append(alldata[i])
            else:
                dman[speaker]=[alldata[i]]
    return dman,dwoman



def sortmanwoman2(alldata):
    dman={}
    dwoman={}
    nr_data=alldata.shape[0]
    for i in range(nr_data):
        filename=alldata[i]['filename']
        if 'woman' in filename:
            speaker=filename[40:42]+'_w'
            if speaker in dwoman:
                dwoman[speaker].append(alldata[i])
            else:
                dwoman[speaker]=[alldata[i]]
        else:
            speaker=filename[38:40]+'_m'
            if speaker in dman:
                dman[speaker].append(alldata[i])
            else:
                dman[speaker]=[alldata[i]]
    return dman,dwoman




def reg(alldata):
    nr_data=alldata.shape[0]
    normdata=alldata
    for i in range(nr_data):
        utterance=alldata[i]
        scaler=StandardScaler()
        scaler.fit(utterance['lmfcc'])
        normdata[i]['lmfcc']=scaler.transform(utterance['lmfcc'])
        scaler2=StandardScaler()
        scaler2.fit(utterance['mspec'])
        normdata[i]['mspec']=scaler2.transform(utterance['mspec'])
    return normdata


def gettraintest(data):
    lmfcc_x=np.empty([0,13])
    mspec_x=np.empty([0,40])

    targets_y=np.empty([0,1])
    for keys in data:
        print(keys)
        for utterance in data[keys]:
            lmfcc=utterance['lmfcc']
            mspec=utterance['mspec']
            target=np.array(utterance['targets'])
            target=np.reshape(target,[target.shape[0],1])
            lmfcc_x=np.vstack((lmfcc_x,lmfcc))
            mspec_x=np.vstack((mspec_x,mspec))
            targets_y=np.vstack((targets_y,target))
    return lmfcc_x,mspec_x,targets_y


def gettraintest2(data):
    mspec_x=np.empty([0,440])

    targets_y=np.empty([0,1])
    for keys in data:
        print(keys)
        for utterance in data[keys]:
            mspec=utterance['mspec']
            n,m=mspec.shape
            print(utterance['filename'])
            for i in range(n):
                if i<5:

                    mspecbig=np.vstack((mspec[::-1],mspec[1:]))
                    n,m=mspec.shape
                    mrow=mspecbig[i+n-5:i+n+6].reshape([1,440])
                    mspec_x=np.vstack((mspec_x,mrow))
                elif i>n-6:
                    mspecbig=np.vstack((mspec[0:-1],mspec[::-1]))
                    mrow=mspecbig[i-5:i+6].reshape([1,440])
                    mspec_x=np.vstack((mspec_x,mrow))
                else:
                    #print(i)
                    m=mspec[i-5:i+6,:].reshape([1,440])
                    mspec_x=np.vstack((mspec_x,m))



                #print(mspec_x.shape)
            target=np.array(utterance['targets'])
            target=np.reshape(target,[target.shape[0],1])
            #mspec_x=np.vstack((mspec_x,mspec))
            targets_y=np.vstack((targets_y,target))
    return mspec_x,targets_y



def createtestset(alldata, flag = False):
    traindict={}
    testdict={}
    l1=[]
    l2=[]

    if flag:
        d1,d2=sortmanwoman(alldata)
    else:
        d1,d2=sortmanwoman2(alldata)

    for keys in d1:
        l2.append(keys)
    for keys in d2:
        l1.append(keys)

    for key in l2[0:len(d1)//10 +1]:
        testdict[key] = d1[key]
    for key in l1[0:len(d1)//10 +1]:
        testdict[key] = d2[key]

    for key in l2[len(d1)//10 +1:]:
        traindict[key] = d1[key]
    for key in l1[len(d1)//10 +1:]:
        traindict[key] = d2[key]
    return traindict,testdict





## Get states
# example=np.load('lab3_example.npz')['example'].item()
# phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()
# phones = sorted(phoneHMMs.keys())
# nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
# stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]

#getfeatures()
alldata=reg(np.load('traindata.npz')['traindata'])
# alldata_2=reg(np.load('testdata.npz')['traindata'])
train,val=createtestset(alldata, True)
# test1,test2=createtestset(alldata_2, False)
lmfcc_val_x,mspec_val_x,val_y = gettraintest2(val)
# lmfcc_train_x,mspec_train_x,train_y = gettraintest(train)
# lmfcc_test,mspec_test_x,test_y = gettraintest(test1.update(test2))
#alldata=random.shuffle(alldata)









#
# ##Get lmfcc
# filename = 'tidigits/disc_4.1.1/tidigits/train/man/nw/z43a.wav'
# samples, samplingrate = loadAudio(filename)
# lmfcc,bla = mfcc(samples)
#
# ## World level transcription
# wordTrans = list(path2info(filename)[2])
# phoneTrans1=words2phones(wordTrans,prondict)
# #print(plist)
# phoneTrans=phoneTrans1[1:-1]
# ##Concatenate HMMS
# utteranceHMM=concatHMMs(phoneHMMs,phoneTrans1)
#
#
#
#
# ##Statetrans
# stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans1
#                   for stateid in range(nstates[phone])]
#
#
#
#
#
#
#
# vit,v=forcedAlignment(lmfcc,phoneHMMs,phoneTrans)
# viterbiStateTrans=[]
# for i in vit[1]:
#     viterbiStateTrans.append(stateTrans[i])
# seq=frames2trans(viterbiStateTrans, outfilename='z43a.lab')
