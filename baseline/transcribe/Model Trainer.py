"""
To do:
    -Randomized Undersampling each round?
"""


#Library
import os, string, scipy.io.wavfile, re, pickle, python_speech_features
from numpy import zeros, array, reshape
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Bidirectional, TimeDistributed, Activation
from keras.preprocessing import sequence
from keras.engine.topology import Merge
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
from functools import partial, update_wrapper
from math import sqrt, pow
from random import sample, randint
import keras.backend as K
import itertools

#Variables
window_length = 0.06
step = 0.03
batch_size = 30
coefficients = 13
tuple_size = 4
output = "C:\\Users\\Ian\\Academic Stuff\\Projects\\Speech Recognition\Test\\"


##Get list of .wav files and associated .phones files in path
def get_files(path):
    file_list = [file for file in os.listdir(path) if file.endswith(".wav")]
    for a in range(len(file_list)):
        phon_file = file_list[a].replace(".wav", ".phones")
        if os.path.isfile(path+phon_file):
            file_list[a] = (file_list[a], phon_file)
        else:
            file_list[a] = (file_list[a], None)
    ##Returns (wav file, phones file/None)
    return file_list
    
def open_dict(input):
    try:
        with open(input, "rb") as f:
            segdict = pickle.load(f)
            print "Loaded Segment Dictionary"
    except IOError:
        print "No Segment Dictionary found - creating new one..."
        segdict = {}
    return segdict

def save_dict(dictionary, output):
    with open(output, "wb") as f:
        pickle.dump(dictionary, f)
    print "Saved dictionary to "+output+"."
    return
    
##Returns processed phones file
def read_in_transcription(path, phon_file, segdict):
    if phon_file != None:
        with open(path+phon_file, "r") as f:
            transcription = [segment.split() for segment in f.readlines()[9:]]
            for b in range(len(transcription)):
                ##Exclude non-speech sounds
                if (transcription[b][1] != "121") | \
                 (transcription[b][2] != "VOCNOISE") | \
                 (transcription[b][2] != "SIL"): 
                    if transcription[b][2] not in segdict:
                        segdict[transcription[b][2]]= 0
                    transcription[b][2]=segdict[transcription[b][2]]
                else:
                    transcription[b][2]=0
        ##Returns (start time, classification [121/2], transcription_index)
        return transcription
    else:
        return None
        
        
##Prepares 
def get_important_info(path, phon_file, segdict, freq_dict, tuple_dict):
    a =  len(segdict)+1
    if phon_file != None:
        with open(path+phon_file, "r") as f:
            transcription = [segment.split() for segment in f.readlines()[9:]]
            numbers = []
            for b in range(len(transcription)):
                    if transcription[b][2] not in segdict:
                        segdict[transcription[b][2]]= a
                        a+=1
                    transcription[b][2]=segdict[transcription[b][2]]
                    numbers.append(transcription[b][2])
                    if transcription[b][2] not in freq_dict:
                        freq_dict[transcription[b][2]]=0
                    freq_dict[transcription[b][2]]+=1
            number_tuple=[numbers[tuple_size*c:(c+1)*(tuple_size)] for c in range(len(numbers)/tuple_size)]
            for item in number_tuple:
                if item[-1] not in tuple_dict:
                    tuple_dict[item[-1]]=0
                tuple_dict[item[-1]]+=1
                    
        ##Returns (start time, classification [121/2], transcription_index)
        return segdict, freq_dict, tuple_dict
    else:
        return None
        
##Reads in wav file, windows, and computes relavant features for audio
def read_in_features(path, sound, segdict, transcription=None, output_sound=False):
    print "Reading: "+path+sound
    sampling_rate, wav_array = scipy.io.wavfile.read(path+sound)
    num_ticks = int(sampling_rate*window_length)
    a = 0
    trindex = 0
    features_list = []
    ##If transcription is present, include this information with features
    if transcription != None:
        for trindex in  range(len(transcription)-1):
            ##Window up to (but not beyond) the end of the segment
            windowed_sound = wav_array[int(sampling_rate*float(transcription[trindex][0])): \
            int(sampling_rate*float(transcription[trindex+1][0]))]
            if len(windowed_sound) > 0:
                features_list= features_list+[(item, transcription[trindex][2]) for \
                item in python_speech_features.mfcc(windowed_sound, samplerate=sampling_rate, \
                winlen=window_length, winstep=step)] ## <- MFCCs
        
        trindex+=1 
        windowed_sound = wav_array[int(sampling_rate*float(transcription[trindex][0]))-1:len(wav_array)]
        ##Get relevant features from windowed audio
        ##fft_list.append((abs(numpy.fft.fft(windowed_sound, n=num_ticks)), transcription[trindex][2])) <- Naive FFT
        if len(windowed_sound) > 0:
            features_list= features_list+[(item, transcription[trindex][2]) for item in python_speech_features.mfcc(windowed_sound, samplerate=sampling_rate, \
            winlen=window_length, winstep=step)] ## <- MFCCs
        ##Returns [...,(DFT, transcription at this point),...]
        return features_list
    ##If not, just include features on their own
    else:
        ##Get relevant fetaures from windowed audio
        ##feature_list.append((abs(numpy.fft.fft(windowed_sound, n=num_ticks)), None)) <- Naive FFT
        features_list = [(item, None) for item in \
        python_speech_features.mfcc(windowed_sound, samplerate=sampling_rate, \
        winlen=window_length, winstep=step)] ## <- MFCCs        
        ##Returns [...,(DFT, None),...]
        return features_list

        
"""
def w_categorical_crossentropy(y_true, y_pred, weights):
    
    differences = K.abs(y_pred - y_true)
    weightings = K.repeat(K.reshape(K.variable(array(weights)), (1,len(weights))), batch_size)
    final_mask = K.sum(differences, axis=2)#K.sum(differences * weightings, axis=2)
    
    return K.categorical_crossentropy(y_pred, y_true) * final_mask
"""        

if __name__ == "__main__":    
    trainpath = "./data/"
    testpath = "./data/"
    segment_dict = "./segment_dict.pkl"
    freq_dict = "./frequency_dict.pkl"
    modelpath = "./SpeechRecognitionModel.h5"
    ##segdict = open_segdict(segment_dict)
    ##hidden_units = 70
    files = get_files(trainpath)
    
    ##Build and save dictionaries that store segment identities and frequencies
    segdict = {}
    freqdict = {}
    tupledict = {}
    
    fft= []
   
    for file in files:
        segdict, freqdict, tupledict  = get_important_info(trainpath, file[1], segdict, freqdict, tupledict)
     
    print tupledict
    balance_number = min(tupledict.values())*9
    
    
    ###If we can only load one file at a time...
    for file in files:
        transcription = read_in_transcription(trainpath, file[1], segdict)
        fft = read_in_features(trainpath, file[0], segdict, transcription)
        slices = [fft[a*tuple_size:(a+1)*(tuple_size)] for a in range(len(fft)/tuple_size)]
        final_array = [[] for a in range(len(segdict))]
        for slice in slices:
            possibility = randint(0,balance_number)
            if slice[-1][-1] < len(final_array):
                if len(final_array[slice[-1][-1]]) < possibility:
                    final_array[slice[-1][-1]].append(slice)
            else:
                print slice[-1]
    model_input = []
    for bucket in final_array:
        model_input= model_input+[val for sublist in bucket for val in sublist]
    model_input = array(model_input)
    save_dict(segdict, segment_dict)
    save_dict(freqdict, freq_dict)
    
    ##freq_sum = sum(freqdict.values())
    
    ##Create weighted penalties to discourage preference for common segments (e.g. /t/)
    ##normalizing_factor = float(freq_sum)/min(freqdict.values())
    ##weighted_penalties=[float(item[1])/(freq_sum+1) for item in sorted(freqdict.items())]
    ##ncce = partial(w_categorical_crossentropy, weights=weighted_penalties)
    ##update_wrapper(ncce, w_categorical_crossentropy)
      
    ##Initialize Model
    model = Sequential()
    
    model.add(Bidirectional(LSTM(output_dim=len(segdict), init='uniform', \
     inner_init='uniform',forget_bias_init='one', return_sequences=True, activation='tanh', \
     inner_activation='sigmoid',), merge_mode='sum', input_shape = (batch_size,coefficients)))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(len(segdict), activation='sigmoid')))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(len(segdict), activation='sigmoid')))
    model.add(Activation('softmax'))
    
    rms = RMSprop()
    
    model.compile(loss="categorical_crossentropy",
        optimizer=rms,
        metrics=["categorical_crossentropy"])

        
    ##model = load_model(modelpath)
    
    model.summary()

    ##Train Model
    for file in files:
        ##transcription = read_in_transcription(trainpath, file[1], segdict)
        ##fft = read_in_features(trainpath, file[0], segdict, transcription)       
            
        
        X_train = array([array(item[0]) for item in model_input])
        X_train = array(array([X_train[x:x+batch_size] for x in range(0, len(X_train), batch_size)]))
        X_train = sequence.pad_sequences(X_train, maxlen=batch_size)
        X_train = reshape(X_train, (len(X_train), len(X_train[0]), len(X_train[0][0])))
        #print fft
        Y_train_values = [[1 if i == item[1] else e for i, e in enumerate(zeros(len(segdict)))] for item in model_input]
        #print Y_train_values
        Y_train = array([Y_train_values[x:x+batch_size] for x in range(0, len(Y_train_values), batch_size)])
        Y_train = sequence.pad_sequences(Y_train, maxlen=batch_size)
        
        hist = model.fit(X_train, Y_train, batch_size=1, nb_epoch=100, validation_split=0.2, verbose=1)
        
        ##Get Predictions
        predictions = model.predict(X_train, batch_size=1)
        sorted_dict = sorted(segdict.items(), key=lambda x:x[1])
        predicted_output=[]
        for item in predictions:
            for segment in item:
                segment_list = list(segment)
                max_index = segment_list.index(max(segment_list))
                if max_index != 0:
                    predicted_output.append(sorted_dict[max_index-1][0])
                else:
                    predicted_output.append(0)
        print predicted_output   
    model.save("./SpeechRecognitionModel.h5")
    """
    
    ##Model Predictions
    for file in files:
        fft = read_in_fft(testpath, file[0])
        X_test = numpy.array([numpy.array(item[0]) for item in fft])
        X_test = numpy.array(numpy.array([X_test[x:x+batch_size] for x in range(0, len(X_test), batch_size)]))
        X_test = sequence.pad_sequences(X_test, maxlen=batch_size)
        X_test = numpy.reshape(X_test, (len(X_test), len(X_test[0]), len(X_test[0][0])))
        Y_test_values = [[1 if i == item[1] else e for i, e in enumerate(numpy.zeros(len(segdict)))] for item in fft]
        Y_test = numpy.array([Y_test_values[x:x+batch_size] for x in range(0, len(Y_test_values), batch_size)])
        Y_test = sequence.pad_sequences(Y_test, maxlen=batch_size)
        Y_test = numpy.reshape(Y_test, (len(Y_test), len(Y_test[0]), len(Y_test[0][0])))
                
        ##Get Predictions
        predictions = model.predict(X_test, batch_size=1)
        sorted_dict = sorted(segdict.items(), key=lambda x:x[1])
        predicted_output=[]
        for item in predictions:
            for segment in item:
                segment_list = list(segment)
                max_index = segment_list.index(max(segment_list))
                if max_index != 0:
                    predicted_output.append(sorted_dict[max_index-1][0])
                else:
                    predicted_output.append(0)
        print predicted_output
        time.sleep(5)
    """