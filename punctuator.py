# coding: utf-8
from __future__ import division

import re

import models
import punctuator_data as data

import theano
import sys
import codecs

import theano.tensor as T
import numpy as np


class Punctuator():

    def __init__(self,model_file,max_subseq=200):
        self.MAX_SUBSEQUENCE_LEN = max_subseq

        x = T.imatrix('x')
        
        #print "Loading model parameters..."
        net, _ = models.load(model_file, 1, x)

        #print "Building model..."
        self.predict = theano.function(
            inputs=[x],
            outputs=net.y
            )

        self.word_vocabulary = net.x_vocabulary
        self.punctuation_vocabulary = net.y_vocabulary
        
        # remap annotated punctuation tokens other than space to just the punctuation mark
        for key in self.punctuation_vocabulary.keys():
            if not "SPACE" in key:
                punc_mark = "".join([char for char in key if not char.isalpha()])
                self.punctuation_vocabulary[punc_mark] = self.punctuation_vocabulary.pop(key)

        self.reverse_word_vocabulary = {v:k for k,v in self.word_vocabulary.items()}
        self.reverse_punctuation_vocabulary = {v:k for k,v in self.punctuation_vocabulary.items()}

    def clean(self,text):
        """Remove spacing before apostrophes"""

        subbed = re.sub(" '(?!cause)","'",text)
        # make sure file ends with EOS token
        if subbed[-1] not in [".","?","!"]:
            subbed = ".".join([subbed,""])
        
        return subbed

    def to_array(self,arr,dtype=np.int32):

        # minibatch of 1 sequence as column
        return np.array([arr], dtype=dtype).T

    def restore(self,text):
        i = 0
        output_segments = []

        while True:

            subsequence = text[i:i+self.MAX_SUBSEQUENCE_LEN]

            if len(subsequence) == 0:
                break

            converted_subsequence = [self.word_vocabulary.get(w,self.word_vocabulary[data.UNK]) for w in subsequence]

            y = self.predict(self.to_array(converted_subsequence))

            output_segments.append(subsequence[0])

            last_eos_idx = 0
            punctuations = []
            for y_t in y:

                p_i = np.argmax(y_t.flatten())
                punctuation = self.reverse_punctuation_vocabulary[p_i]
                punctuations.append(punctuation)

                if punctuation in data.EOS_TOKENS:
                    last_eos_idx = len(punctuations) # we intentionally want the index of next element

            if subsequence[-1] == data.END:
                step = len(subsequence) - 1
            elif last_eos_idx != 0:
                step = last_eos_idx
            else:
                step = len(subsequence) - 1

            for j in range(step):
                output_segments.append(punctuations[j] + " " if punctuations[j] != data.SPACE else " ")
                if j < step - 1:
                    output_segments.append(subsequence[1+j])

            if subsequence[-1] == data.END:
                break

            i += step

        return "".join(output_segments)

    def punctuate(self,input_text):

        text = [w for w in input_text.split() if w not in self.punctuation_vocabulary and w not in data.PUNCTUATION_MAPPING and not w.startswith(data.PAUSE_PREFIX)] + [data.END]
        pauses = [float(s.replace(data.PAUSE_PREFIX,"").replace(">","")) for s in input_text.split() if s.startswith(data.PAUSE_PREFIX)]
        
        result = self.restore(text)
        cleaned = self.clean(result)

        return cleaned
