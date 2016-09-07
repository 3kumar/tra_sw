# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 14:21:23 2016

This script is used to create subplots for activations of given sentences.

@author: fox
"""
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

try:
    import cPickle as pickle
except:
    import pickle as pickle

def load_activations(corpus='462_45'):
    """
        returns:
            sentence_activations: list of arrays where each array is the activation of a sentence
            tokenized_sentences: list of list of tokenizes sentences
    """
    data_file='../outputs/activations/corpus-'+corpus+'.act'
    with open(data_file,'r') as f:
        data=pickle.load(f)
        sentence_activations=data[0]
        tokenized_sentences=data[1]
    return sentence_activations,tokenized_sentences

def plot_outputs(corpus='45',subplots=2,plot_noun=2,sentences_order=None):
    """
        subplots: No of subplots to be drawn
        sentence_order:list of sentences numbers in order to be drawn in subplots

    Note: no of subplots should be same as length of sentences_order
    """

    if subplots!=len(sentences_order):
        raise Exception("No of subplots should be same length of sentence order")

    sent_activations,tok_sentences=load_activations(corpus=corpus)

    plt.close('all')

    if subplots==2:
        f, axs= plt.subplots(1, 2, sharey='row')
    elif subplots==4:
        f, axs = plt.subplots(2, 2, sharey='row')

    for ax_index,ax in enumerate(f.axes):
        sent_no=sentences_order[ax_index]
        s_act=sent_activations[sent_no]
        tok_sent=tok_sentences[sent_no]

        for j in range(nr_nouns):
            if j==plot_noun-1:
                ax.plot(s_act[:,TOSpN*j:TOSpN*(j+1)])
                if ax_index==0:
                    ax.legend(unique_labels[TOSpN*j:TOSpN*(j+1)],fancybox=True,prop=fontP,loc="upper left").get_frame().set_alpha(0.4)

                ax.set_title("Sent-"+str(sent_no+1)+ ": '"+" ".join(tok_sent[1:-1]))
                ax.set_xticks(range(0,s_act.shape[0]))
                tok_sent[0]=""
                tok_sent[-1]=""
                ax.set_xticklabels(tok_sent,rotation=40)
                ax.axhline(y=0, c="brown", linewidth=1)
                ax.set_ylim([-2.0,2.0])

    plt.show()

if __name__=="__main__":

    nr_nouns=4
    TOSpN = 3*2  #number of Total Output Signal per Noun (for AOR, it's 3 times the number of verbs)
    unique_labels=['N1-A1','N1-O1','N1-R1','N1-A2','N1-O2','N1-R2','N2-A1','N2-O1','N2-R1','N2-A2','N2-O2','N2-R2',
                   'N3-A1','N3-O1','N3-R1','N3-A2','N3-O2','N3-R2','N4-A1','N4-O1','N4-R1','N4-A2','N4-O2','N4-R2']

    fontP = FontProperties()
    fontP.set_size('small')

    plot_outputs(corpus='45',subplots=4,plot_noun=2,sentences_order=[17-15,23-15,27-15,16-15])