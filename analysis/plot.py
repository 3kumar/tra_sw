# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 10:57:14 2015

@author: twiefel
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
#plt.ioff()
import numpy as np
import sys
import glob
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import AxesGrid
#plot(i,sent,idx,ocw,states_out_train[i],rcl)
def plot(sent_idx,sent,word_idx, word, states_out_train, labels,config_name,N,sr,iss,line_to_end,prefix):
    #matplotlib.use('Agg')
    print "plotting"
    sent_dir = str(sent_idx)+"_"+sent.replace(" ","_")
    out_dir = prefix+'results/'+sent_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    #print word_idx, word
    #print states_out_train.shape
    #print states_out_train[0].shape
    sentence=  states_out_train#[sent_idx]
    #print sentence
    outputs = sentence[:,word_idx*14:(word_idx+1)*14]
    #outputs = np.random.random_sample((10,14))

    #print outputs

    #print outputs.shape
    markers = ["s",",","o","v","<",">","1","2","3","4","8",".","p","*"]
    fig = plt.figure(1)
    fig.clear()
    ax = plt.subplot(111)
    #print outputs
    #print outputs.T
    #print outputs.shape
    #print outputs
    outsT = outputs.T
    #mu, sigma = 200, 25
    #x = mu + sigma*np.random.randn(1000,14)
    #print x.shape
    #n, bins, patches = ax.hist(x, 14, histtype='bar')
    #ax.legend()
    #plt.show()
    #return
    #ax.hist(outputs)
    print "outputs shape: ",outputs.shape
    print "outputs shape: ",outputs.shape
    for idx,output in enumerate(outsT):
        "data length:",output.shape
        ax.plot(output,marker=markers[idx])
        #ax.hist(output)
    #ax.plot(outputs[0:11,1:2],marker="o")
    ax.set_xticks(range(outputs.shape[0]))
    #axes = ax.xaxis
    ax.xaxis.set_major_locator(plt.FixedLocator(range(0, outputs.shape[0], 2)))
    ax.xaxis.set_minor_locator(plt.FixedLocator(range(1, outputs.shape[0], 2)))
    ax.xaxis.set_minor_formatter(plt.FormatStrFormatter("%d"))
    ax.tick_params(which='major', pad=20, axis='x')
    ax.set_xticklabels(("_ "+sent+" _").split(" ")[::2])
    ax.xaxis.set_ticklabels(("_ "+sent+" _").split(" ")[1::2],minor=True)
    lgd=ax.legend(labels,loc='upper left', bbox_to_anchor=(1.0, 1.0))
    #ax.subplots_adjust(right=1)
    #print fig.get_default_bbox_extra_artists()
    filename = str(word_idx)+"_"+word+"_"+config_name+'_'+str(N)+'_'+str(sr)+'_'+str(iss)+'_'+str(line_to_end)
    fig.savefig(out_dir+"/"+filename+'.png',bbox_extra_artists=(lgd,),bbox_inches='tight')
    plt.show()
    print "plotted"
def blub():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(-5, 10, .25)
    Y = np.arange(-5, 5, .25)
    print X.shape
    print Y.shape
    X, Y = np.meshgrid(X, Y)
    print X.shape
    print Y.shape
    R = np.sqrt(X**2 + Y**2)
    print R.shape
    Z = np.sin(R)
    print Z.shape
    Gx, Gy = np.gradient(Z) # gradients with respect to x and y
    print Z.shape
    print Gx.shape
    print Gy.shape
    G = (Gx**2+Gy**2)**.5  # gradient magnitude
    print G.shape
    N = G/G.max()  # normalize 0..1
    print N.shape
    surf = ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1,
        facecolors=cm.jet(N),
        linewidth=0, antialiased=False, shade=False)
    plt.show()

def plot_exp_1():
    with open('/informatik/isr/wtm/home/twiefel/ros_workspace/scripts/iCub_code_for_Language_Comprehension/iCub_language_src/iCub_language/tests_15_01_28/results.csv') as f:
        content = f.readlines()
    #print content
    #print content[0::9]
    idx = 0

    graph_labels = []
    wers = []
    sers = []

    for line in content[1:]:
        if idx >=9:
            idx = 0
        row = line.split(';')
        graph_labels.append(row[0])
        wers.append(row[5])
        sers.append(row[6])


    print graph_labels
    print wers
    print sers

    graph_labels=graph_labels[::9]
    wers=zip(*[iter(wers)]*9)
    sers=zip(*[iter(sers)]*9)
    print graph_labels
    print wers
    print sers
    fig = plt.figure(1)
    corpus_size=[10,50,100,250,500,1000,1500,2000,2500]
    matplotlib.rcParams.update({'font.size': 18})

    ax = plt.subplot(111)
    for idx,label in enumerate(graph_labels):
        marker = 'x'
        n = label.split("_")[-1]
        method = "BL"

        if idx >= len(graph_labels)/2:
            marker = 'o'
            method ="CBL"

        graph_labels[idx]=method+" "+n
        print idx
        col_idx = idx
        if col_idx>=5:
            col_idx-=5
        plt.plot(wers[idx], label='WER '+label, marker = marker,color=get_color(col_idx,5))
        #plt.setp(lines, color=0, linewidth=2.0)

    plt.ylabel('Word Error Rate')
    plt.xlabel('Corpus Size')#
    lgd=ax.legend(graph_labels,loc='upper left', bbox_to_anchor=(1.0, 1.0))
    ax.set_xticklabels(corpus_size)
    #plt.show()

    #fig.savefig('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/'+'exp_1.png',bbox_extra_artists=(lgd,),bbox_inches='tight')
    pp = PdfPages('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/'+'exp_1.pdf')
    plt.savefig(pp,format='pdf',bbox_extra_artists=(lgd,),bbox_inches='tight')
    pp.close()
    plt.show()

def plot_exp_2_new(method,vmin,vmax):
    csv_file='/media/wtmgws3_data/experiments/15.02.16_11:24/results.csv'
    df = pd.read_csv(csv_file,sep=';')
    sr = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
    iss = [0.1,0.4,0.7,1,1.3,1.6,1.9,2.2]
    lr = [0.1,0.3,0.5,0.7,0.9]
    lr_labels = [0.1,0.5,0.9]
    #wer = list(df['Word Error Rate'])[:480]
    if method is 'bl':
        wer = list(df['Word Error Rate'])[:480]
    else:
        wer = list(df['Word Error Rate'])[480:960]

    #sr = np.array(sr)
    #iss = np.array(iss)
    #lr = np.array(lr)
    wer = np.array(wer)
    #iss:8
    #sr:10
    #lr:6
    matplotlib.rcParams.update({'font.size': 7})
    wer_reshaped = np.reshape(wer,(8,10,6))

    print wer_reshaped.shape
    print wer_reshaped[0]
    print wer_reshaped[0,:,:-1]
    #gs = gridspec.GridSpec(3, 3)
    #
    fig = plt.figure(1)
    #plt.imshow(wer_reshaped[0])
    #plt.show()
    #gs.update(left=0.05, right=0.48, hspace=.17,wspace=0.01)
    gs = ImageGrid(fig, 111, # similar to subplot(111)
                nrows_ncols = (3, 3), # creates 2x2 grid of axes
                axes_pad=0.1, # pad between axes in inch.
                add_all=True,
                cbar_mode='None',
                cbar_location='right',
                share_all=True,
                direction='row',
                ngrids=9
                )
    #gs.update(wspace=0.01)
    for i in range(8):
        ax = gs[i]
        ax.set_xticks([0,2,4])
        ax.set_xticklabels(lr_labels)
        im = ax.imshow(wer_reshaped[i,:,:-1],cmap = cm.Greys_r,vmin=vmin, vmax=vmax,  aspect='auto')

    fig.colorbar(im,cax=gs[i])
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #plt.colorbar(im, cax=cax)
    #gs[0].imshow(wer_reshaped[0])#,cmap = cm.Greys_r)
    #gs[8].colorbar()
    #divider = make_axes_locatable(gs[8])
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #gs[8].colorbar(im, cax=cax)
    #gs.cbar_axes[7].colorbar(im)
    pre = 'IS = '
    #grid.cbar_axes[i].colorbar(im)
    #gs.set_yticklabels(sr)
    plt.show()
    return

    pre = 'IS = '
    ax = plt.subplot(gs[0,0])
    ax.set_title(pre+str(iss[0]))
    plt.imshow(wer_reshaped[0],cmap = cm.Greys_r,vmin=vmin, vmax=vmax,  aspect='auto')
    #ax.set_xticks(range(3))
    ax.set_xticklabels('')
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    ax.set_yticklabels(sr)
    ax.set_yticks(range(len(sr)))
    plt.ylabel('Spectral Radius')
    #plt.xlabel('Leak Rate')
    ax = plt.subplot(gs[0,1])
    #ax.set_title(iss[1])
    ax.set_title(pre+str(iss[1]))
    plt.imshow(wer_reshaped[1],cmap = cm.Greys_r,vmin=vmin, vmax=vmax,  aspect='auto')
    #ax.set_xticklabels(lr)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_yticks(range(len(sr)))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    #plt.ylabel('Spectral Radius')
    #plt.xlabel('Leak Rate')
    ax = plt.subplot(gs[0,2])
    #ax.set_title(iss[2])
    ax.set_title(pre+str(iss[2]))
    plt.imshow(wer_reshaped[2],cmap = cm.Greys_r,vmin=vmin, vmax=vmax,  aspect='auto')
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_yticks(range(len(sr)))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    #ax.set_xticklabels(lr)
    #plt.ylabel('Spectral Radius')
    #plt.xlabel('Leak Rate')
    ax = plt.subplot(gs[1,0])
    #ax.set_title(iss[3])
    ax.set_title(pre+str(iss[3]))
    plt.imshow(wer_reshaped[3],cmap = cm.Greys_r,vmin=vmin, vmax=vmax,  aspect='auto')
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    #ax.set_xticklabels(lr)
    ax.set_yticklabels(sr)
    ax.set_yticks(range(len(sr)))
    plt.ylabel('Spectral Radius')
    #plt.xlabel('Leak Rate')
    ax = plt.subplot(gs[1,1])
    #ax.set_title(iss[4])
    ax.set_title(pre+str(iss[4]))
    plt.imshow(wer_reshaped[4],cmap = cm.Greys_r,vmin=vmin, vmax=vmax,  aspect='auto')
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_yticks(range(len(sr)))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    #ax.set_xticklabels(lr)
    #plt.ylabel('Spectral Radius')
    #plt.xlabel('Leak Rate')
    ax = plt.subplot(gs[1,2])
    #ax.set_title(iss[5])
    ax.set_title(pre+str(iss[5]))
    plt.imshow(wer_reshaped[5],cmap = cm.Greys_r,vmin=vmin, vmax=vmax,  aspect='auto')
    ax.set_xticklabels(lr_labels)
    ax.set_yticklabels('')
    ax.set_yticks(range(len(sr)))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    #ax.set_xticklabels(lr)
    #plt.ylabel('Spectral Radius')
    #plt.colorbar()
    plt.xlabel('Leak Rate')
    ax = plt.subplot(gs[2,0])
    #ax.set_title(iss[6])
    ax.set_title(pre+str(iss[6]))
    plt.imshow(wer_reshaped[6],cmap = cm.Greys_r,vmin=vmin, vmax=vmax,  aspect='auto')
    #ax.set_xticklabels('')
    #ax.set_xticks(range(4))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    ax.set_xticklabels(lr_labels)
    ax.set_yticklabels(sr)
    ax.set_yticks(range(len(sr)))
    plt.ylabel('Spectral Radius')
    plt.xlabel('Leak Rate')
    ax = plt.subplot(gs[2,1])
    #ax.set_title(iss[7])
    ax.set_title(pre+str(iss[7]))
    im = plt.imshow(wer_reshaped[7],cmap = cm.Greys_r,vmin=vmin, vmax=vmax,  aspect='auto')
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    ax.set_xticklabels(lr_labels)
    ax.set_yticklabels('')
    ax.set_yticks(range(len(sr)))
    ax.set_anchor('W')
    #plt.ylabel('Spectral Radius')
    plt.xlabel('Leak Rate')

    #ax = plt.subplot(gs[2,2])
    ax = plt.subplot(gs[2,2])
    cbar = fig.colorbar(im,cax=ax)
    cbar.ax.set_ylabel('     Word Error Rate', rotation=270)
    #gs.tight_layout(fig)
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    #plt.tight_layout()
    #fig.savefig('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/'+'exp_2_'+method+'.png',bbox_inches='tight')
    pp = PdfPages('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/'+'exp_2_'+method+'.pdf')
    plt.savefig(pp,format='pdf',bbox_inches='tight')
    pp.close()
    plt.show()

def plot_exp_2(method,vmin,vmax):
    csv_file='/media/wtmgws3_data/experiments/15.02.16_11:24/results.csv'
    df = pd.read_csv(csv_file,sep=';')
    sr = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
    iss = [0.1,0.4,0.7,1,1.3,1.6,1.9,2.2]
    lr = ['',0.1,0.3,0.5,0.7,0.9,1.1]
    lr_labels = ['',0.1,0.5,0.9]
    #wer = list(df['Word Error Rate'])[:480]
    wer = list(df['Word Error Rate'])[:480]
    #wer = list(df['Word Error Rate'])[480:960]

    #sr = np.array(sr)
    #iss = np.array(iss)
    #lr = np.array(lr)
    wer = np.array(wer)
    #iss:8
    #sr:10
    #lr:6
    matplotlib.rcParams.update({'font.size': 7})
    wer_reshaped = np.reshape(wer,(8,10,6))

    print wer_reshaped.shape
    print wer_reshaped[0]

    gs = gridspec.GridSpec(3, 7)
    #
    gs.update(left=0.05, right=1, hspace=.7, wspace=0.1, bottom=0.3)
    #gs.update(wspace=0.01)
    fig = plt.figure(1)
    pre = 'IS = '
    ax = plt.subplot(gs[0,0])
    ax.set_title(pre+str(iss[0]))
    plt.imshow(wer_reshaped[0],cmap = cm.Greys_r,vmin=vmin, vmax=vmax,  aspect='auto')
    #ax.set_xticks(range(3))
    ax.set_xticklabels('')
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    ax.set_yticklabels(sr)
    ax.set_yticks(range(len(sr)))
    plt.ylabel('Spectral Radius')
    #plt.xlabel('Leak Rate')
    ax = plt.subplot(gs[0,1])
    #ax.set_title(iss[1])

    plt.imshow(wer_reshaped[1],cmap = cm.Greys_r,vmin=vmin, vmax=vmax,  aspect='auto')
    #ax.set_xticklabels(lr)
    ax.set_title(pre+str(iss[1]))
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_yticks(range(len(sr)))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    #plt.ylabel('Spectral Radius')
    #plt.xlabel('Leak Rate')
    ax = plt.subplot(gs[0,2])
    #ax.set_title(iss[2])
    ax.set_title(pre+str(iss[2]))
    plt.imshow(wer_reshaped[2],cmap = cm.Greys_r,vmin=vmin, vmax=vmax,  aspect='auto')
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_yticks(range(len(sr)))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    #ax.set_xticklabels(lr)
    #plt.ylabel('Spectral Radius')
    #plt.xlabel('Leak Rate')
    ax = plt.subplot(gs[1,0])
    #ax.set_title(iss[3])
    ax.set_title(pre+str(iss[3]))
    plt.imshow(wer_reshaped[3],cmap = cm.Greys_r,vmin=vmin, vmax=vmax,  aspect='auto')
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    #ax.set_xticklabels(lr)
    ax.set_yticklabels(sr)
    ax.set_yticks(range(len(sr)))
    plt.ylabel('Spectral Radius')
    #plt.xlabel('Leak Rate')
    ax = plt.subplot(gs[1,1])
    #ax.set_title(iss[4])
    ax.set_title(pre+str(iss[4]))
    plt.imshow(wer_reshaped[4],cmap = cm.Greys_r,vmin=vmin, vmax=vmax,  aspect='auto')
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_yticks(range(len(sr)))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    #ax.set_xticklabels(lr)
    #plt.ylabel('Spectral Radius')
    #plt.xlabel('Leak Rate')
    ax = plt.subplot(gs[1,2])
    #ax.set_title(iss[5])
    ax.set_title(pre+str(iss[5]))
    plt.imshow(wer_reshaped[5],cmap = cm.Greys_r,vmin=vmin, vmax=vmax,  aspect='auto')
    ax.set_xticklabels(lr_labels)
    ax.set_yticklabels('')
    ax.set_yticks(range(len(sr)))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    #ax.set_xticklabels(lr)
    #plt.ylabel('Spectral Radius')
    #plt.colorbar()
    plt.xlabel('Leak Rate')
    ax = plt.subplot(gs[2,0])
    #ax.set_title(iss[6])
    ax.set_title(pre+str(iss[6]))
    plt.imshow(wer_reshaped[6],cmap = cm.Greys_r,vmin=vmin, vmax=vmax,  aspect='auto')
    #ax.set_xticklabels('')
    #ax.set_xticks(range(4))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    ax.set_xticklabels(lr_labels)
    ax.set_yticklabels(sr)
    ax.set_yticks(range(len(sr)))
    plt.ylabel('Spectral Radius')
    plt.xlabel('Leak Rate')
    ax = plt.subplot(gs[2,1])
    #ax.set_title(iss[7])
    ax.set_title(pre+str(iss[7]))
    im = plt.imshow(wer_reshaped[7],cmap = cm.Greys_r,vmin=vmin, vmax=vmax,  aspect='auto')
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    ax.set_xticklabels(lr_labels)
    ax.set_yticklabels('')
    ax.set_yticks(range(len(sr)))
    ax.set_anchor('W')
    #plt.ylabel('Spectral Radius')
    plt.xlabel('Leak Rate')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    wer = list(df['Word Error Rate'])[480:960]

    #sr = np.array(sr)
    #iss = np.array(iss)
    #lr = np.array(lr)
    wer = np.array(wer)
    #iss:8
    #sr:10
    #lr:6
    matplotlib.rcParams.update({'font.size': 7})
    wer_reshaped = np.reshape(wer,(8,10,6))

    print wer_reshaped.shape
    print wer_reshaped[0]

    #gs = gridspec.GridSpec(6, 3)
    #
    gs.update(left=0.05, right=0.48, hspace=.17,wspace=0.01)
    #gs.update(wspace=0.01)
    fig = plt.figure(1)
    pre = 'IS = '
    ax = plt.subplot(gs[0,4])
    ax.set_title(pre+str(iss[0]))
    plt.imshow(wer_reshaped[0],cmap = cm.Greys_r,vmin=vmin, vmax=vmax,  aspect='auto')
    #ax.set_xticks(range(3))
    ax.set_xticklabels('')
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    ax.set_yticklabels(sr)
    ax.set_yticks(range(len(sr)))
    plt.ylabel('Spectral Radius')
    #plt.xlabel('Leak Rate')
    ax = plt.subplot(gs[0,5])
    #ax.set_title(iss[1])
    ax.set_title(pre+str(iss[1]))
    plt.imshow(wer_reshaped[1],cmap = cm.Greys_r,vmin=vmin, vmax=vmax,  aspect='auto')
    #ax.set_xticklabels(lr)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_yticks(range(len(sr)))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    #plt.ylabel('Spectral Radius')
    #plt.xlabel('Leak Rate')
    ax = plt.subplot(gs[0,6])
    #ax.set_title(iss[2])
    ax.set_title(pre+str(iss[2]))
    plt.imshow(wer_reshaped[2],cmap = cm.Greys_r,vmin=vmin, vmax=vmax,  aspect='auto')
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_yticks(range(len(sr)))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    #ax.set_xticklabels(lr)
    #plt.ylabel('Spectral Radius')
    #plt.xlabel('Leak Rate')
    ax = plt.subplot(gs[1,4])
    #ax.set_title(iss[3])
    ax.set_title(pre+str(iss[3]))
    plt.imshow(wer_reshaped[3],cmap = cm.Greys_r,vmin=vmin, vmax=vmax,  aspect='auto')
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    #ax.set_xticklabels(lr)
    ax.set_yticklabels(sr)
    ax.set_yticks(range(len(sr)))
    plt.ylabel('Spectral Radius')
    #plt.xlabel('Leak Rate')
    ax = plt.subplot(gs[1,5])
    #ax.set_title(iss[4])
    ax.set_title(pre+str(iss[4]))
    plt.imshow(wer_reshaped[4],cmap = cm.Greys_r,vmin=vmin, vmax=vmax,  aspect='auto')
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_yticks(range(len(sr)))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    #ax.set_xticklabels(lr)
    #plt.ylabel('Spectral Radius')
    #plt.xlabel('Leak Rate')
    ax = plt.subplot(gs[1,6])
    #ax.set_title(iss[5])
    ax.set_title(pre+str(iss[5]))
    plt.imshow(wer_reshaped[5],cmap = cm.Greys_r,vmin=vmin, vmax=vmax,  aspect='auto')
    ax.set_xticklabels(lr_labels)
    ax.set_yticklabels('')
    ax.set_yticks(range(len(sr)))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    #ax.set_xticklabels(lr)
    #plt.ylabel('Spectral Radius')
    #plt.colorbar()
    plt.xlabel('Leak Rate')
    ax = plt.subplot(gs[2,4])
    #ax.set_title(iss[6])
    ax.set_title(pre+str(iss[6]))
    plt.imshow(wer_reshaped[6],cmap = cm.Greys_r,vmin=vmin, vmax=vmax,  aspect='auto')
    #ax.set_xticklabels('')
    #ax.set_xticks(range(4))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    ax.set_xticklabels(lr_labels)
    ax.set_yticklabels(sr)
    ax.set_yticks(range(len(sr)))
    plt.ylabel('Spectral Radius')
    plt.xlabel('Leak Rate')
    ax = plt.subplot(gs[2,5])
    #ax.set_title(iss[7])
    ax.set_title(pre+str(iss[7]))
    im = plt.imshow(wer_reshaped[7],cmap = cm.Greys_r,vmin=vmin, vmax=vmax,  aspect='auto')
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    ax.set_xticklabels(lr_labels)
    ax.set_yticklabels('')
    ax.set_yticks(range(len(sr)))
    ax.set_anchor('W')
    #plt.ylabel('Spectral Radius')
    plt.xlabel('Leak Rate')

    #ax = plt.subplot(gs[2,2])
    #ax = plt.subplot(gs[2,2])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    #cbar = fig.colorbar(im,cax=ax)
    #cbar.ax.set_ylabel('     Word Error Rate', rotation=270)
    #gs.tight_layout(fig)
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    #plt.tight_layout()
    #fig.savefig('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/'+'exp_2_'+method+'.png',bbox_inches='tight')
    pp = PdfPages('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/'+'exp_2_'+method+'.pdf')
    plt.savefig(pp,format='pdf',bbox_inches='tight')
    pp.close()
    plt.show()

def plot_exp_3(csv_file_1,csv_file_2,csv_file_3,csv_file_4,csv_file_5,outfile):

    matplotlib.rcParams.update({'font.size': 10})
    fig = plt.figure(1)

    #csv_file='/media/wtmgws3_data/experiments/15.02.19_12:09/results.csv'
    df = pd.read_csv(csv_file_1,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    ax = plt.subplot(231)
    wer_reshaped = np.reshape(wer,(29,8))

    labels=np.arange(0.1,0.81,0.1)
    #label_leak = mdp.numx.arange(0.1, 0.81, 0.1)
    x_scale = 4
    label_sr = np.arange(3, 10.1, 0.25*x_scale)
    for i in range(8):
        data=wer_reshaped[:,i]
        #print data
        plt.plot(data, label="LR="+str(labels[i]),color=get_color(i,9))


    #ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
    #ax.set_xticklabels(label_sr)
    ax.set_xticklabels('')
    plt.ylabel('Word Error Rate')
    ax.set_title("N=1000")
    ax.set_ylim([10,75])
    #plt.xlabel('Spectral Radius')


    #csv_file='/media/wtmgws3_data/experiments/15.02.19_21:54/results.csv'
    df = pd.read_csv(csv_file_2,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    ax = plt.subplot(232)
    wer_reshaped = np.reshape(wer,(29,8))

    labels=np.arange(0.1,0.81,0.1)
    #label_leak = mdp.numx.arange(0.1, 0.81, 0.1)
    #x_scale = 4
    label_sr = np.arange(3, 10.1, 0.25*x_scale)
    for i in range(8):
        data=wer_reshaped[:,i]
        #print data
        plt.plot(data, label="LR="+str(labels[i]),color=get_color(i,9))

    ax.set_title("N=2000")
    ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
    ax.set_xticklabels(label_sr)
    ax.set_yticklabels('')
    ax.set_ylim([10,75])
    #plt.ylabel('Word Error Rate')
    plt.xlabel('Spectral Radius')
    #lgd=ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

    #csv_file='/media/wtmgws3_data/experiments/15.02.22_01:43_0_0/results.csv'
    df = pd.read_csv(csv_file_3,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    ax = plt.subplot(233)
    #wer_reshaped = np.reshape(wer,(29,8))



    labels=np.arange(0.1,0.81,0.1)
    #label_leak = mdp.numx.arange(0.1, 0.81, 0.1)
    #x_scale = 2
    label_sr = np.arange(3, 10.1, 0.25*x_scale)

    for i in range(3):
        data = wer[i*2::6]
        print data
        plt.plot(data, label="LR="+str(labels[i]),color=get_color(i,9))

    #csv_file='/media/wtmgws3_data/experiments/15.02.22_01:44_1_0/results.csv'
    df = pd.read_csv(csv_file_4,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    for i in range(3):
        data = wer[i*2::6]
        #print data
        plt.plot(data, label="LR="+str(labels[i+3]),color=get_color(i+3,9))

    #csv_file='/media/wtmgws3_data/experiments/15.02.23_11:03_2_0/results.csv'
    df = pd.read_csv(csv_file_5,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    for i in range(2):
        data = wer[i*2::6]
        #print data
        plt.plot(data, label="LR="+str(labels[i+6]),color=get_color(i+6,9))

    ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
    ax.set_xticklabels(label_sr)
    plt.ylabel('Word Error Rate')
    plt.xlabel('Spectral Radius')
    ax.set_title("N=5000")
    ax.set_ylim([10,75])
    #LR=0.1 - 0.3
    #/media/wtmgws3_data/experiments/15.02.20_21:58_0_0
    #/media/wtmgws3_data/experiments/15.02.22_01:43_0_0
    #LR=0.4 - 0.6
    #/media/wtmgws3_data/experiments/15.02.20_23:33_1_0
    #/media/wtmgws3_data/experiments/15.02.22_01:44_1_0
    #LR=0.7 - 0.9
    #/media/wtmgws3_data/experiments/15.02.23_11:03_2_0



    lgd=ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    #fig.savefig('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/'+ outfile,bbox_extra_artists=(lgd,),bbox_inches='tight')
    pp = PdfPages('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/'+ outfile+'.pdf')
    plt.savefig(pp,format='pdf',bbox_extra_artists=(lgd,),bbox_inches='tight')
    pp.close()
    plt.show()

def plot_exp_3_all(csv_file_1,csv_file_2,csv_file_3,csv_file_4,csv_file_5,
                   csv_file_1_2,csv_file_2_2,csv_file_3_2,csv_file_4_2,csv_file_5_2 ):




    matplotlib.rcParams.update({'font.size': 15})
    fig = plt.figure(1)

    #csv_file='/media/wtmgws3_data/experiments/15.02.19_12:09/results.csv'
    df = pd.read_csv(csv_file_1,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    ax = plt.subplot(231)
    wer_reshaped = np.reshape(wer,(29,8))

    labels=np.arange(0.1,0.81,0.1)
    #label_leak = mdp.numx.arange(0.1, 0.81, 0.1)
    x_scale = 9
    label_sr = np.arange(3, 10.1, 0.25*x_scale)
    for i in range(8):
        data=wer_reshaped[:,i]
        #print data
        plt.plot(data, label="LR="+str(labels[i]),color=get_color(i,9))


    #ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
    #ax.set_xticklabels(label_sr)
    ax.set_xticklabels('')
    plt.ylabel('Word Error Rate')
    ax.set_title("N=1000")
    ax.set_ylim([5,75])
    #plt.xlabel('Spectral Radius')


    #csv_file='/media/wtmgws3_data/experiments/15.02.19_21:54/results.csv'
    df = pd.read_csv(csv_file_2,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    ax = plt.subplot(232)
    wer_reshaped = np.reshape(wer,(29,8))

    labels=np.arange(0.1,0.81,0.1)
    #label_leak = mdp.numx.arange(0.1, 0.81, 0.1)
    #x_scale = 4
    label_sr = np.arange(3, 10.1, 0.25*x_scale)
    for i in range(8):
        data=wer_reshaped[:,i]
        #print data
        plt.plot(data, label="LR="+str(labels[i]),color=get_color(i,9))

    ax.set_title("N=2000")
    ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
    #ax.set_xticklabels(label_sr)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_ylim([5,75])
    #plt.ylabel('Word Error Rate')
    #plt.xlabel('Spectral Radius')
    #lgd=ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

    #csv_file='/media/wtmgws3_data/experiments/15.02.22_01:43_0_0/results.csv'
    df = pd.read_csv(csv_file_3,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    ax = plt.subplot(233)
    #wer_reshaped = np.reshape(wer,(29,8))



    labels=np.arange(0.1,0.81,0.1)
    #label_leak = mdp.numx.arange(0.1, 0.81, 0.1)
    #x_scale = 2
    label_sr = np.arange(3, 10.1, 0.25*x_scale)

    for i in range(3):
        data = wer[i*2::6]
        print data
        plt.plot(data, label="LR="+str(labels[i]),color=get_color(i,9))

    #csv_file='/media/wtmgws3_data/experiments/15.02.22_01:44_1_0/results.csv'
    df = pd.read_csv(csv_file_4,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    for i in range(3):
        data = wer[i*2::6]
        #print data
        plt.plot(data, label="LR="+str(labels[i+3]),color=get_color(i+3,9))

    #csv_file='/media/wtmgws3_data/experiments/15.02.23_11:03_2_0/results.csv'
    df = pd.read_csv(csv_file_5,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    for i in range(2):
        data = wer[i*2::6]
        #print data
        plt.plot(data, label="LR="+str(labels[i+6]),color=get_color(i+6,9))

    ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
    #ax.set_xticklabels(label_sr)
    ax.set_xticklabels('')
    #plt.ylabel('Word Error Rate')
    ax.set_yticklabels('')
    #plt.xlabel('Spectral Radius')
    #plt.xlabel('Spectral Radius')
    ax.set_title("N=5000")
    ax.set_ylim([5,75])
    #LR=0.1 - 0.3
    #/media/wtmgws3_data/experiments/15.02.20_21:58_0_0
    #/media/wtmgws3_data/experiments/15.02.22_01:43_0_0
    #LR=0.4 - 0.6
    #/media/wtmgws3_data/experiments/15.02.20_23:33_1_0
    #/media/wtmgws3_data/experiments/15.02.22_01:44_1_0
    #LR=0.7 - 0.9
    #/media/wtmgws3_data/experiments/15.02.23_11:03_2_0

    df = pd.read_csv(csv_file_1_2,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    ax = plt.subplot(234)
    wer_reshaped = np.reshape(wer,(29,8))

    labels=np.arange(0.1,0.81,0.1)
    #label_leak = mdp.numx.arange(0.1, 0.81, 0.1)
    #x_scale = 4

    label_sr = np.arange(3, 10.1, 0.25*x_scale)
    for i in range(8):
        data=wer_reshaped[:,i]
        #print data
        plt.plot(data, label="LR="+str(labels[i]),color=get_color(i,9))


    ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
    ax.set_xticklabels(label_sr)
    #ax.set_xticklabels('')
    plt.ylabel('Word Error Rate')
    #ax.set_title("N=1000")
    ax.set_ylim([5,75])
    plt.xlabel('Spectral Radius')


    #csv_file='/media/wtmgws3_data/experiments/15.02.19_21:54/results.csv'
    df = pd.read_csv(csv_file_2_2,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    ax = plt.subplot(235)
    wer_reshaped = np.reshape(wer,(29,8))

    labels=np.arange(0.1,0.81,0.1)
    #label_leak = mdp.numx.arange(0.1, 0.81, 0.1)
    #x_scale = 4
    label_sr = np.arange(3, 10.1, 0.25*x_scale)
    for i in range(8):
        data=wer_reshaped[:,i]
        #print data
        plt.plot(data, label="LR="+str(labels[i]),color=get_color(i,9))

    #ax.set_title("N=2000")
    ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
    ax.set_xticklabels(label_sr)
    ax.set_yticklabels('')
    ax.set_ylim([5,75])
    #plt.ylabel('Word Error Rate')
    plt.xlabel('Spectral Radius')
    #lgd=ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

    #csv_file='/media/wtmgws3_data/experiments/15.02.22_01:43_0_0/results.csv'
    df = pd.read_csv(csv_file_3_2,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    ax = plt.subplot(236)
    #wer_reshaped = np.reshape(wer,(29,8))



    labels=np.arange(0.1,0.81,0.1)
    #label_leak = mdp.numx.arange(0.1, 0.81, 0.1)
    #x_scale = 2
    label_sr = np.arange(3, 10.1, 0.25*x_scale)

    for i in range(3):
        data = wer[i*2::6]
        print data
        plt.plot(data, label="LR="+str(labels[i]),color=get_color(i,9))

    #csv_file='/media/wtmgws3_data/experiments/15.02.22_01:44_1_0/results.csv'
    df = pd.read_csv(csv_file_4_2,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    for i in range(3):
        data = wer[i*2::6]
        #print data
        plt.plot(data, label="LR="+str(labels[i+3]),color=get_color(i+3,9))

    #csv_file='/media/wtmgws3_data/experiments/15.02.23_11:03_2_0/results.csv'
    df = pd.read_csv(csv_file_5_2,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    for i in range(2):
        data = wer[i*2::6]
        #print data
        plt.plot(data, label="LR="+str(labels[i+6]),color=get_color(i+6,9))

    ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
    ax.set_xticklabels(label_sr)
    ax.set_yticklabels('')
    #plt.ylabel('Word Error Rate')
    plt.xlabel('Spectral Radius')
    #ax.set_title("N=5000")
    ax.set_ylim([5,75])
    #LR=0.1 - 0.3
    #/media/wtmgws3_data/experiments/15.02.20_21:58_0_0
    #/media/wtmgws3_data/experiments/15.02.22_01:43_0_0
    #LR=0.4 - 0.6
    #/media/wtmgws3_data/experiments/15.02.20_23:33_1_0
    #/media/wtmgws3_data/experiments/15.02.22_01:44_1_0
    #LR=0.7 - 0.9
    #/media/wtmgws3_data/experiments/15.02.23_11:03_2_0



    lgd=ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    #fig.savefig('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/'+ outfile,bbox_extra_artists=(lgd,),bbox_inches='tight')
    pp = PdfPages('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/exp_3.pdf')
    plt.savefig(pp,format='pdf',bbox_extra_artists=(lgd,),bbox_inches='tight')
    pp.close()
    plt.show()

def plot_exp_3_cbl():

    #/media/wtmgws3_data/experiments/15.02.19_12:11
    #method=CBL
    #N=1000
    #SR=3-10
    #ISS=1.5
    #LR=0.1-0.8

     #/media/wtmgws3_data/experiments/15.02.19_20:41
    #method=CBL
    #N=2000
    #SR=3-10
    #ISS=1.5
    #LR=0.1-0.8

    #'/media/wtmgws3_data/experiments/15.02.19_12:11/results.csv'
    #'/media/wtmgws3_data/experiments/15.02.19_20:41/results.csv'
    #'/media/wtmgws2_data/experiments/15.02.22_22:15_0_1/results.csv'
    #'/media/wtmgws2_data/experiments/15.02.22_22:08_1_1/results.csv'
    #'/media/wtmgws2_data/experiments/15.02.22_01:08_2_1/results.csv'




    matplotlib.rcParams.update({'font.size': 10})
    fig = plt.figure(1)

    csv_file='/media/wtmgws3_data/experiments/15.02.19_12:11/results.csv'
    df = pd.read_csv(csv_file,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    ax = plt.subplot(221)
    wer_reshaped = np.reshape(wer,(29,8))


    labels=np.arange(0.1,0.81,0.1)
    #label_leak = mdp.numx.arange(0.1, 0.81, 0.1)
    x_scale = 4
    label_sr = np.arange(3, 10.1, 0.25*x_scale)
    for i in range(8):
        data=wer_reshaped[:,i]
        #print data
        plt.plot(data, label="LR="+str(labels[i]))

    ax.set_title("N=1000")
    ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
    ax.set_xticklabels(label_sr)
    plt.ylabel('Word Error Rate')
    plt.xlabel('Spectral Radius')


    csv_file='/media/wtmgws3_data/experiments/15.02.19_20:41/results.csv'
    df = pd.read_csv(csv_file,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    ax = plt.subplot(222)
    wer_reshaped = np.reshape(wer,(29,8))

    labels=np.arange(0.1,0.81,0.1)
    #label_leak = mdp.numx.arange(0.1, 0.81, 0.1)
    #x_scale = 4
    label_sr = np.arange(3, 10.1, 0.25*x_scale)
    for i in range(8):
        data=wer_reshaped[:,i]
        #print data
        plt.plot(data, label="LR="+str(labels[i]))

    ax.set_title("N=2000")
    ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
    ax.set_xticklabels(label_sr)
    plt.ylabel('Word Error Rate')
    plt.xlabel('Spectral Radius')


    csv_file='/media/wtmgws2_data/experiments/15.02.22_22:15_0_1/results.csv'
    df = pd.read_csv(csv_file,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    ax = plt.subplot(223)
    #wer_reshaped = np.reshape(wer,(29,8))



    labels=np.arange(0.1,0.81,0.1)
    #label_leak = mdp.numx.arange(0.1, 0.81, 0.1)
    #x_scale = 2
    label_sr = np.arange(3, 10.1, 0.25*x_scale)

    for i in range(3):
        data = wer[i::6]
        #print data
        plt.plot(data, label="LR="+str(labels[i]))

    csv_file='/media/wtmgws2_data/experiments/15.02.22_22:08_1_1/results.csv'
    df = pd.read_csv(csv_file,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    for i in range(3):
        data = wer[i::6]
        #print data
        plt.plot(data, label="LR="+str(labels[i+3]))

    csv_file='/media/wtmgws2_data/experiments/15.02.22_01:08_2_1/results.csv'
    df = pd.read_csv(csv_file,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    for i in range(2):
        data = wer[i::6]
        #print data
        plt.plot(data, label="LR="+str(labels[i+6]))

    ax.set_title("N=5000")
    ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
    ax.set_xticklabels(label_sr)
    plt.ylabel('Word Error Rate')
    plt.xlabel('Spectral Radius')

    #LR=0.1 - 0.3
    #/media/wtmgws3_data/experiments/15.02.20_21:58_0_0
    #/media/wtmgws3_data/experiments/15.02.22_01:43_0_0
    #LR=0.4 - 0.6
    #/media/wtmgws3_data/experiments/15.02.20_23:33_1_0
    #/media/wtmgws3_data/experiments/15.02.22_01:44_1_0
    #LR=0.7 - 0.9
    #/media/wtmgws3_data/experiments/15.02.23_11:03_2_0

    #'/media/wtmgws2_data/experiments/15.02.22_22:15_0_1/results.csv'
    #'/media/wtmgws2_data/experiments/15.02.22_22:08_1_1/results.csv'
    #'/media/wtmgws2_data/experiments/15.02.22_01:08_2_1/results.csv'

    lgd=ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.savefig('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/'+'exp_3_cbl.png',bbox_extra_artists=(lgd,),bbox_inches='tight')
    plt.show()

def plot_exp_4():

    matplotlib.rcParams.update({'font.size': 24})

    fig = plt.figure(1)

    csv_file='/media/wtmgws3_data/experiments/15.02.24_21:34_0/results.csv'
    df = pd.read_csv(csv_file,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)


    wer_reshaped = np.reshape(wer,(3,29,9))

    #lgd=ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    csv_file='/media/wtmgws3_data/experiments/15.02.24_21:36_1/results.csv'
    df = pd.read_csv(csv_file,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)
    wer_reshaped2 = np.reshape(wer,(3,29,9))

    labels=np.arange(0.1,0.91,0.1)
    #label_leak = mdp.numx.arange(0.1, 0.81, 0.1)
    x_scale = 1
    label_sr = np.arange(3, 10.1, 0.25*x_scale)


    ax =fig.add_subplot(111)
    for i in range(9):
        data1=wer_reshaped[0,:,i]
        data2=wer_reshaped[1,:,i]
        data3=wer_reshaped[2,:,i]
        data4=wer_reshaped2[0,:,i]
        data5=wer_reshaped2[1,:,i]
        data6=wer_reshaped2[2,:,i]
        mean = np.mean([data1,data2,data3,data4,data5,data6],axis=0)
        std = np.std([data1,data2,data3,data4,data5,data6],axis=0)
        print mean.shape
        print label_sr.shape
        #plt.plot(np.mean([data1,data2,data3,data4,data5,data6],axis=0), label="LR="+str(labels[i]))
        plt.errorbar(label_sr,mean ,std, label="LR="+str(labels[i]),marker='^')

    plt.xlabel('Spectral Radius')
    plt.ylabel('Word Error Rate')
    lgd=ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    #plt.title("RS=5000, Trainset=2083, Testset=417")
    pp = PdfPages('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/'+'exp_4.pdf')
    plt.savefig(pp,format='pdf',bbox_extra_artists=(lgd,),bbox_inches='tight')
    pp.close()
    plt.show()


def plot_exp_4_old():







    matplotlib.rcParams.update({'font.size': 10})
    fig = plt.figure(1)

    csv_file='/media/wtmgws3_data/experiments/15.02.24_21:34_0/results.csv'
    df = pd.read_csv(csv_file,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)


    wer_reshaped = np.reshape(wer,(3,29,9))



    labels=np.arange(0.1,0.91,0.1)
    #label_leak = mdp.numx.arange(0.1, 0.81, 0.1)
    x_scale = 8
    label_sr = np.arange(3, 10.1, 0.25*x_scale)
    gs = gridspec.GridSpec(2, 3)
    #
    gs.update(left=0.05, right=0.48, hspace=.17,wspace=0.01,bottom=0.5)
    #grid = 231
    for j in range(3):
        ax = plt.subplot(gs[j])
        for i in range(9):
            data=wer_reshaped[j,:,i]
            print data
            #print data
            plt.plot(data, label="LR="+str(labels[i]))
        #ax.set_xticklabels('')
        ax = plt.subplot(gs[j])
        for i in range(9):
            data=wer_reshaped[j,:,i]
            print data
            #print data
            plt.plot(data, label="LR="+str(labels[i]))
        ax.set_xticklabels('')
        ax.set_yticklabels('')
        #ax.set_yticklabels('')
        ax = plt.subplot(gs[j])
        for i in range(9):
            data=wer_reshaped[j,:,i]
            print data
            #print data
            plt.plot(data, label="LR="+str(labels[i]))
        ax.set_xticklabels('')
        ax.set_yticklabels('')
        ax.set_ylim([25,60])
        #plt.ylabel('Word Error Rate')
    lgd=ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    csv_file='/media/wtmgws3_data/experiments/15.02.24_21:36_1/results.csv'
    df = pd.read_csv(csv_file,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)
    for j in range(3):
        ax = plt.subplot(gs[j+3])
        for i in range(9):
            data=wer_reshaped[j,:,i]
            print data
            #print data
            plt.plot(data, label="LR="+str(labels[i]))

        ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
        ax.set_xticklabels(label_sr)
        ax = plt.subplot(gs[j+3])
        plt.xlabel('Spectral Radius')
        plt.ylabel('Word Error Rate')
        for i in range(9):
            data=wer_reshaped[j,:,i]
            print data
            #print data
            plt.plot(data, label="LR="+str(labels[i]))
        ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
        ax.set_xticklabels(label_sr)
        ax.set_yticklabels('')
        ax = plt.subplot(gs[j+3])

        plt.xlabel('Spectral Radius')
        for i in range(9):
            data=wer_reshaped[j,:,i]
            print data
            #print data
            plt.plot(data, label="LR="+str(labels[i]))
        ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
        ax.set_xticklabels(label_sr)
        ax.set_yticklabels('')
        plt.xlabel('Spectral Radius')
        ax.set_ylim([25,60])




    #LR=0.1 - 0.3
    #/media/wtmgws3_data/experiments/15.02.20_21:58_0_0
    #/media/wtmgws3_data/experiments/15.02.22_01:43_0_0
    #LR=0.4 - 0.6
    #/media/wtmgws3_data/experiments/15.02.20_23:33_1_0
    #/media/wtmgws3_data/experiments/15.02.22_01:44_1_0
    #LR=0.7 - 0.9
    #/media/wtmgws3_data/experiments/15.02.23_11:03_2_0





    #fig.savefig('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/'+'exp_4_fold_'+str(fold)+'.png',bbox_extra_artists=(lgd,),bbox_inches='tight')
    pp = PdfPages('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/'+'exp_4.pdf')
    plt.savefig(pp,format='pdf',bbox_extra_artists=(lgd,),bbox_inches='tight')
    pp.close()
    plt.show()

def plot_exp_4_1_fold(k):







    matplotlib.rcParams.update({'font.size': 10})
    fig = plt.figure(1)
    csv_file=''
    fold = k
    if k<3:
        csv_file='/media/wtmgws3_data/experiments/15.02.24_21:34_0/results.csv'
    else:
        csv_file='/media/wtmgws3_data/experiments/15.02.24_21:36_1/results.csv'
        k-=3

    df = pd.read_csv(csv_file,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)


    wer_reshaped = np.reshape(wer,(3,29,9))



    labels=np.arange(0.1,0.91,0.1)
    #label_leak = mdp.numx.arange(0.1, 0.81, 0.1)
    x_scale = 4
    label_sr = np.arange(3, 10.1, 0.25*x_scale)

    #
    ax = plt.subplot(111)
    for i in range(9):
        data=wer_reshaped[k,:,i]
        print data
        #print data
        plt.plot(data, label="LR="+str(labels[i]),color=get_color(i,9))

    ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
    ax.set_xticklabels(label_sr)
    plt.ylabel('Word Error Rate')
    plt.xlabel('Spectral Radius')
    ax.set_ylim([25,55])





    #LR=0.1 - 0.3
    #/media/wtmgws3_data/experiments/15.02.20_21:58_0_0
    #/media/wtmgws3_data/experiments/15.02.22_01:43_0_0
    #LR=0.4 - 0.6
    #/media/wtmgws3_data/experiments/15.02.20_23:33_1_0
    #/media/wtmgws3_data/experiments/15.02.22_01:44_1_0
    #LR=0.7 - 0.9
    #/media/wtmgws3_data/experiments/15.02.23_11:03_2_0



    lgd=ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    #fig.savefig('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/'+'exp_4_fold_'+str(fold)+'.png',bbox_extra_artists=(lgd,),bbox_inches='tight')
    pp = PdfPages('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/'+'exp_4_fold_'+str(fold)+'.pdf')
    plt.savefig(pp,format='pdf',bbox_extra_artists=(lgd,),bbox_inches='tight')
    pp.close()
    plt.show()




def plot_exp_5_bl():





    #'/media/wtmgws3_data/experiments/15.02.22_01:43_0_0/results.csv'
    #'/media/wtmgws3_data/experiments/15.02.22_01:44_1_0/results.csv'
    #'/media/wtmgws3_data/experiments/15.02.23_11:03_2_0/results.csv'

    matplotlib.rcParams.update({'font.size': 10})
    fig = plt.figure(1)


    csv_file='/media/wtmgws3_data/experiments/15.02.22_01:43_0_0/results.csv'
    df = pd.read_csv(csv_file,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    ax = plt.subplot(111)
    #wer_reshaped = np.reshape(wer,(29,8))


    x_scale=4
    labels=np.arange(0.1,0.91,0.1)
    #label_leak = mdp.numx.arange(0.1, 0.81, 0.1)
    #x_scale = 2
    label_sr = np.arange(3, 10.1, 0.25*x_scale)

    for i in range(3):
        data = wer[i*2+1::6]
        print data
        plt.plot(data, label="LR="+str(labels[i]))

    csv_file='/media/wtmgws3_data/experiments/15.02.22_01:44_1_0/results.csv'
    df = pd.read_csv(csv_file,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    for i in range(3):
        data = wer[i*2+1::6]
        print data
        plt.plot(data, label="LR="+str(labels[i+3]))

    csv_file='/media/wtmgws3_data/experiments/15.02.23_11:03_2_0/results.csv'
    df = pd.read_csv(csv_file,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    for i in range(3):
        data = wer[i*2+1::6]
        print data
        plt.plot(data, label="LR="+str(labels[i+6]))
    ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
    ax.set_xticklabels(label_sr)
    plt.ylabel('Word Error Rate')
    plt.xlabel('Spectral Radius')

    #LR=0.1 - 0.3
    #/media/wtmgws3_data/experiments/15.02.20_21:58_0_0
    #/media/wtmgws3_data/experiments/15.02.22_01:43_0_0
    #LR=0.4 - 0.6
    #/media/wtmgws3_data/experiments/15.02.20_23:33_1_0
    #/media/wtmgws3_data/experiments/15.02.22_01:44_1_0
    #LR=0.7 - 0.9
    #/media/wtmgws3_data/experiments/15.02.23_11:03_2_0



    lgd=ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.savefig('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/'+'exp_5_bl.png',bbox_extra_artists=(lgd,),bbox_inches='tight')
    plt.show()

def get_color(idx,total,scaling=1):
    total*=scaling
    idx*=scaling
    values = range(total)
    jet = cm = plt.get_cmap('jet')
    cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    return scalarMap.to_rgba(values[idx])
def plot_exp_5_cbl():

    matplotlib.rcParams.update({'font.size': 10})
    fig = plt.figure(1)
    #'/media/wtmgws2_data/experiments/15.02.22_22:15_0_1/results.csv'
    #'/media/wtmgws2_data/experiments/15.02.22_22:08_1_1/results.csv'
    #'/media/wtmgws2_data/experiments/15.02.22_01:08_2_1/results.csv'
    csv_file='/media/wtmgws2_data/experiments/15.02.22_22:15_0_1/results.csv'
    df = pd.read_csv(csv_file,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    ax = plt.subplot(111)
    #wer_reshaped = np.reshape(wer,(29,8))

    x_scale=4
    labels=np.arange(0.1,0.91,0.1)
    #label_leak = mdp.numx.arange(0.1, 0.81, 0.1)
    #x_scale = 2
    label_sr = np.arange(3, 10.1, 0.25*x_scale)

    for i in range(3):
        data = wer[i*2+1::6]
        print data
        plt.plot(data, label="LR="+str(labels[i]),color=get_color(i,9))

    csv_file='/media/wtmgws2_data/experiments/15.02.22_22:08_1_1/results.csv'
    df = pd.read_csv(csv_file,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    for i in range(3):
        data = wer[i*2+1::6]
        print data
        plt.plot(data, label="LR="+str(labels[i+3]),color=get_color(i+3,9))

    csv_file='/media/wtmgws2_data/experiments/15.02.22_01:08_2_1/results.csv'
    df = pd.read_csv(csv_file,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    for i in range(3):
        data = wer[i*2+1::6]
        print data
        plt.plot(data, label="LR="+str(labels[i+6]),color=get_color(i+6,9))
    ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
    ax.set_xticklabels(label_sr)
    plt.ylabel('Word Error Rate')
    plt.xlabel('Spectral Radius')


    lgd=ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.savefig('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/'+'exp_5_cbl.png',bbox_extra_artists=(lgd,),bbox_inches='tight')
    plt.show()

def plot_exp_5(csv_file_1,csv_file_2,csv_file_3,outfile):

    matplotlib.rcParams.update({'font.size': 10})
    fig = plt.figure(1)
    #'/media/wtmgws2_data/experiments/15.02.22_22:15_0_1/results.csv'
    #'/media/wtmgws2_data/experiments/15.02.22_22:08_1_1/results.csv'
    #'/media/wtmgws2_data/experiments/15.02.22_01:08_2_1/results.csv'
    #csv_file='/media/wtmgws2_data/experiments/15.02.22_22:15_0_1/results.csv'
    df = pd.read_csv(csv_file_1,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    ax = plt.subplot(111)
    #wer_reshaped = np.reshape(wer,(29,8))

    x_scale=4
    labels=np.arange(0.1,0.91,0.1)
    #label_leak = mdp.numx.arange(0.1, 0.81, 0.1)
    #x_scale = 2
    label_sr = np.arange(3, 10.1, 0.25*x_scale)

    for i in range(3):
        data = wer[i*2+1::6]
        print data
        plt.plot(data, label="LR="+str(labels[i]),color=get_color(i,9))

    #csv_file='/media/wtmgws2_data/experiments/15.02.22_22:08_1_1/results.csv'
    df = pd.read_csv(csv_file_2,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    for i in range(3):
        data = wer[i*2+1::6]
        print data
        plt.plot(data, label="LR="+str(labels[i+3]),color=get_color(i+3,9))

    #csv_file='/media/wtmgws2_data/experiments/15.02.22_01:08_2_1/results.csv'
    df = pd.read_csv(csv_file_3,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    for i in range(3):
        data = wer[i*2+1::6]
        print data
        plt.plot(data, label="LR="+str(labels[i+6]),color=get_color(i+6,9))
    ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
    ax.set_xticklabels(label_sr)
    plt.ylabel('Word Error Rate')
    plt.xlabel('Spectral Radius')


    lgd=ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    #fig.savefig('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/'+outfile,bbox_extra_artists=(lgd,),bbox_inches='tight')
    pp = PdfPages('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/'+outfile+'.pdf')
    plt.savefig(pp,format='pdf',bbox_extra_artists=(lgd,),bbox_inches='tight')
    pp.close()
    plt.show()

def plot_exp_5_all(csv_file_1,csv_file_2,csv_file_3,csv_file_1_2,csv_file_2_2,csv_file_3_2):

    matplotlib.rcParams.update({'font.size': 15})
    fig = plt.figure(1)
    #'/media/wtmgws2_data/experiments/15.02.22_22:15_0_1/results.csv'
    #'/media/wtmgws2_data/experiments/15.02.22_22:08_1_1/results.csv'
    #'/media/wtmgws2_data/experiments/15.02.22_01:08_2_1/results.csv'
    #csv_file='/media/wtmgws2_data/experiments/15.02.22_22:15_0_1/results.csv'
    df = pd.read_csv(csv_file_1,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    ax = plt.subplot(221)
    #wer_reshaped = np.reshape(wer,(29,8))

    x_scale=4
    labels=np.arange(0.1,0.91,0.1)
    #label_leak = mdp.numx.arange(0.1, 0.81, 0.1)
    #x_scale = 2
    label_sr = np.arange(3, 10.1, 0.25*x_scale)

    for i in range(3):
        data = wer[i*2+1::6]
        print data
        plt.plot(data, label="LR="+str(labels[i]),color=get_color(i,9))

    #csv_file='/media/wtmgws2_data/experiments/15.02.22_22:08_1_1/results.csv'
    df = pd.read_csv(csv_file_2,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    for i in range(3):
        data = wer[i*2+1::6]
        print data
        plt.plot(data, label="LR="+str(labels[i+3]),color=get_color(i+3,9))

    #csv_file='/media/wtmgws2_data/experiments/15.02.22_01:08_2_1/results.csv'
    df = pd.read_csv(csv_file_3,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    for i in range(3):
        data = wer[i*2+1::6]
        print data
        plt.plot(data, label="LR="+str(labels[i+6]),color=get_color(i+6,9))
    ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
    ax.set_xticklabels(label_sr)
    plt.ylabel('Word Error Rate')
    plt.xlabel('Spectral Radius')
    ax.set_ylim([30,75])
    #wer_reshaped = np.reshape(wer,(29,8))


    df = pd.read_csv(csv_file_1_2,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    ax = plt.subplot(222)
    x_scale=4
    labels=np.arange(0.1,0.91,0.1)
    #label_leak = mdp.numx.arange(0.1, 0.81, 0.1)
    #x_scale = 2
    label_sr = np.arange(3, 10.1, 0.25*x_scale)

    for i in range(3):
        data = wer[i*2+1::6]
        print data
        plt.plot(data, label="LR="+str(labels[i]),color=get_color(i,9))

    #csv_file='/media/wtmgws2_data/experiments/15.02.22_22:08_1_1/results.csv'
    df = pd.read_csv(csv_file_2_2,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    for i in range(3):
        data = wer[i*2+1::6]
        print data
        plt.plot(data, label="LR="+str(labels[i+3]),color=get_color(i+3,9))

    #csv_file='/media/wtmgws2_data/experiments/15.02.22_01:08_2_1/results.csv'
    df = pd.read_csv(csv_file_3_2,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    for i in range(3):
        data = wer[i*2+1::6]
        print data
        plt.plot(data, label="LR="+str(labels[i+6]),color=get_color(i+6,9))
    ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
    ax.set_xticklabels(label_sr)
    #plt.ylabel('Word Error Rate')
    ax.set_yticklabels('')
    plt.xlabel('Spectral Radius')
    ax.set_ylim([30,75])
    lgd=ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    #fig.savefig('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/'+outfile,bbox_extra_artists=(lgd,),bbox_inches='tight')
    pp = PdfPages('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/exp_5.pdf')
    plt.savefig(pp,format='pdf',bbox_extra_artists=(lgd,),bbox_inches='tight')
    pp.close()
    plt.show()

def plot_exp_6_old():







    matplotlib.rcParams.update({'font.size': 10})
    fig = plt.figure(1)


    csv_file='/media/wtmgws3_data/experiments/15.02.24_13:44_r_s/results.csv'
    df = pd.read_csv(csv_file,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    ax = plt.subplot(111)
    #wer_reshaped = np.reshape(wer,(29,8))



    #labels=np.arange(0.1,0.91,0.1)
    #label_leak = mdp.numx.arange(0.1, 0.81, 0.1)
    #x_scale = 2
    #label_sr = np.arange(3, 10.1, 0.25*x_scale)

    data = wer[0::2]
    print data
    plt.plot(data, label="Training")
    data = wer[1::2]
    print data
    plt.plot(data, label="Testing")
    ax.set_ylim([10,40])
    #ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
    #ax.set_xticklabels(label_sr)
    plt.ylabel('Word Error Rate')
    plt.xlabel('Instance')


    #LR=0.1 - 0.3
    #/media/wtmgws3_data/experiments/15.02.20_21:58_0_0
    #/media/wtmgws3_data/experiments/15.02.22_01:43_0_0
    #LR=0.4 - 0.6
    #/media/wtmgws3_data/experiments/15.02.20_23:33_1_0
    #/media/wtmgws3_data/experiments/15.02.22_01:44_1_0
    #LR=0.7 - 0.9
    #/media/wtmgws3_data/experiments/15.02.23_11:03_2_0



    lgd=ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    #fig.savefig('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/'+'exp_6.png',bbox_extra_artists=(lgd,),bbox_inches='tight')
    pp = PdfPages('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/'+'exp_6.pdf')
    plt.savefig(pp,format='pdf',bbox_extra_artists=(lgd,),bbox_inches='tight')
    pp.close()
    plt.show()


def plot_exp_6():







    matplotlib.rcParams.update({'font.size': 18})
    fig = plt.figure(1)


    csv_file='/media/wtmgws3_data/experiments/15.02.24_13:44_r_s/results.csv'
    df = pd.read_csv(csv_file,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    ax = plt.subplot(111)
    #wer_reshaped = np.reshape(wer,(29,8))



    #labels=np.arange(0.1,0.91,0.1)
    #label_leak = mdp.numx.arange(0.1, 0.81, 0.1)
    #x_scale = 2
    #label_sr = np.arange(3, 10.1, 0.25*x_scale)

    data = wer[0::2]
    print data
    print np.mean(data)
    print np.std(data)
    #plt.plot(data, label="Training")
    data = wer[1::2]
    print data
    print np.mean(data)
    print np.std(data)
    #print data
    #plt.plot(data, label="Testing")
    #ax.set_ylim([10,40])
    #ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
    #ax.set_xticklabels(label_sr)
    #plt.ylabel('Word Error Rate')
    #plt.xlabel('Instance')


    #LR=0.1 - 0.3
    #/media/wtmgws3_data/experiments/15.02.20_21:58_0_0
    #/media/wtmgws3_data/experiments/15.02.22_01:43_0_0
    #LR=0.4 - 0.6
    #/media/wtmgws3_data/experiments/15.02.20_23:33_1_0
    #/media/wtmgws3_data/experiments/15.02.22_01:44_1_0
    #LR=0.7 - 0.9
    #/media/wtmgws3_data/experiments/15.02.23_11:03_2_0



    #lgd=ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    #fig.savefig('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/'+'exp_6.png',bbox_extra_artists=(lgd,),bbox_inches='tight')
    #pp = PdfPages('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/'+'exp_6.pdf')
    #plt.savefig(pp,format='pdf',bbox_extra_artists=(lgd,),bbox_inches='tight')
    #pp.close()
    #plt.show()


def plot_exp_7(k,csv_file,name):
    """
    6 fold cross validation with word2vec
    """

    matplotlib.rcParams.update({'font.size': 10})

    fold = k
    print os.path.isfile(csv_file)
    df = pd.read_csv(csv_file,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)

    print wer.shape
    folds = 6 #folds
    srii = 29 #sr
    lrs = 9 #lr
    sets = 2

    print folds*srii*lrs*sets
    wer_reshaped = wer.reshape(folds,srii,lrs,sets)


    labels=np.arange(0.1,0.91,0.1)
    #label_leak = mdp.numx.arange(0.1, 0.81, 0.1)
    x_scale = 4
    label_sr = np.arange(3, 10.1, 0.25*x_scale)

    fig = plt.figure(1)
    ax = plt.subplot(111)
    for i in range(9):
        data=wer_reshaped[k,:,i,0]
        print data
        #print data
        plt.plot(data, label="LR="+str(labels[i]),color=get_color(i,9))

    ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
    ax.set_xticklabels(label_sr)
    plt.ylabel('Word Error Rate')
    plt.xlabel('Spectral Radius')
    ax.set_ylim([0,100])

    lgd=ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    #fig.savefig('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/'+'exp_4_fold_'+str(fold)+'.png',bbox_extra_artists=(lgd,),bbox_inches='tight')
    pp = PdfPages('./'+'exp_7_fold_'+str(fold)+'_'+name+'_train.pdf')
    plt.savefig(pp,format='pdf',bbox_extra_artists=(lgd,),bbox_inches='tight')
    pp.close()
    plt.show()
    plt.close()

    fig = plt.figure(1)
    ax = plt.subplot(111)
    for i in range(9):
        data=wer_reshaped[k,:,i,1]
        print data
        #print data
        plt.plot(data, label="LR="+str(labels[i]),color=get_color(i,9))

    ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
    ax.set_xticklabels(label_sr)
    plt.ylabel('Word Error Rate')
    plt.xlabel('Spectral Radius')
    ax.set_ylim([0,100])

    lgd=ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    #fig.savefig('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/'+'exp_4_fold_'+str(fold)+'.png',bbox_extra_artists=(lgd,),bbox_inches='tight')
    pp = PdfPages('./'+'exp_7_fold_'+str(fold)+'_'+name+'_test.pdf')
    plt.savefig(pp,format='pdf',bbox_extra_artists=(lgd,),bbox_inches='tight')
    pp.close()
    plt.close()
    #plt.show()
def plot_exp_7(out_folder,csv_file,name,srii,lrs,folds,sets):

    matplotlib.rcParams.update({'font.size': 24})



    df = pd.read_csv(csv_file,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)


    print wer.shape
    #folds = 6 #folds
    #srii = 11 #sr
    #lrs = 9 #lr
    #sets = 2
    #folds = 6 #folds
    #srii = 29 #sr
    #lrs = 9 #lr
    #sets = 2



    #param_leak = mdp.numx.arange(0.01, 0.091, 0.01)
    #param_sr = mdp.numx.arange(3, 10.1, 0.25)

    #labels=np.arange(0.01,0.091,0.01)

    #label_leak = mdp.numx.arange(0.1, 0.81, 0.1)
    #x_scale = 1
    #label_sr = np.arange(3, 10.1, 0.25*x_scale)
    #label_sr = np.arange(0.25, 2.76, 0.25*x_scale)
    #label_sr = np.arange(0.25, 4.01, 0.25*x_scale)

    #folds = 6 #folds
    len_srii = len(srii) #sr
    len_lrs = len(lrs) #lr
    #sets = 2

    print folds*len_srii*len_lrs*sets
    wer_reshaped = wer.reshape(folds,len_srii,len_lrs,sets)



    fig = plt.figure(1)
    ax =fig.add_subplot(111)
    for i in range(9):

        data1=wer_reshaped[0,:,i,0]
        data2=wer_reshaped[1,:,i,0]
        data3=wer_reshaped[2,:,i,0]
        data4=wer_reshaped[3,:,i,0]
        data5=wer_reshaped[4,:,i,0]
        data6=wer_reshaped[5,:,i,0]
        mean = np.mean([data1,data2,data3,data4,data5,data6],axis=0)
        std = np.std([data1,data2,data3,data4,data5,data6],axis=0)
        print mean.shape
        print srii.shape
        #plt.plot(np.mean([data1,data2,data3,data4,data5,data6],axis=0), label="LR="+str(labels[i]))
        plt.errorbar(srii,mean ,std, label="LR="+str(lrs[i]),marker='^')

    plt.xlabel('Spectral Radius')
    plt.ylabel('Word Error Rate')
    lgd=ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    #plt.title("RS=5000, Trainset=2083, Testset=417")
    pp = PdfPages(out_folder+'/'+'exp_7_'+name+'_train.pdf')
    plt.savefig(pp,format='pdf',bbox_extra_artists=(lgd,),bbox_inches='tight')
    pp.close()
    #plt.show()
    plt.close()
    fig = plt.figure(1)
    ax =fig.add_subplot(111)
    for i in range(9):
        data1=wer_reshaped[0,:,i,1]
        data2=wer_reshaped[1,:,i,1]
        data3=wer_reshaped[2,:,i,1]
        data4=wer_reshaped[3,:,i,1]
        data5=wer_reshaped[4,:,i,1]
        data6=wer_reshaped[5,:,i,1]
        mean = np.mean([data1,data2,data3,data4,data5,data6],axis=0)
        std = np.std([data1,data2,data3,data4,data5,data6],axis=0)
        print mean.shape
        print srii.shape
        #plt.plot(np.mean([data1,data2,data3,data4,data5,data6],axis=0), label="LR="+str(labels[i]))
        plt.errorbar(srii,mean ,std, label="LR="+str(lrs[i]),marker='^')

    plt.xlabel('Spectral Radius')
    plt.ylabel('Word Error Rate')
    lgd=ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    #plt.title("RS=5000, Trainset=2083, Testset=417")
    pp = PdfPages(out_folder+'/'+'exp_7_'+name+'_test.pdf')
    plt.savefig(pp,format='pdf',bbox_extra_artists=(lgd,),bbox_inches='tight')
    pp.close()
    #plt.show()
    plt.close()


def plot_exp_8():
    """
    a 6 fold cross validation with chunks
    """
    matplotlib.rcParams.update({'font.size': 24})

    fig = plt.figure(1)

    path = '/media/wtmgws6_data/experiments/15.08.18_16:28_cross_validation_chunks/'
    csv_file = path+'results.csv'
    df = pd.read_csv(csv_file,sep=';')
    wer = list(df['Word Error Rate'])
    wer = np.array(wer)


    runs_per_parameter = 2
    folds = 6
    labels=np.arange(0.1,0.91,0.1)
    x_scale = 1
    label_sr = np.arange(0.25, 10.1, 0.25*x_scale)
    print len(labels)
    print len(label_sr)
    print folds
    print folds *len(labels) *len(label_sr)
    print wer.shape
    wer_reshaped = np.reshape(wer,(folds,len(label_sr),len(labels),runs_per_parameter))
    print wer_reshaped[0][0][0][0]

    #label_leak = mdp.numx.arange(0.1, 0.81, 0.1)



    ax =fig.add_subplot(111)
    for i in range(9):
        data1=wer_reshaped[0,:,i]
        data2=wer_reshaped[1,:,i]
        data3=wer_reshaped[2,:,i]
        data4=wer_reshaped[3,:,i]
        data5=wer_reshaped[4,:,i]
        data6=wer_reshaped[5,:,i]
        mean = np.mean([data1,data2,data3,data4,data5,data6],axis=0)
        std = np.std([data1,data2,data3,data4,data5,data6],axis=0)
        print mean.shape
        print label_sr.shape
        #plt.plot(np.mean([data1,data2,data3,data4,data5,data6],axis=0), label="LR="+str(labels[i]))
        plt.errorbar(label_sr,mean ,std, label="LR="+str(labels[i]),marker='^')

    plt.xlabel('Spectral Radius')
    plt.ylabel('Word Error Rate')
    lgd=ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    #plt.title("RS=5000, Trainset=2083, Testset=417")
    pp = PdfPages(path+'cross_val_chunks.pdf')
    plt.savefig(pp,format='pdf',bbox_extra_artists=(lgd,),bbox_inches='tight')
    pp.close()
    plt.show()


import cPickle as pickle
import data_conversion as dc

def plot_exp_9(method,vmin,vmax):
    path = '/media/wtmgws6_data/experiments/15.10.16_12:51_improved_cross_validation/'

    output_labels = dc.getOutputLabels()
    #ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
    #ax.set_xticklabels(label_sr)

    matplotlib.rcParams.update({'font.size': 7})
    #fig = plt.figure(1)
    fig, ax = plt.subplots()

    folds = 10

    instances = 5
    param_leak = np.arange(0.1, 0.91, 0.1)
    param_sr = np.arange(0.25, 10.01, 0.25)

    average_mean_summed_squared_errors = []
    instance = 0
    for leak in param_leak:
        for sr in param_sr:
            data = pickle.load(open(path+"amse_"+str(instance)+"_"+str(leak)+"_"+str(sr)+".pkl","rb"))
            average_mean_summed_squared_errors.append(data)

    #im = plt.imshow(average_mean_summed_squared_errors,cmap = cm.Greys_r, vmin=vmin, vmax=vmax,  aspect='auto')
    im = plt.imshow(average_mean_summed_squared_errors,cmap = cm.Greys_r)
    plt.xticks(np.arange(0,1080,len(output_labels)))
    ax.set_xticklabels(np.arange(0,40,1))
    #ax2 = ax.twinx()

    fig.show()
    plt.show()
    print "exit"
def plot_exp_10(method,vmin,vmax):
    path = '/media/wtmgws6_data/experiments/15.10.16_12:51_improved_cross_validation/'

    output_labels = dc.getOutputLabels()
    #ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
    #ax.set_xticklabels(label_sr)

    matplotlib.rcParams.update({'font.size': 1})
    #fig = plt.figure(1)
    fig, ax = plt.subplots()

    folds = 10

    instances = 5
    param_leak = np.arange(0.1, 0.91, 0.1)
    param_sr = np.arange(0.25, 10.01, 0.25)

    average_mean_summed_squared_errors = []
    data_mask = []
    instance = 0
    empty = [1]*1080
    for leak in param_leak:
        for sr in param_sr:
            data = pickle.load(open(path+"amse_"+str(instance)+"_"+str(leak)+"_"+str(sr)+".pkl","rb"))
            average_mean_summed_squared_errors.append(data)
            #average_mean_summed_squared_errors.append(empty)

    average_mean_summed_squared_errors = np.array(average_mean_summed_squared_errors).T
    #im = plt.imshow(average_mean_summed_squared_errors,cmap = cm.Greys_r, vmin=vmin, vmax=vmax,  aspect='auto')
    im = plt.imshow(average_mean_summed_squared_errors,cmap = cm.Greys_r)

    #plt.yticks(np.arange(0,1080,len(output_labels)))
    #ax.set_yticklabels(np.arange(0,40,1))

    plt.yticks(np.arange(0,1080,1))
    ax.set_yticklabels(np.tile(np.arange(0,len(output_labels),1),40))
    #ax2 = ax.twinx()

    pp = PdfPages(path+'cross_val_chunks.pdf')
    plt.savefig(pp,format='pdf',bbox_inches='tight')
    pp.close()

    fig.show()
    plt.show()
    print "exit"

def plot_exp_11(method,vmin,vmax):
    path = '/media/wtmgws6_data/experiments/15.10.16_12:51_improved_cross_validation/'

    output_labels = dc.getOutputLabels()
    #ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
    #ax.set_xticklabels(label_sr)

    matplotlib.rcParams.update({'font.size': 10})
    #fig = plt.figure(1)
    fig, ax = plt.subplots()

    folds = 10

    instances = 5
    param_leak = np.arange(0.1, 0.91, 0.1)
    param_sr = np.arange(0.25, 10.01, 0.25)

    average_mean_summed_squared_errors = []
    data_mask = []
    instance = 0

    average_mse = []

    for leak in param_leak:

        average_mse_for_leak = []
        for sr in param_sr:
            data = pickle.load(open(path+"amse_"+str(instance)+"_"+str(leak)+"_"+str(sr)+".pkl","rb"))
            average_mean_summed_squared_errors.append(data)
            average_mse_for_leak.append( np.mean(data))
        average_mse.append(average_mse_for_leak)

            #average_mean_summed_squared_errors.append(empty)

    average_mse = np.array(average_mse)
    print average_mse.shape

    average_mean_summed_squared_errors = np.array(average_mean_summed_squared_errors).T
    best_parameters = []
    max_min_error = 0
    for label in average_mean_summed_squared_errors:
        min_error = 1
        min_error_index = -1
        #print label.shape
        #print label
        #sys.exit()
        for idx_error,error in enumerate(label):
            if error < min_error:
                min_error = error
                min_error_index = idx_error
        line = [0]*360
        #print "min_error_index",min_error_index
        #print "min_error",min_error
        if min_error > max_min_error:
            max_min_error = min_error
        line[min_error_index] = 1
        best_parameters.append(line)

    #print "max_min_error",max_min_error
    #im = plt.imshow(best_parameters,cmap = cm.Greys_r, interpolation='none')

    labels=np.arange(0.1,0.91,0.1)
    for leak_idx,leak in enumerate(average_mse):

        plt.plot(leak, label="LR="+str(labels[leak_idx]),color=get_color(leak_idx,9))

    x_scale = 2
    label_sr = np.arange(0.25, 10.01, 0.25*x_scale)

    ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))
    ax.set_xticklabels(label_sr)
    plt.ylabel('Word Error Rate')
    plt.xlabel('Spectral Radius')
    #plt.yticks(np.arange(0,1080,len(output_labels)))
    #ax.set_yticklabels(np.arange(0,40,1))

    #plt.yticks(np.arange(0,1080,1))
    #ax.set_yticklabels(np.tile(np.arange(0,len(output_labels),1),40))
    #ax2 = ax.twinx()

    #pp = PdfPages(path+'cross_val_chunks_plot.pdf')
    #plt.savefig(pp,format='pdf',bbox_inches='tight')
    #pp.close()

    #fig.show()
    #plt.show()

    print "exit"



    lgd=ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.savefig(path+'cross_val_chunks_plot.png',bbox_extra_artists=(lgd,),bbox_inches='tight')
    plt.show()
    #0.4
    #3.75

#ESANN experiment
#plot the test performance for network size 100 ... 10000
def plot_exp_12():

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    #print content
    #print content[0::9]

    fig = plt.figure(1)
    corpus_size=               [100 , 200 , 500 , 1000, 2000, 5000, 10000, 20000]
    tree_acc_chunks_from_hmm = [43.0, 47.7, 56.5, 59.7, 61.9, 62.6, 63.9, 64.2]
    tree_acc_perfect_chunks =  [49.2, 54.5, 64.8, 68.0, 71.3, 71.8, 73.8, 74.1]

    matplotlib.rcParams.update({'font.size': 22})

    ax = plt.subplot(111)
    graph_labels = ["perfect chunks","chunks from HMM"]
    plt.plot(tree_acc_perfect_chunks,label=graph_labels[0])
    plt.plot(tree_acc_chunks_from_hmm,label=graph_labels[1])

        #plt.setp(lines, color=0, linewidth=2.0)






    plt.ylabel('RCL Tree Accuracy')
    plt.xlabel('Reservoir Size')#
    lgd=ax.legend(graph_labels,loc='lower right', bbox_to_anchor=(1.0, 0.0))
    ax.set_xticklabels(corpus_size)
    #plt.show()

    #fig.savefig('/informatik/isr/wtm/home/twiefel/papers/twiefel_IROS2015/images/'+'exp_1.png',bbox_extra_artists=(lgd,),bbox_inches='tight')
    pp = PdfPages('/informatik/isr/wtm/home/twiefel/papers/esann_2016/images/'+'exp_1.pdf')
    plt.savefig(pp,format='pdf',bbox_extra_artists=(lgd,),bbox_inches='tight')
    pp.close()
    plt.show()

def plot_cross_val(label_x,label_y,label_z,path):

    matplotlib.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots()

    lines = []
    data = dict()
    vals_x = []
    vals_z = []
    with open(path) as f:
        for line in f:
            line_as_list = line.split(";")
            lines.append(line_as_list)
            #colors.add(line_as_list[1])
            line_data = np.mean(np.array([float(value) for value in line_as_list[2:]]))
            if line_as_list[1] not in data:
                data[line_as_list[1]] = [line_data]
            else:
                data[line_as_list[1]].append(line_data)
            vals_z.append(line_as_list[1])
            vals_x.append(line_as_list[0])

    x_scale = 2
    #label_sr = np.arange(0.25, 10.01, 0.25*x_scale)

    #ax.set_xticks(np.arange(0,len(label_sr)*x_scale,x_scale))


    vals_x = sorted(list(set(vals_x)))
    labels_x = vals_x[::x_scale]
    vals_z = sorted(list(set(vals_z)))
    for idx_val_z,val_z in enumerate(vals_z):
        plt.plot(data[val_z], label="LR="+val_z,color=get_color(idx_val_z,len(vals_z)))

    ax.set_xticks(np.arange(0,len(labels_x)*x_scale,x_scale))
    ax.set_xticklabels(labels_x)
    plt.ylabel(label_y)
    plt.xlabel(label_x)



    lgd=ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.savefig(path+'.png',bbox_extra_artists=(lgd,),bbox_inches='tight')
    print "plot saved."
    #plt.show()
    #0.4
    #3.75

def plot_confusion_matrix(confusion_matrix,labels,path):
    print "printing"
    print confusion_matrix
    conf_arr = confusion_matrix
    target_names = np.array(labels, dtype='|S10')

    #norm_conf = []
    #for i in conf_arr:
    #    a = 0
    #    tmp_arr = []
    #    a = sum(i, 0)
    #    for j in i:
    #        tmp_arr.append(float(j)/float(a))
    #    norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(conf_arr), cmap=plt.cm.Greys,
                    interpolation='nearest')

    width = len(conf_arr)
    height = len(conf_arr[0])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    for x in xrange(width):
        for y in xrange(height):
            if (conf_arr[x][y] != 0):
                if (x==y):
                    ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center',color="white")
                else:
                    ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center',color="black")

    #cb = fig.colorbar(res)
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.savefig(path+'_confusion_matrix.png', format='png')
    #plt.show()

if __name__ == "__main__":
    #plot_exp_1()
    #plot_exp_2('bl',0,100)
    #plot_exp_2('cbl',0,100)
    """
    plot_exp_3(
    '/media/wtmgws3_data/experiments/15.02.19_12:09/results.csv',
    '/media/wtmgws3_data/experiments/15.02.19_21:54/results.csv',
    '/media/wtmgws3_data/experiments/15.02.22_01:43_0_0/results.csv',
    '/media/wtmgws3_data/experiments/15.02.22_01:44_1_0/results.csv',
    '/media/wtmgws3_data/experiments/15.02.23_11:03_2_0/results.csv',
    'exp_3_bl'
    )
    """
    """
    plot_exp_3_all(
    '/media/wtmgws3_data/experiments/15.02.19_12:09/results.csv',
    '/media/wtmgws3_data/experiments/15.02.19_21:54/results.csv',
    '/media/wtmgws3_data/experiments/15.02.22_01:43_0_0/results.csv',
    '/media/wtmgws3_data/experiments/15.02.22_01:44_1_0/results.csv',
    '/media/wtmgws3_data/experiments/15.02.23_11:03_2_0/results.csv',
    '/media/wtmgws3_data/experiments/15.02.19_12:11/results.csv',
    '/media/wtmgws3_data/experiments/15.02.19_20:41/results.csv',
    '/media/wtmgws2_data/experiments/15.02.22_22:15_0_1/results.csv',
    '/media/wtmgws3_data/experiments/15.02.25_22:17_1_1/results.csv',
    #'/media/wtmgws2_data/experiments/15.02.22_22:08_1_1/results.csv',
    '/media/wtmgws2_data/experiments/15.02.22_01:08_2_1/results.csv',
    )
    """
    """
    plot_exp_3(
    '/media/wtmgws3_data/experiments/15.02.19_12:11/results.csv',
    '/media/wtmgws3_data/experiments/15.02.19_20:41/results.csv',
    '/media/wtmgws2_data/experiments/15.02.22_22:15_0_1/results.csv',
    '/media/wtmgws2_data/experiments/15.02.22_22:08_1_1/results.csv',
    '/media/wtmgws2_data/experiments/15.02.22_01:08_2_1/results.csv',
    'exp_3_cbl'
    )
    """
    #for i in range(6):
    #    plot_exp_4_1_fold(i)
    #plot_exp_4()
    """
    plot_exp_5_all(
    '/media/wtmgws3_data/experiments/15.02.22_01:43_0_0/results.csv',
    '/media/wtmgws3_data/experiments/15.02.22_01:44_1_0/results.csv',
    '/media/wtmgws3_data/experiments/15.02.23_11:03_2_0/results.csv',
    '/media/wtmgws2_data/experiments/15.02.22_22:15_0_1/results.csv',
    '/media/wtmgws3_data/experiments/15.02.25_22:17_1_1/results.csv',
    #'/media/wtmgws2_data/experiments/15.02.22_22:08_1_1/results.csv',
    '/media/wtmgws2_data/experiments/15.02.22_01:08_2_1/results.csv'
    )
    """
    """
    plot_exp_5(
    '/media/wtmgws2_data/experiments/15.02.22_22:15_0_1/results.csv',
    '/media/wtmgws2_data/experiments/15.02.22_22:08_1_1/results.csv',
    '/media/wtmgws2_data/experiments/15.02.22_01:08_2_1/results.csv',
    'exp_5_cbl'
    )
    """
    """
    plot_exp_5(
    '/media/wtmgws3_data/experiments/15.02.22_01:43_0_0/results.csv',
    '/media/wtmgws3_data/experiments/15.02.22_01:44_1_0/results.csv',
    '/media/wtmgws3_data/experiments/15.02.23_11:03_2_0/results.csv',
    'exp_5_bl'
    )
    """
    #plot_exp_6()
    #csv_file = '/media/wtmgws6_data/experiments/15.06.03_18:36_cross_val_word2vec_50_n_500/results.csv'
    #name = 'cross_val_word2vec_50_n_500'


    #csv_file = '/media/wtmgws6_data/experiments/15.06.03_23:02_cross_val_word2vec_150_n_500/results.csv'
    #name = 'cross_val_word2vec_150_n_500'
    #csv_file = '/media/wtmgws6_data/experiments/15.06.03_23:18_cross_val_word2vec_200_n_500/results.csv'
    #name = 'cross_val_word2vec_200_n_500'
    #csv_file = '/media/wtmgws6_data/experiments/15.06.08_12:55_cross_val_word2vec_50_n_500/results.csv'
    #name = 'cross_val_word2vec_50_n_500_low_lr'
    #csv_file = '/media/wtmgws6_data/experiments/15.06.08_16:14_cross_val_word2vec_50_n_500/results.csv'
    #name = 'cross_val_word2vec_50_n_500_low_lr_wide_range'

    #csv_file = '/media/wtmgws6_data/experiments/15.06.08_13:53_cross_val_word2vec_50_n_500/results.csv'
    #name = 'cross_val_word2vec_50_n_500_very_low_lr'

    #csv_file = '/media/wtmgws6_data/experiments/15.06.08_13:53_cross_val_word2vec_50_n_500/results.csv'
    #name = 'cross_val_word2vec_50_n_500_very_low_lr'
    #for i in range(6):
    #    plot_exp_7(i,csv_file,'cross_val_word2vec_50_n_500')
    #sys.exit()

#    csv_file = '/media/wtmgws6_data/experiments/15.08.18_16:28_cross_validation_chunks/results.csv'
#    name = 'cross_val_chunks_500'
#    out_folder = '/media/wtmgws6_data/experiments/15.08.18_16:28_cross_validation_chunks'
#    lrs = np.arange(0.1, 0.91, 0.1)
#    srii = np.arange(0.25, 10.1, 0.25)
#    folds = 6
#    sets = 2
#    plot_exp_7(out_folder,csv_file,name,srii,lrs,folds,sets)
#    sys.exit()

    #plot_exp_10('bl',0,1)
    #plot_exp_11('bl',0,1)
    #plot_exp_12()
    #plot_cross_val("Spectral Radius","RMSE","LR",'/media/wtmgws2_data/experiments/16.02.10_16:24_esn_parser_localist/cross_val_tag_tagging.csv')

    #this is for crossval on rmse
    #plot_cross_val("Spectral Radius","RMSE","LR",'/media/wtmgws2_data/experiments/16.02.10_17:08_esn_parser_localist/cross_val_tag_tagging.csv')
    #plot_cross_val("Spectral Radius","RMSE","LR",'/media/wtmgws2_data/experiments/16.02.10_17:08_esn_parser_localist/cross_val_word_tagging.csv')

    #crossval on masked rmse
    #plot_cross_val("Spectral Radius","RMSE","LR",'/media/wtmgws2_data/experiments/16.02.15_16:23_esn_parser_localist/cross_val_tag_tagging.csv')
    #plot_cross_val("Spectral Radius","RMSE","LR",'/media/wtmgws2_data/experiments/16.02.15_16:23_esn_parser_localist/cross_val_word_tagging.csv')

    #crossval on accuracy
    plot_cross_val("Spectral Radius","RMSE","LR",'/media/wtmgws2_data/experiments/16.02.16_17:05_esn_parser_localist/cross_val_tag_tagging.csv')
    plot_cross_val("Spectral Radius","RMSE","LR",'/media/wtmgws2_data/experiments/16.02.16_17:05_esn_parser_localist/cross_val_word_tagging.csv')
    sys.exit()

    #plot_exp_8()
    #exp 1: WER w.r.t. size of corpus and network size
    #'/informatik/isr/wtm/home/twiefel/ros_workspace/scripts/iCub_code_for_Language_Comprehension/iCub_language_src/iCub_language/tests_15_01_28/results.csv'

    #exp 2
    #/media/wtmgws3_data/experiments/15.02.16_11:24
    #N=500
    #SR=0.5-5.0
    #ISS=0.1-2.2
    #LR=0.2-0.8

    #exp 3
    #/media/wtmgws3_data/experiments/15.02.19_12:09
    #method=BL
    #N=1000
    #SR=3-10
    #ISS=1.5
    #LR=0.1-0.8

    #exp 4
    #/media/wtmgws3_data/experiments/15.02.19_12:11
    #method=CBL
    #N=1000
    #SR=3-10
    #ISS=1.5
    #LR=0.1-0.8

    #exp 5
    #/media/wtmgws3_data/experiments/15.02.19_21:54
    #method=BL
    #N=2000
    #SR=3-10
    #ISS=1.5
    #LR=0.1-0.8

    #exp 6
    #/media/wtmgws3_data/experiments/15.02.19_20:41
    #method=CBL
    #N=2000
    #SR=3-10
    #ISS=1.5
    #LR=0.1-0.8

    #exp7
    #N=5000
    #LR=0.1 - 0.3
    #/media/wtmgws3_data/experiments/15.02.20_21:58_0_0
    #/media/wtmgws3_data/experiments/15.02.22_01:43_0_0
    #LR=0.4 - 0.6
    #/media/wtmgws3_data/experiments/15.02.20_23:33_1_0
    #/media/wtmgws3_data/experiments/15.02.22_01:44_1_0
    #LR=0.7 - 0.9
    #/media/wtmgws3_data/experiments/15.02.23_11:03_2_0

    #exp8
    #rs
    #/media/wtmgws3_data/experiments/15.02.24_13:44_r_s


    #exp8
    #method=
    #blub()
    csv_file='/media/wtmgws3_data/experiments/15.02.16_11:24/results.csv'
    df = pd.read_csv(csv_file,sep=';')
    sr = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
    iss = [0.1,0.4,0.7,1,1.3,1.6,1.9,2.2]
    lr = [0.1,0.3,0.5,0.7,0.9,1.1]

    #wer = list(df['Word Error Rate'])[:480]
    wer = list(df['Word Error Rate'])[480:960]

    #sr = np.array(sr)
    #iss = np.array(iss)
    #lr = np.array(lr)
    wer = np.array(wer)
    #iss:8
    #sr:10
    #lr:6
    wer_reshaped = np.reshape(wer,(8,10,6))

    print wer_reshaped.shape
    print wer_reshaped[0]
    plt.imshow(wer_reshaped[4],cmap = cm.Greys_r)
    plt.colorbar()
    plt.show()
    #print wer_reshaped[1,1,1]



    sys.exit()
    sys.exit()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(-5, 10, .25)
    Y = np.arange(-5, 5, .25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)
    Gx, Gy = np.gradient(Z) # gradients with respect to x and y
    G = (Gx**2+Gy**2)**.5  # gradient magnitude
    N = G/G.max()  # normalize 0..1
    surf = ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1,
        facecolors=cm.jet(N),
        linewidth=0, antialiased=False, shade=False)
    plt.show()
    sys.exit()

    myarray = [[[1,2,3],
                [1,2,3]],
                [[4,5,6],
                 [4,5,6],
                 [4,5,6]]]
    myarray = np.array(myarray)
    print np.random.permutation(np.array(myarray))
    #print myarray
    #np.savez("tmp.npz",myarray)
    sent_idx = 0
    sent = "place green pyramid on top of red brick"
    word_idx = 0
    word = "place"
    rcl_labels = ["action",
         "cardinal",
         "color",
         "destination",
         "entity",
         "event",
         "id",
         "indicator",
         "measure",
         "reference-id",
         "relation",
         "sequence",
         "spatial-relation",
         "type"]
    myfile = np.load("tmp.npz")
    #print myfile.files
    states_out_train=myfile['arr_0']
    i = 1
    j = 0
    #l = states_out_train[i].shape[0]
    states_out_train = np.array(states_out_train)
    plot(sent_idx,sent,word_idx,word,states_out_train[0],rcl_labels)
    sent_idx = 0
    word_idx = 1
    word = 'green'
    #plot(sent_idx,sent,word_idx,word,states_out_train[0],rcl_labels)


    sys.exit()
    sentence = states_out_train[i]
    #print sentence
    outputs = sentence[0:11,0:14]
    outputs = np.random.random_sample((10,14))

    #print outputs

    #print outputs.shape
    markers = [".",",","o","v","<",">","1","2","3","4","8","s","p","*"]
    fig = plt.figure(1)
    ax = plt.subplot(111)
    #print outputs
    #print outputs.T
    outsT = outputs.T
    for idx,output in enumerate(outsT):
        ax.plot(output,marker=markers[idx])
    #ax.plot(outputs[0:11,1:2],marker="o")
    ax.set_xticklabels(("_ "+sent+" _").split(" "))
    lgd=ax.legend(rcl_labels,loc='upper left', bbox_to_anchor=(1.0, 1.0))
    #ax.subplots_adjust(right=1)
    print fig.get_default_bbox_extra_artists()
    fig.savefig('evaluation_wer.png',bbox_extra_artists=(lgd,),bbox_inches='tight')

    plt.show()
    print "finished"
    sys.exit()

    x = np.arange(-2*np.pi, 2*np.pi, 0.1)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(x, np.sin(x), label='Sine')
    ax.plot(x, np.cos(x), label='Cosine')
    ax.plot(x, np.arctan(x), label='Inverse tan')
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels,loc='upper left', bbox_to_anchor=(1.0, 1.0))
    ax.grid('on')
    fig.savefig('samplefigure.png', bbox_extra_artists=(lgd,), bbox_inches='tight')



    for idx,label in enumerate(graph_labels):
        marker = 'x'
        if idx >= len(graph_labels)/2:
            marker = 'o'
        print idx
        plt.plot(wers[idx], label='WER '+label, marker = marker)





    plt.xlabel('Corpus Size')
    plt.legend(loc=2)
    ax.set_xticklabels(corpus_size)
    #plt.show()
    plt.savefig('evaluation_wer.png')
    plt.show()
    sys.exit()
    """
    path = 'tests_15_01_30/*'
    graph_labels = []
    wers = []
    sers = []
    files=glob.glob(path)
    files.sort(key=lambda x: os.stat(os.path.join('.', x)).st_mtime)
    for file in files:
        l_label=file.split('_',5)[5]
        l_label=l_label.rsplit('.')[0]
        #print l_label
        graph_labels.append( l_label)
        f=open(file)
        for line in f:
            if 'ERROR RATE WORDS' in line:
                l_wer = f.next().split(' ')[0]
                #print l_wer
                wers.append(l_wer)
            if 'ERROR RATE SENTENCES' in line:
                l_ser = f.next().split(' ')[0]
                #print l_wer
                sers.append(l_ser)


        #f=open(file, 'r')
        #f.readlines()
        #f.close()

    """
    #sys.exit()

    with open('results.csv') as f:
        content = f.readlines()
    #print content
    #print content[0::9]
    idx = 0

    graph_labels = []
    wers = []
    sers = []

    for line in content[1:]:
        if idx >=9:
            idx = 0
        row = line.split(';')
        graph_labels.append(row[0])
        wers.append(row[5])
        sers.append(row[6])


    print graph_labels
    print wers
    print sers

    graph_labels=graph_labels[::9]
    wers=zip(*[iter(wers)]*9)
    sers=zip(*[iter(sers)]*9)
    print graph_labels
    print wers
    print sers

    corpus_size=[10,50,100,250,500,1000,1500,2000,2500]

    fig = plt.figure(1)
    ax = plt.subplot(111)
    for idx,label in enumerate(graph_labels):
        marker = 'x'
        if idx >= len(graph_labels)/2:
            marker = 'o'
        print idx
        plt.plot(wers[idx], label='WER '+label, marker = marker)





    plt.xlabel('Corpus Size')
    plt.legend(loc=2)
    ax.set_xticklabels(corpus_size)
    #plt.show()
    plt.savefig('evaluation_wer.png')
    plt.show()

    fig = plt.figure(2)
    ax = plt.subplot(111)
    for idx,label in enumerate(graph_labels):
        marker = 'x'
        if idx >= len(graph_labels)/2:
            marker = 'o'
        plt.plot(sers[idx], label='SER '+label, marker=marker)





    plt.xlabel('Corpus Size')
    plt.legend(loc=2)
    ax.set_xticklabels(corpus_size)
    #plt.show()
    plt.savefig('evaluation_ser.png')
    plt.show()
    print 'done'



    sys.exit()
    corpus_size=[10,50,100,250,500,1000,1500,2000,2500]
    simple_word_err=[
    0,
    1.522,
    7.831,
    13.715,
    20.042,
    29.292,
    31.011,
    32.32,
    31.803
    ]
    simple_sent_err=[
    0,
    8,
    16,
    30.4,
    41.4,
    57.3,
    53.667,
    54.75,
    55.2
    ]
    multi_word_err=[
    0,
    2.826,
    8.042,
    8.204,
    9.512,
    9.825,
    11.072,
    13.752,
    16.694
    ]
    mult_sent_err=[
    0,
    4,
    12,
    14.8,
    18.6,
    17.9,
    21.933,
    25.85,
    30.12
    ]

    fig = plt.figure(1)
    ax = plt.subplot(111)
    plt.plot(simple_word_err, label='WER 1 OCW')
    plt.plot(simple_sent_err, label='SER 1 OCW')
    plt.plot(multi_word_err, label='WER 6 OCW')
    plt.plot(mult_sent_err, label='SER 6 OCW')
    plt.xlabel('Corpus Size')
    plt.legend(loc=2)
    ax.set_xticklabels(corpus_size)
    #plt.show()
    plt.savefig('evaluation_5000.png')
    plt.show()
    print 'done'
    """
    x = np.arange(-2*np.pi, 2*np.pi, 0.1)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(x, np.sin(x), label='Sine')
    ax.plot(x, np.cos(x), label='Cosine')
    ax.plot(x, np.arctan(x), label='Inverse tan')
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.1))
    ax.grid('on')
    #fig.savefig('samplefigure', bbox_extra_artists=(lgd,), bbox_inches='tight')
    fig.show()
    print "finished"
    #sys.exit()
    """