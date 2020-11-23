import sys
import numpy as np
import scipy
from matplotlib import pyplot as plt
from collections import *

import gc
import os
import re

# %tensorflow_version 2.x
import tensorflow as tf
import tensorflow.keras as tkeras

import networkx as nx
import sklearn.neighbors
import guppy

from load_data import *
from data_generators import *
from align import *
from models import *
from train import *
from multilingual import *
from complexity import *
from classify import *

if __name__ == "__main__":
    dd = sys.argv[1]
    run = sys.argv[2]

    srcLang = None
    targLang = None
    targDev = None
    for ff in os.listdir(dd):
        if "train-high" in ff:
            srcLang = os.path.abspath(dd + "/" + ff)
        elif "train-low" in ff:
            targLang = os.path.abspath(dd + "/" + ff)
        elif "-dev" in ff:
            targDev = os.path.abspath(dd + "/" + ff)

    print("Source", srcLang, "targ", targLang, "dev", targDev)

    src = np.loadtxt(srcLang, dtype=str, delimiter="\t").tolist()
    targ = np.loadtxt(targLang, dtype=str, delimiter="\t").tolist()
    dev = np.loadtxt(targDev, dtype=str, delimiter="\t").tolist()

    src = [(lemma, form, feats) for (lemma, form, feats) in src]
    targ = [(lemma, form, feats) for (lemma, form, feats) in targ]
    dev = [(lemma, form, feats) for (lemma, form, feats) in dev]

    charset = set()
    outputSize = 0
    for (lemma, form, feats) in src + targ:
      for ch in form.lower():
        charset.update(ch)
      outputSize = max(outputSize, len(form) + 1)

    print("Output size", outputSize)
    print("Chars", len(charset), charset)

    workingSet = src

    data = GenData(workingSet) #dummy for microclasses

    microclasses = getMicroclassesByCell(data)
    classes5 = {}

    for cell, mcs in microclasses.items():
        mc5 = { mc : members for (mc, members) in mcs.items()
                if len(members) >= 5 }
        classes5[cell] = mc5

        print("Classes for cell:", cell)
        for mclass, members in sorted(mc5.items(), 
                                  key=lambda xx: len(xx[1]), reverse=True):
            print(min(members, key=len), mclass, len(members))
        print()
 
    data = PretrainData(workingSet, classes5, charset=charset, 
                        outputSize=outputSize, balance=True,
                        nInstances=1000, batchSize=1,
                        pUseAlignment=1, sourceIsLemma=True)

    embedSeq, embedChar = buildEmbedder(data)
    inflect = buildModel(data, embedSeq, embedChar)

    run = run + "-targ-"
    currentFiles = os.listdir("data/" + run)

    latest = [0, 0]
    for fi in currentFiles:
        mtch = re.search("%s-(\d+)tr-(\d+)mr" % run, fi)
        if mtch:
            nTrain, nMerge = mtch.groups()
            nTrain, nMerge = int(nTrain), int(nMerge)
            if [nMerge, nTrain] > latest:
                latest = [nMerge, nTrain]

    nMerge, nTrain = latest
    print("Latest target file found was", nTrain, nMerge)

    data = restore(inflect, run, nTrain, nMerge)

    nPolicies = len(data.policyMembers())
    nReferences = len(data.referenceMembers())
    print(nPolicies, "policies", nReferences, "references")

    guesser = buildGuesser(data, inflect)
    guesser.load_weights("data/%s/guesser.h5" % run)

    results = classify(dev, data, inflect, guesser, filterGuess=True)
    acc = 0
    total = 0
    for inst, res in zip(dev, results):
        (lemma, form, fts) = inst
        if form == res:
            acc += 1
        total += 1

    print(acc, "/", total, acc/total)

