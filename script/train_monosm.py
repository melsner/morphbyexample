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

if __name__ == "__main__":
    dd = sys.argv[1]
    run = sys.argv[2]

    os.makedirs("data/%s" % run, exist_ok=True)

    srcLang = dd
    print("Source", srcLang)

    src = np.loadtxt(srcLang, dtype=str, delimiter="\t").tolist()

    src = [(lemma, form, feats) for (lemma, form, feats) in src]

    charset = set()
    outputSize = 0
    for (lemma, form, feats) in src:
      for ch in form:
        charset.update(ch)
      outputSize = max(outputSize, len(form) + 1)

    print("Output size", outputSize)
    print("Chars", len(charset), charset)

    workingSet = src

    data = GenData(workingSet) #dummy for microclasses

    microclasses = getMicroclassesByCell(data)
    classes5 = {}

    for cell, mcs in microclasses.items():
        mc20 = { mc : members for (mc, members) in mcs.items()
                if len(members) >= 20 }
        classes5[cell] = mc20

        print("Classes for cell:", cell)
        for mclass, members in sorted(mc20.items(), 
                                  key=lambda xx: len(xx[1]), reverse=True):
            print(min(members, key=len), mclass, len(members))
        print()
 
    data = PretrainData(workingSet, classes5, charset=charset, 
                        outputSize=outputSize, balance=True,
                        nInstances=1000, batchSize=1,
                        pUseAlignment=1, sourceIsLemma=True)

    embedSeq, embedChar = buildEmbedder(data)
    inflect = buildModel(data, embedSeq, embedChar)

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
    print("Latest file found was", nTrain, nMerge)
    if nTrain < 4 and nMerge == 0:
        print("Pretraining")
        trainModel(data, inflect, run, 5, verbose=2)
    else:
        data = restore(inflect, run, nTrain, nMerge)

    print("Done with pretraining")

    if nMerge < 9:
        print("Training")
        if nTrain > 0:
            train(run, data, inflect, (nTrain, nMerge))
        else:
            train(run, data, inflect)
    else:
        data = restore(inflect, run, nTrain, nMerge)

    print("Done with training")

    nPolicies = len(data.policyMembers())
    nReferences = len(data.referenceMembers())
    print(nPolicies, "policies", nReferences, "references")

    rstats = defaultdict(Counter)
    for (mc, lemma, cell, form) in data.formIterator():
        if (mc, cell) in data.referenceTab:
            ref = data.referenceTab[mc, cell]
            rstats[cell][ref] += 1

    maj = 0
    total = 0
    for ki, sub in rstats.items():
        print(ki)
        for kk, vv in sub.items():
            print("\t", kk, vv)
        maj += sub.most_common(1)[0][1]
        total += sum(sub.values())

    print("majority", maj / total, maj, "/", total)

    if "guesser.h5" not in currentFiles:
        guesser = buildGuesser(data, inflect)
        trainGuesser(data, guesser)
        guesser.save_weights("data/%s/guesser.h5" % run)
    else:
        guesser = buildGuesser(data, inflect)
        guesser.load_weights("data/%s/guesser.h5" % run)
    
    print("Done with guesser")

    frequency = {}
    for (mc, lemma, cell, form) in data.formIterator():
        frequency[form] = 1

    if "stats.dump" not in currentFiles:
        stats = measureComplexity(data, inflect, guesser, frequency)
        pickle.dump(stats, open("data/%s/stats.dump" % run, "wb"))
