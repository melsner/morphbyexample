import sys
import numpy as np
import scipy
import re
from matplotlib import pyplot as plt
from collections import *

# %tensorflow_version 2.x
import tensorflow as tf
import tensorflow.keras as tkeras

import networkx as nx
import sklearn.neighbors

from load_data import *
from data_generators import *
from align import *
from models import *
from train import *
from multilingual import *
from complexity import *

if __name__ == "__main__":
    irish = np.loadtxt("https://raw.githubusercontent.com/unimorph/gle/master/gle", dtype=str, delimiter="\t").tolist()
    english = np.loadtxt("/home/elsner.14/celex/celex-en-unimorph", dtype=str, delimiter="\t").tolist()

    irishMode = False
    if len(sys.argv) > 1 and sys.argv[1] == "irish":
        irishMode = True
        os.makedirs("data/irish-hybrid-pre", exist_ok=True)
        os.makedirs("data/irish-hybrid", exist_ok=True)

    def badRow(row):
        lrow = [xx.lower() for xx in row]
        for gr in ["present", "singular", "plural"]:
            if gr in lrow:
                return True

    irish = [row for row in irish if not badRow(row)]

    nom = [row for row in irish if hasFeats(row[-1], ["N", "NOM"], exclude=["DEF"]) and nospaces(row)]
    nom = fullParadigms(nom)

    def fixFeats(row):
        return row[0], row[1], row[2].replace("NOM;", "")

    nom = [fixFeats(row) for row in nom]

    charset = set()
    outputSize = 0
    for (lemma, form, feats) in nom + english:
      for ch in form.lower():
        charset.update(ch)
      outputSize = max(outputSize, len(form) + 1)

    print("Output size", outputSize)
    print("Chars", charset)

    if irishMode:
        workingSet = nom
    else:
        workingSet = english

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
                        pUseAlignment=1)

    embedSeq, embedChar = buildEmbedder(data)
    inflect = buildModel(data, embedSeq, embedChar)

    if not irishMode:
        currentFiles = os.listdir("data/english-hybrid-pre")
        if "inflect-english-hybrid-pre-4tr-0mr.h5" in currentFiles:
            print("Loading pretrained model")
            data = restore(inflect, "english-hybrid-pre", 4, 0)
        else:
            trainModel(data, inflect, "english-hybrid-pre", 5, verbose=2)

        currentFiles = os.listdir("data/english-hybrid")
        latest = [0, 0]
        for fi in currentFiles:
            mtch = re.search("english-hybrid-(\d+)tr-(\d+)mr", fi)
            if mtch:
                nTrain, nMerge = mtch.groups()
                nTrain, nMerge = int(nTrain), int(nMerge)
                if [nMerge, nTrain] > latest:
                    latest = [nMerge, nTrain]

        nMerge, nTrain = latest
        print("Latest file found was", nTrain, nMerge)

        if nMerge < 9:
            print("Training")
            if nTrain > 0:
                train("english-hybrid", data, inflect, (nTrain, nMerge))
            else:
                train("english-hybrid", data, inflect)
        else:
            data = restore(inflect, "english-hybrid", nTrain, nMerge)

        print("Done training!")

        nPolicies = len(data.policyMembers())
        nReferences = len(data.referenceMembers())
        print(nPolicies, "policies", nReferences, "references")
        if "guesser.h5" not in currentFiles:
            guesser = buildGuesser(data, inflect)
            trainGuesser(data, guesser)
            guesser.save_weights("data/english-hybrid/guesser.h5")
        else:
            guesser = buildGuesser(data, inflect)
            guesser.load_weights("data/english-hybrid/guesser.h5")

        print("Done with guesser")

        targD = GenData(nom)
        targClasses = getMicroclassesByCell(targD)
        classes5 = {}

        for cell, mcs in targClasses.items():
            mc5 = { mc : members for (mc, members) in mcs.items()
                    if len(members) >= 5 }

            classes5[cell] = mc5

            print("Classes for cell:", cell)
            for mclass, members in sorted(mc5.items(), 
                                          key=lambda xx: len(xx[1]), reverse=True):
                print(min(members, key=len), mclass, len(members))
            print()

        extend(data, nom, classes5, "target")

        data.generateTrainingData()

        currentFiles = os.listdir("data/english-x-irish")

        latest = [0, 0]
        for fi in currentFiles:
            mtch = re.search("english-x-irish-(\d+)tr-(\d+)mr", fi)
            if mtch:
                nTrain, nMerge = mtch.groups()
                nTrain, nMerge = int(nTrain), int(nMerge)
                if [nMerge, nTrain] > latest:
                    latest = [nMerge, nTrain]

        nMerge, nTrain = latest
        print("Latest target file found was", nTrain, nMerge)

        if nMerge < 9:
            print("Training")
            if nTrain > 0:
                train("english-x-irish", data, inflect, (nTrain, nMerge))
            else:
                train("english-x-irish", data, inflect)
        else:
            data = restore(inflect, "english-x-irish", nTrain, nMerge)

        print("Done training")

        nPolicies = len(data.policyMembers())
        nReferences = len(data.referenceMembers())
        print(nPolicies, "policies", nReferences, "references")
        if "guesser.h5" not in currentFiles:
            guesser = buildGuesser(data, inflect)
            trainGuesser(data, guesser)
            guesser.save_weights("data/english-x-irish/guesser.h5")
        else:
            guesser = buildGuesser(data, inflect)
            guesser.load_weights("data/english-x-irish/guesser.h5")

        print("Done with guesser")

    else:
        print("IRISH mode")
        currentFiles = os.listdir("data/irish-hybrid-pre")
        if "inflect-irish-hybrid-pre-4tr-0mr.h5" in currentFiles:
            print("Loading pretrained model")
            data = restore(inflect, "irish-hybrid-pre", 4, 0)
        else:
            trainModel(data, inflect, "irish-hybrid-pre", 5, verbose=2)

        currentFiles = os.listdir("data/irish-hybrid")
        latest = [0, 0]
        for fi in currentFiles:
            mtch = re.search("irish-hybrid-(\d+)tr-(\d+)mr", fi)
            if mtch:
                nTrain, nMerge = mtch.groups()
                nTrain, nMerge = int(nTrain), int(nMerge)
                if [nMerge, nTrain] > latest:
                    latest = [nMerge, nTrain]

        nMerge, nTrain = latest
        print("Latest file found was", nTrain, nMerge)

        if nMerge < 9:
            print("Training")
            if nTrain > 0:
                train("irish-hybrid", data, inflect, (nTrain, nMerge))
            else:
                train("irish-hybrid", data, inflect)
        else:
            data = restore(inflect, "irish-hybrid", nTrain, nMerge)

        nPolicies = len(data.policyMembers())
        nReferences = len(data.referenceMembers())
        print(nPolicies, "policies", nReferences, "references")
        if "guesser.h5" not in currentFiles:
            guesser = buildGuesser(data, inflect)
            trainGuesser(data, guesser)
            guesser.save_weights("data/irish-hybrid/guesser.h5")
        else:
            guesser = buildGuesser(data, inflect)
            guesser.load_weights("data/irish-hybrid/guesser.h5")

        print("Done with guesser")
