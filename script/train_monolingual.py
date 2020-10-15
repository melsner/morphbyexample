import sys
import numpy as np
import scipy
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

if __name__ == "__main__":
    irish = np.loadtxt("https://raw.githubusercontent.com/unimorph/gle/master/gle", dtype=str, delimiter="\t").tolist()
    english = np.genfromtxt("https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/master/en_gum-ud-train.conllu",
                            dtype=str, delimiter="\t", invalid_raise=False).tolist()

    englishWords, englishFreq = readUD(english, ["NOUN"], exclude=[])

    def badRow(row):
        lrow = [xx.lower() for xx in row]
        for gr in ["present", "singular", "plural"]:
            if gr in lrow:
                return True

    irish = [row for row in irish if not badRow(row)]

    nom = [row for row in irish if hasFeats(row[-1], ["N", "NOM"], exclude=["DEF"]) and nospaces(row)]
    nom = fullParadigms(nom)

    charset = set()
    outputSize = 0
    for (lemma, form, feats) in nom + englishWords:
      for ch in form.lower():
        charset.update(ch)
      outputSize = max(outputSize, len(form) + 1)

    print("Output size", outputSize)
    print("Chars", charset)

    workingSet = nom

    data = GenData(workingSet) #dummy for microclasses

    microclasses = getMicroclasses(data)
    classes5 = { mc : members for (mc, members) in microclasses.items()
                 if len(members) >= 5 }

    print(len(microclasses), "microclasses found")
    for mclass, members in sorted(classes5.items(), 
                                  key=lambda xx: len(xx[1]), reverse=True):
        print(min(members, key=len), mclass, len(members))


    data = PretrainData(workingSet, classes5, charset=charset, 
                        outputSize=outputSize, balance=True,
                        nInstances=1000, batchSize=1,
                        pUseAlignment=1)

    embedSeq, embedChar = buildEmbedder(data)
    inflect = buildModel(data, embedSeq, embedChar)

    trainModel(data, inflect, "irish-hybrid-pre", 5)
