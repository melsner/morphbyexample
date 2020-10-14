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

if __name__ == "__main__":
    irish = np.loadtxt("https://raw.githubusercontent.com/unimorph/gle/master/gle", dtype=str, delimiter="\t").tolist()
    english = np.genfromtxt("https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/master/en_gum-ud-train.conllu",
                            dtype=str, delimiter="\t", invalid_raise=False).tolist()

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
