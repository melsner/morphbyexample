import sys
import numpy as np
import scipy
from matplotlib import pyplot as plt
from collections import *

import os

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
    dd = sys.argv[1]
    run = sys.argv[2]

    srcLang = None
    targLang = None
    for ff in os.listdir(dd):
        if "train-high" in ff:
            srcLang = os.path.abspath(dd + "/" + ff)
        elif "train-low" in ff:
            targLang = os.path.abspath(dd + "/" + ff)

    print("Source", srcLang, "targ", targLang)

    src = np.loadtxt(srcLang, dtype=str, delimiter="\t").tolist()
    targ = np.loadtxt(targLang, dtype=str, delimiter="\t").tolist()

    charset = set()
    outputSize = 0
    for (lemma, form, feats) in src + targ:
      for ch in form.lower():
        charset.update(ch)
      outputSize = max(outputSize, len(form) + 1)

    print("Output size", outputSize)
    print("Chars", charset)

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
 
