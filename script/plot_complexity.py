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

from assess_complexity import buildInflect

def comparisonPlot(monoStats, biStats, dimensions):
  comparison = {}
  outputName = "-".join(dimensions)

  for mc, sub in monoStats.items():
    for cell, stt in sub.items():
      #print(stt)
      prPol, prRef, prExe, prInfl, nItems = stt
      values = {"policy" : prPol, "reference" : prRef, "exemplar": prExe, "inflection" : prInfl}
      total = 0
      for di in dimensions:
        total += np.abs(values[di])
      total /= nItems

    if mc in stats:
      prPol, prRef, prExe, prInfl, nItems = biStats[mc][cell]
      values = {"policy" : prPol, "reference" : prRef, "exemplar": prExe, "inflection" : prInfl}
      totalX = 0
      for di in dimensions:
        totalX += np.abs(values[di])
      totalX /= nItems
    
      comparison[mc, cell] = (total, totalX)
    else:
      print("nd code for", mc)

  fig, ax = plt.subplots()
  its = list(comparison.items())
  v1s = [xx[1][0] for xx in its]
  v2s = [xx[1][1] for xx in its]
  labels = [xx[0] for xx in its]
  colors = []
  nLabels = []
  for mc, cell in labels:
    ex = data.microclasses[cell][mc][0]
    nLabels.append(ex)
    sharedPolicy = False
    policy = data.policies[mc, cell]
    members = data.policyMembers()[policy]
    for mMc, mCell in members:
      lems = data.microclasses[mCell][mMc]
      if set(lems).intersection(data.langLemmas["src"]):
        sharedPolicy = True
        break
    
    if sharedPolicy:
      colors.append("r")
    else:
      colors.append("b")
    
  plt.scatter(v1s, v2s, c=colors)
  for ii in range(len(labels)):
    plt.text(v1s[ii], v2s[ii], nLabels[ii])
  lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
  ]
  plt.plot([lims[0], lims[1]], [lims[0], lims[1]])
  plt.savefig("figures/%s.png" % outputName)


def srcMicroclass(mc, cell, data):
  lems = data.microclasses[cell][mc]
  return set(lems).intersection(data.langLemmas["src"])

def classComp(stats, dimensions):
  comps = {}
  for mc, sub in stats.items():
    for cell, stt in sub.items():
      total = 0
      #print(stt)
      prPol, prRef, prExe, prInfl, nItems = stt
      values = {"policy" : prPol, "reference" : prRef, "exemplar": prExe, "inflection" : prInfl}
      if any([np.isinf(xx) for xx in values.values()]):
        continue
      
      try:
        if srcMicroclass(mc, cell, data):
          continue
      except:
        continue

      for dim in dimensions:
        total += np.abs(values[dim])
      
      comps[mc, cell] = total / nItems
  
  return comps

if __name__ == "__main__":
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
    print("Latest file found was", nTrain, nMerge)
    inflect = buildInflect()
    data = restore(inflect, "english-x-irish", nTrain, nMerge)

    stats = pickle.load(open("data/english-x-irish/stats.dump", 'rb'))
    irStats = pickle.load(open("data/irish-hybrid/stats.dump", 'rb'))

    comparisonPlot(irStats, stats, ["policy", "reference", "exemplar", "inflection"])

    comparisonPlot(irStats, stats, ["policy"])

    comparisonPlot(irStats, stats, ["reference"])

    comparisonPlot(irStats, stats, ["exemplar"])

    comparisonPlot(irStats, stats, ["inflection"])
    irComp = classComp(irStats, ["policy", "reference", "exemplar", "inflection"])
    for xx in sorted(irComp.items(), key=lambda xx: xx[1], reverse=True)[:10]:
      mc, cell = xx[0]
      lems = data.microclasses[cell][mc]
      rep = sorted(lems)[0]
      print(mc, rep, xx[1])

    print()

    xComp = classComp(stats, ["policy", "reference", "exemplar", "inflection"])
    for xx in sorted(xComp.items(), key=lambda xx: xx[1], reverse=True)[:10]:
      mc, cell = xx[0]
      lems = data.microclasses[cell][mc]
      rep = sorted(lems)[0]
      print(mc, rep, xx[1])

    irSubset = defaultdict(dict)
    for mc, sub in stats.items():
        for cell, stts in sub.items():
            if not srcMicroclass(mc, cell, data):
                irSubset[mc][cell] = stts

    print("Total complexity of Irish", totalComplexity(irStats, ["policy", "reference", "exemplar", "inflection"]))
    print("Total complexity of xIrish", totalComplexity(irSubset, ["policy", "reference", "exemplar", "inflection"]))

    for dim in ["policy", "reference", "exemplar", "inflection"]:
        print("%s complexity of Irish" % dim, totalComplexity(irStats, [dim]))
        print("%s complexity of xIrish" % dim, totalComplexity(irSubset, [dim]))
