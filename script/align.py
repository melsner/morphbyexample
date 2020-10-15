import sys
import numpy as np
import scipy
from matplotlib import pyplot as plt
from collections import *

import networkx as nx

#Grace's alignment code
cache = {}

##Returns minimum edit distance and all possible alignments for a pair of forms
def EditDistanceWithAlignment(s1, s2, level=0):
    if(len(s1)==0):
        return len(s2), set([
            (tuple(), tuple([(char, False) for char in s2]))
            ])
    if(len(s2)==0):
        return len(s1), set([
            (tuple([(char, False) for char in s1]), tuple())    
            ])
    if(s1, s2) in cache:
        return cache[(s1, s2)]
  
    if(s1[-1]==s2[-1]):
        cost = 0
    else:
        cost = 2

    op1, solutions1 = EditDistanceWithAlignment(s1[:-1], s2, level=level + 1)
    op2, solutions2 = EditDistanceWithAlignment(s1, s2[:-1], level=level + 1)
    op3, solutions3 = EditDistanceWithAlignment(s1[:-1], s2[:-1], level=level + 1)

    op1 += 1
    op2 += 1
    op3 += cost

    solutions = set()
    mincost = min(op1, op2, op3)

    if op1==mincost:
        for (sol1, sol2) in solutions1:
            solutions.add( (sol1 + ((s1[-1], False),), sol2) )
    if op2==mincost:
        for (sol1, sol2) in solutions2:
            solutions.add( (sol1, sol2 + ((s2[-1], False),)) )
    if op3==mincost and cost==0:
        for (sol1, sol2) in solutions3:
            solutions.add( (sol1 + ((s1[-1], True),), sol2 + ((s2[-1], True),)) )
    if op3==mincost and cost>0:
        for (sol1, sol2) in solutions3:
            solutions.add( (sol1 + ((s1[-1], False),), sol2 + ((s2[-1], False),)) )
    cache[(s1, s2)] = (mincost, solutions)
    
    return mincost, solutions

def parseAlt(solution, get="theme"):
  res = []
  for char, alt in solution[0]:
    if alt and get == "theme":
      res.append(char)
    elif get != "theme" and not alt:
      res.append(char)
  return "".join(res)

def segmentAll(forms):
  theme = forms[0]
  for fi in forms[1:]:
    cost, solutions = EditDistanceWithAlignment(theme, fi)
    theme = parseAlt(list(solutions)[0], "theme")
  
  #print("theme of all forms", theme)

  dists = []
  for fi in forms:
    cost, solutions = EditDistanceWithAlignment(fi, theme)
    dists.append(parseAlt(list(solutions)[0], "dist"))
    #print("distinguisher for", fi, theme, "is", dists[-1])

  return dists

def getMicroclasses(data):
    microclasses = defaultdict(list)

    for ii, (verb, forms) in enumerate(data.lemmaToForms.items()):
      sortedForms = [form for (cell, form) in sorted(forms.items())]
      dists = segmentAll(list(sortedForms))
      microclasses[tuple(dists)].append(verb)

    return microclasses

def getMicroclassesByCell(data):
    microclasses = defaultdict(lambda: defaultdict(list))

    for ii, (lemma, forms) in enumerate(data.lemmaToForms.items()):
        for (cell, form) in forms.items():
            dists = segmentAll([lemma.lower(), form.lower()])
            microclasses[cell][tuple(dists)].append(lemma)

    return microclasses

def extractTree(forms):
  table = defaultdict(list)
  for formTab in forms:
    flist = list(formTab.items())
    for ii, (featsI, formI) in enumerate(flist):
      for jj, (featsJ, formJ) in enumerate(flist[ii + 1:]):
        cost, solutions = EditDistanceWithAlignment(formI, formJ)
        table[(featsI, featsJ)].append(cost)

  graph = nx.Graph()
  for (featsI, featsJ), costs in table.items():
    cost = np.mean(costs)
    graph.add_edge(featsI, featsJ, weight=cost)

  spanning = nx.algorithms.tree.branchings.minimum_spanning_arborescence(graph.to_directed())

  return spanning.edges()
