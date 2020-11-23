import sys
import os
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
from models import *
from multilingual import *

import h5py
import pickle

def attestedValues(data, feats, language, mode):
  if mode == "policy":
    tab = data.policyMembers()
  else:
    tab = data.referenceMembers()

  res = set()
  for pol, users in tab.items():
    for (mc, cell) in users:
      if cell == feats:
        if ((srcMicroclass(mc, cell, data) and language == "source") or
            (not srcMicroclass(mc, cell, data) and language == "target")):
          res.add(pol)

  return res

def attestedForLanguage(data, language, mode):
  if mode == "policy":
    tab = data.policyMembers()
  else:
    tab = data.referenceMembers()

  res = set()
  for pol, users in tab.items():
    for (mc, cell) in users:
      if ((srcMicroclass(mc, cell, data) and language == "source") or
          (not srcMicroclass(mc, cell, data) and language == "target")):
        print(mc, cell, "attested for language", language, "in mode", mode, "with mem", data.microclasses[cell][mc][0])
        res.add(pol)

  return res

def classify(instances, data, inflect, guess, filterGuess=False, language="target"):
  res = []

  inForms = [form for (lemma, form, feats) in instances]
  xs, ys, polNames, refNames = data.classAssignmentData()
  polInv = { vv:kk for (kk, vv) in polNames.items() }
  refInv = { vv:kk for (kk, vv) in refNames.items() }

  (forms, posns, fts) = data.testAssignmentData(instances)
  guessPol, guessRef = guess.predict([forms, fts])

  if filterGuess:
    languagePolicies = attestedForLanguage(data, language, "policy")
    languageRefs = attestedForLanguage(data, language, "reference")

    maskPol = np.zeros_like(guessPol)
    for ind, (lemma, outForm, feats) in enumerate(instances):
      attested = attestedValues(data, feats, language, "policy")

      print("Checking attestation for", feats, "with value", attested)

      if not attested:
        for ai in languagePolicies:
          maskPol[ind, ai] = 1
      else:
        for ai in attested:
          print("policy", ai, "name", polNames[ai])
          maskPol[ind, polNames[ai]] = 1

    guessPol *= maskPol
  
    maskRef = np.zeros_like(guessRef)
    for ind, (lemma, outForm, feats) in enumerate(instances):
      attested = attestedValues(data, feats, language, "reference")
      if not attested:
        for ai in languageRefs:
          maskRef[ind, ai] = 1
      else:
        for ai in attested:
          print("ref", ai, "name", refNames[ai])
          maskRef[ind, refNames[ai]] = 1
      
    guessRef *= maskRef

  bestPol = np.argmax(guessPol, axis=1)
  bestRef = np.argmax(guessRef, axis=1)

  bestPol = [polInv[xx] for xx in bestPol.astype("int")]
  bestRef = [refInv[xx] for xx in bestRef.astype("int")]

  for ind in range(len(forms)):
      fi = forms[ind:ind + 1]
      pi = posns[ind:ind + 1]
      pol = np.array([bestPol[ind]])
      print("trying to produce", instances[ind], "using source", data.vocab.decode(fi[0]), 
            "and ref class", bestRef[ind])
      rel = data.getExemplar(bestRef[ind][0], bestRef[ind][1])
      alignment = np.zeros((1, data.outputSize,))
      switch = np.array([0,])
      infl = inflect.predict(x=[
      (fi, pi, pol, data.matrixize(rel, length=data.inputSize, pad=False), alignment, switch)
      ])
      rStr = data.vocab.decode(infl[0])
      rStr = rStr.strip("$")
      if rStr == inForms[ind]:
        correct = "correct"
      else:
        correct = "wrong"
      print(inForms[ind], rStr, correct)

      res.append(rStr)

  return res
