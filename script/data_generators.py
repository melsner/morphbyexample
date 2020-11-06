# Commented out IPython magic to ensure Python compatibility.
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

from align import *

class Vocab:
  def __init__(self):
    self.alphaToInd = {}
    self.indToAlpha = {}
    self.nChars = 0
  
  def get(self, ss):
    val = self.alphaToInd.get(ss, None)
    if val is not None:
      return val

    self.alphaToInd[ss] = self.nChars
    self.indToAlpha[self.nChars] = ss
    self.nChars += 1
    return self.nChars

  def decode(self, vec):
    res = []
    for ii in range(vec.shape[0]):
      ind = np.argmax(vec[ii, :])
      if vec[ii, ind] == 0:
        res.append("0")
      else:
        res.append(self.indToAlpha[ind])
    return "".join(res)

  def decodeIndices(self, vec):
    return "".join([self.indToAlpha[ind] for ind in vec])

class GenData(tkeras.utils.Sequence):
  def __init__(self, words):
    self.vocab = Vocab()
    self.vocab.get("$")
    self.words = words
    self.references = None
    self.lemmaToForms = defaultdict(dict)

    for lemma, form, feats in words:
      for ch in form.lower():
        self.vocab.get(ch)
      self.lemmaToForms[lemma][feats] = form.lower()

    fLimit = 1 #fixme
    delete = []
    for lemma, sub in self.lemmaToForms.items():
      if len(sub) < fLimit:
        delete.append(lemma)

    for lemma in delete:
      del self.lemmaToForms[lemma]

    self.lemmas = list(self.lemmaToForms.keys())
    np.random.shuffle(self.lemmas)

    allForms = []
    for sub in self.lemmaToForms.values():
      allForms += list(sub.values())

    self.outputSize = int(np.percentile([len(xx) for xx in allForms], 100)) + 2
    self.inputSize = self.outputSize

    self.policy = np.random.normal(size=(7, 16))

  def setReferences(self, references):
    self.references = references
    for ri in references:
      for ci in ri:
        self.vocab.get(ci)
    
    self.nRefChars = self.inputSize * len(self.references) + 1

  def __len__(self):
    return len(self.lemmas)
  
  def getPolicy(self, targetForm):
    if targetForm.endswith("mos"):
      category = 0
    elif targetForm.endswith("s"):
      category = 3
    else:
      category = 6
    return self.policy[category, :]

  def __getitem__(self, idx):
    #print("Accessing batch", idx)
    lemma = self.lemmas[idx]
    forms = self.lemmaToForms[lemma]
    #fixme
    if np.random.random() < .5:
      lIn = self.getForm(forms, ["3", "SG"])
      lOut = self.getForm(forms, ["2", "SG"])
      policy = self.getPolicy(lOut)
    else:
      lIn = self.getForm(forms, ["3", "SG"])
      lOut = self.getForm(forms, ["1", "PL"])
      policy = self.getPolicy(lOut)
    
    #policy = np.zeros((16,))
    #lIn = forms["in"]
    #lOut = forms["out"]

    positions = np.arange(1, self.inputSize + 1, dtype="int")

    #refReprs = list(zip(
    #    self.matrixizeRefs(self.references, length=self.inputSize, pad=True),
    #    self.indexizeRefs(self.references, length=self.inputSize, tagRef=True, pad=True), 
    #))

    refReprs = self.matrixizeRefs(self.references, length=self.inputSize, pad=True)

    rval = ([self.matrixize(lIn, length=self.inputSize, pad=True),
             positions[None, ...],
             policy[None, ...],
             ] +
             refReprs,
             self.matrixize(lOut, length=self.outputSize, pad=True))
    #print("shapes", [xx.shape for xx in rval[0]], rval[1].shape)
    return rval

  def getForm(self, forms, target):
    for feats, form in forms.items():
      if hasFeats(feats, target):
        return form

  def indexize(self, ss, length=None, pad=False):
    ss = "$%s$" % ss

    if pad:
      ss += "$" * (length - len(ss))

    if length is None:
      length = len(ss)

    mat = np.zeros((1, length,))
    for ii, si in enumerate(ss[:length]):
      mat[0, ii] = self.vocab.get(si)
    return mat

  def matrixize(self, ss, length=None, pad=False):
    ss = "$%s$" % ss

    if length is None:
      length = len(ss)
    
    if pad:
      ss += "$" * (length - len(ss))

    mat = np.zeros((1, length, self.vocab.nChars))
    for ii, si in enumerate(ss[:length]):
      mat[0, ii, self.vocab.get(si)] = 1
    return mat

  def matrixizeRefs(self, refs, length, pad=False):
    return [self.matrixize(ri, length, pad=pad) for ri in refs]

  def indexizeRefs(self, refs, length, tagRef=False, pad=False):
    if not tagRef:
      return [self.indexize(ri, length) for ri in refs]
    else:
      res = []
      for rid, ri in enumerate(self.references):
        ss = "$%s$" % ri
        if pad:
          ss += "$" * (length - len(ss))

        row = np.zeros((1, length))
        for cid, ci in enumerate(ss):
          row[0, cid] = (length * rid + cid + 1)
        res.append(row)
      return res

class BatchAdapter(tkeras.utils.Sequence):
  def __init__(self, underlying, batchSize):
    self.underlying = underlying
    self.batchSize = batchSize
    self.indices = list(range(len(self.underlying)))
    np.random.shuffle(self.indices)

  def __len__(self):
    return len(self.underlying) // self.batchSize
  
  def __getitem__(self, idx):
    inds = self.indices[self.batchSize * idx : self.batchSize * (idx + 1)]
    batch = []
    for ii in inds:
      batch.append(self.underlying[ii])
    
    return self.restructure(batch)
  
  def restructure(self, stuff):
    if isinstance(stuff[0], np.ndarray):
      return np.concatenate(stuff)
    else:
      res = []
      for dim in range(len(stuff[0])):
        res.append(self.restructure([si[dim] for si in stuff]))
      return res

class MicroclassData(GenData):
  def __init__(self, words, microclasses, charset=None, outputSize=None, balance=False, nInstances=200, batchSize=1, copy=None, sourceIsLemma=False):
    super(MicroclassData, self).__init__(words)
    self.microclasses = microclasses
    self.copy = copy
    self.batchSize = batchSize
    self.nInstances = nInstances

    self.balance = balance

    if outputSize is not None:
      self.outputSize = outputSize
      self.inputSize = outputSize

    if charset is not None:
      for ch in charset:
        self.vocab.get(ch)

    if copy is None:
      self.sourceTab = {}
      self.referenceTab = {}
      self.policies = {}
      self.trees = {}

      mcLemmas = set()

      #this configures the model, but the string should not be used
      self.references = [ "NOTUSED" ]

      if not sourceIsLemma:
        singleTree = extractTree(list(self.lemmaToForms.values())[:100])
      else:
        allFeats = set()
        for sub in self.lemmaToForms.values():
          for fi in sub:
            allFeats.add(fi)
        singleTree = [ ("LEMMA", feats) for feats in allFeats ]

      nxt = 0
      for cell, msub in self.microclasses.items():
        for (mc, members) in msub.items():
          mcLemmas.update(members)
          exemplar = list(members)[0]
          exemplarForms = self.lemmaToForms[exemplar]
          self.trees[cell, mc] = singleTree

          for (featsI, featsJ) in self.trees[cell, mc]:
            if featsJ == cell:
              self.sourceTab[mc, featsJ] = featsI
              self.referenceTab[mc, featsJ] = mc, featsJ
              self.policies[mc, featsJ] = nxt
              nxt += 1
      
      self.lemmas = list(mcLemmas)

    else:
      self.lemmas = None #save some memory?
      self.lemmaToForms = copy.lemmaToForms #save some memory?

      self.referenceTab = copy.referenceTab
      self.sourceTab = copy.sourceTab
      self.policies = copy.policies
      #for (mc, cell), pol in copy.policies.items():
      #  if mc in self.microclasses:
      #    self.policies[(mc, cell)] = pol
      self.trees = copy.trees
      self.vocab = copy.vocab
      self.outputSize = copy.outputSize
      self.inputSize = copy.inputSize

    self.generateTrainingData()

    self.mode = "train"

  def policyMembers(self):
    invPol = defaultdict(list)
    for (pi, pid) in self.policies.items():
      invPol[pid].append(pi)
    return invPol

  def referenceMembers(self):
    invRef = defaultdict(list)
    for (ri, rid) in self.referenceTab.items():
      invRef[rid].append(ri)
    return invRef

  def __len__(self):
    return len(self.batchIndices) // self.batchSize

  def supports(self, mc, lemma, cell):
    forms = self.lemmaToForms[lemma]
    src = self.sourceTab.get((mc, cell))
    #print("checking support", mc, lemma, forms, src, src in forms, cell in forms)
    if (src in forms or src == "LEMMA") and cell in forms:
      return True

  def generateTrainingData(self):
    self.batchIndices = []

    for policy, users in self.policyMembers().items():
      toGenerate = []
      for mc, cell in users:
        toGenerate += [(mc, li, cell) for li in self.microclasses.get(cell, {}).get(mc, []) if self.supports(mc, li, cell)]
      
      if self.balance is True:
        nInstances = self.nInstances
      elif type(self.balance) is dict:
        nInstances = self.balance[policy]
      else:
        nInstances = None

      if toGenerate:
        print("getting batch indices for policy", policy, len(users), "users", nInstances, "instances")  
        items = self.getBatchIndices(toGenerate, nInstances)
        self.batchIndices += items
    
    print("Generated", len(self.batchIndices), "items over", len(self.policyMembers()), "policies")

  def getBatchIndices(self, members, nInstances):
    inst = []
    if nInstances is None:
      nInstances = len(members) + 1

    while len(inst) < nInstances:
      np.random.shuffle(members)
      for mc, li, cell in members:
          inst.append((mc, li, cell))

    inst = inst[:nInstances]
    return inst

  def report(self, inflect):
    for cell, msub in self.microclasses.items():
      print(cell, ":", len(msub), "microclasses known")

      for mc, members in sorted(msub.items(), key=lambda xx: len(xx[1]), reverse=True):
        if (mc, cell) not in self.sourceTab:
          continue

        #find an example verb
        example = members[0]

        mcd = { cell : {mc : members} }
        minidat = self.__class__(self.words, mcd, pUseAlignment=0, copy=self)
        minidat.batchIndices.sort(key=lambda xx: (xx[1] != example, np.random.randint(0, 500)))

        preds = inflect.predict(minidat, verbose=False, steps=min(100, len(minidat)))

        #print how many members
        overallAcc = 0
        overallDen = 0
        catAcc = defaultdict(int)
        catDen = defaultdict(int)

        for ii in range(preds.shape[0]):
          pi = self.vocab.decode(preds[ii]).strip("$0")
          dummy, li, ci = minidat.batchIndices[ii]
          ai = self.lemmaToForms[li][ci]
          overallDen += 1
          catDen[ci] += 1
          if pi == ai:
            overallAcc += 1
            catAcc[ci] += 1

        catAcc = { cat : (catAcc[cat] / den) for cat, den in catDen.items() }
        macroAvg = sum(catAcc.values()) / len(catAcc.values())
        print("#%d" % len(members), "overall acc", overallAcc / overallDen, "macroavg", macroAvg)

        #for each form, print the appropriate form, the policy in use, the reference in use, and the model prediction
        form = self.lemmaToForms[example][cell]
        print("{0: <40}".format("%s %s" % (cell, form)), end="\t")
        print()
        pol = self.policies.get((mc, cell), None)
        if pol is None:
          polStr = "[-]"
        else:
         polStr = "[%d]" % pol
        ref = self.referenceTab.get((mc, cell), None)
        if ref == (mc, cell):
          refStr = "[self]"
        elif ref == None:
          refStr = "[-]"
        else:
          ex = self.microclasses[cell][ref[0]][0]
          exA = self.lemmaToForms[ex][ref[1]]
          refStr = "[%s]" % exA
        print("{0: <40}".format("%s %s" % (polStr, refStr)), end="\t")
        print()

        acc = 0
        den = 0
        specific = None
        for ii, (dummy, lx, cx) in enumerate(minidat.batchIndices[:preds.shape[0]]):
          if lx == example and cx == cell:
            specific = self.vocab.decode(preds[ii]).strip("$0")
          if cx == cell:
            answer = self.vocab.decode(preds[ii]).strip("$0")
            if answer == self.lemmaToForms[lx][cx]:
              acc += 1
            den += 1
        if den > 0:
          print("{0: <40}".format("{0} {1}/{2} = {3:.3g}".format(specific, acc, den, acc/den)), end="\t")
        else:
          print("{0: <40}".format("- -/- -"), end="\t")
      print()
      print()

  def __getitem__(self, batchIdx):
    #print("Accessing batch", idx)
    if self.batchSize == 1:
      return self.get(batchIdx)

    ins = np.zeros((self.batchSize, self.inputSize, self.vocab.nChars))
    posns = np.zeros((self.batchSize, self.inputSize))
    policies = np.zeros((self.batchSize, 1))
    #refs = [np.zeros((self.batchSize, self.inputSize, self.vocab.nChars)) for ri in self.references]
    ref = np.zeros((self.batchSize, self.inputSize, self.vocab.nChars))
    outs = np.zeros((self.batchSize, self.outputSize, self.vocab.nChars))

    for ii, idx in enumerate(range(self.batchSize * batchIdx, self.batchSize * (batchIdx + 1))):
      xs, ys = self.get(idx)
      ins[ii] = xs[0][0]
      posns[ii] = xs[1][0]
      policies[ii] = xs[2][0]
      #xrs = xs[3]
      #for rid, ri in enumerate(xrs):
      #  refs[rid][ii] = ri[0]
      ref[ii] = xs[3][0]
    
    return (ins, posns, policies, ref), outs

  def getExemplar(self, refMC, refFeats, omit=None):
    #try:
    #  references = self.microclasses[refMC]
    #except KeyError:
    #  references = self.copy.microclasses[refMC]
    
    #print("getting exemplar for", refMC, refFeats, omit)

    references = []
    try:
      ref = self.referenceMembers()[refMC, refFeats]
    except KeyError:
      ref = self.copy.referenceMembers()[refMC, refFeats]

    for mc, cell in ref:
      try:
        references += self.microclasses[cell][mc]
      except KeyError:
        references += self.copy.microclasses[cell][mc]

    #print("fetched mcs", ref)
    #print("list of references", references)

    if omit in references:
      references.remove(omit)

    assert(len(references) >= 1)
    exemplar = np.random.choice(references)

    forms = self.lemmaToForms[exemplar]
    if refFeats in forms:
      relevantForm = forms[refFeats]
      return relevantForm

    np.random.shuffle(references)
    for exemplar in references:
      forms = self.lemmaToForms[exemplar]
      if refFeats in forms:
        relevantForm = forms[refFeats]
        return relevantForm

    assert(0), "No exemplar for %s in %s" % (refFeats, refMC)

  def get(self, idx):
    mc, lemma, featsJ = self.batchIndices[idx]
    forms = self.lemmaToForms[lemma]

    featsI = self.sourceTab[mc, featsJ]
    refMC, refFeats = self.referenceTab[mc, featsJ]
    if featsI == "LEMMA":
      lIn = lemma
    else:
      lIn = forms[featsI]
    lOut = forms[featsJ]
    policy = self.policies[mc, featsJ]
    
    positions = np.arange(1, self.inputSize + 1, dtype="int")

    relevantForm = self.getExemplar(refMC, refFeats, omit=lemma)

    #print("Instance", lIn, lOut, "policy class", 
    #      self.policies[mc, featsJ], featsI, featsJ, relevantForm)

    rval = ([self.matrixize(lIn, length=self.inputSize, pad=False),
             positions[None, ...],
             policy[None, ...],
             self.matrixize(relevantForm, length=self.inputSize, pad=False)],
             self.matrixize(lOut, length=self.outputSize, pad=True),)
    #print("shapes", [xx.shape for xx in rval[0]], rval[1].shape)
    return rval

  def formIterator(self, policyOnly=False):
    for cell, msub in self.microclasses.items():
      for mc, lemmas in msub.items():
        for li in lemmas:
          forms = self.lemmaToForms[li]
          for cell, form in forms.items():
            if policyOnly and (mc, cell) not in self.policies:
              continue
          
            yield (mc, li, cell, form)

  def classAssignmentData(self):
    nPolicies = len(self.policyMembers())
    nReferences = len(self.referenceMembers())
    polNames = {}
    refNames = {}

    nForms = len(list(self.formIterator(policyOnly=True)))

    xs = np.zeros((nForms, self.inputSize, self.vocab.nChars))
    positions = np.zeros((nForms, self.inputSize))
    yPolicy = np.zeros((nForms, nPolicies))
    yReference = np.zeros((nForms, nReferences))

    for ind, (mc, lemma, cell, form) in enumerate(self.formIterator(policyOnly=True)):
          if (mc, cell) not in self.policies: #root of the infl. tree
            continue

          policy = self.policies[mc, cell]
          reference = self.referenceTab[mc, cell]
          if policy not in polNames:
            polNames[policy] = len(polNames)
          pN = polNames[policy]
          if reference not in refNames:
            refNames[reference] = len(refNames)
          rN = refNames[reference]
          xs[ind] = self.matrixize(form, length=self.inputSize, pad=False)
          positions[ind] = np.arange(1, self.inputSize + 1, dtype="int")
          yPolicy[ind, pN] = 1
          yReference[ind, rN] = 1

    return [xs, positions], [yPolicy, yReference], polNames, refNames

class PretrainData(MicroclassData):
  def __init__(self, *args, **kwargs):
    self.pUseAlignment = kwargs.pop("pUseAlignment")
    super(PretrainData, self).__init__(*args, **kwargs)
  
  def get(self, idx):
    mc, lemma, featsJ = self.batchIndices[idx]
    forms = self.lemmaToForms[lemma]

    featsI = self.sourceTab[mc, featsJ]
    refMC, refFeats = self.referenceTab[mc, featsJ]
    if featsI == "LEMMA":
      lIn = lemma
    else:
      lIn = forms[featsI]
    lOut = forms[featsJ]
    policy = np.array([self.policies[mc, featsJ]])
    
    positions = np.arange(1, self.inputSize + 1, dtype="int")

    relevantForm = self.getExemplar(refMC, refFeats, omit=lemma)

    if np.random.random() < .5:
      cost, alts = EditDistanceWithAlignment(lIn, lOut)
      altI = np.random.randint(len(alts))
      alt = list(alts)[altI]
      altIn, altOut = alt

      bits = np.ones((self.outputSize), dtype="float32")
      for ii, xi in enumerate(altOut):
        bits[ii + 1] = xi[1]
    else:
      cost, alts = EditDistanceWithAlignment(relevantForm, lOut)
      altI = np.random.randint(len(alts))
      alt = list(alts)[altI]
      altIn, altOut = alt

      bits = np.ones((self.outputSize), dtype="float32")
      for ii, xi in enumerate(altOut):
        bits[ii + 1] = 1 - xi[1]
    #print("aligned", lIn, lOut, bits)

    switch = np.array([np.random.random() < self.pUseAlignment], dtype="float32")

    #print("Instance", lIn, lOut, "policy class", 
    #      self.policies[mc, featsJ], featsI, featsJ, relevantForm)

    rval = ([self.matrixize(lIn, length=self.inputSize, pad=False),
             positions[None, ...],
             policy[None, ...],
             self.matrixize(relevantForm, length=self.inputSize, pad=False),
             bits[None, ...],
             switch[None, ...]],
             self.matrixize(lOut, length=self.outputSize, pad=True))
    #print("shapes", [xx.shape for xx in rval[0]], rval[1].shape)
    return rval
