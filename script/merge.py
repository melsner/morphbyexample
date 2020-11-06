import sys
from size import getsize
from data_generators import *
import seaborn as sns
import matplotlib

def policyMatrix(data, inflect):
  embedPolicy = inflect.get_layer(name="embedPolicy")
  policies = data.policyMembers()
  plm = np.zeros((len(policies), embedPolicy.output.shape[-1]))
  labels = []
  pids = []
  for ii, (pid, pis) in enumerate(policies.items()):
    labels.append(pis)
    pids.append(pid)
    plm[ii] = embedPolicy(np.array([pid]))

  return plm, labels, pids

def referenceMatrix(data, inflect):
  mdl = inflect.get_layer(name="inputEmbedding")
  refs = data.referenceMembers()
  plm = np.zeros((len(refs), data.inputSize))
  labels = []
  pids = []
  for ii, (rid, users) in enumerate(refs.items()):
    labels.append(rid)
    pids.append(rid)
    mc, cell = rid
    #exemplar = list(data.microclasses[cell][mc])[0]
    #form = data.lemmaToForms[exemplar][cell]
    form = data.getExemplar(mc, cell)
    positions = np.arange(1, data.inputSize + 1, dtype="int")
    embed, cemb = mdl([data.matrixize(form, length=data.inputSize, pad=False),
                positions[None, ...]])
    plm[ii] = np.mean(embed[0], axis=1)

  return plm, labels, pids

def closestPolicies(data, inflect):
  plm, labels, pids = policyMatrix(data, inflect)
  nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=plm.shape[0]).fit(plm)
  distances, indices = nbrs.kneighbors(plm)
  return indices[:, 1:], labels, pids

def closestReferences(data, inflect):
  rlm, labels, rids = referenceMatrix(data, inflect)
  nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=rlm.shape[0]).fit(rlm)
  distances, indices = nbrs.kneighbors(rlm)

  for ii in range(indices.shape[0]):
    row = indices[ii, :].tolist()
    rowCell = rids[ii][1]
    rowWithCell = [(rids[num][1], num) for num in row]
    
    #print(rowCell)
    #print(rowWithCell)

    rowWithCell.sort(key=lambda xx: xx[0] != rowCell)
    
    #print(rowWithCell)
    #assert(0)
    
    indices[ii] = [num for (cellVal, num) in rowWithCell]

  return indices[:, 1:], labels, rids

def policyName(policyMembers, short=False):
  if short:
    return policyMembers[0][1]
  return "/".join([lab[1] for lab in policyMembers])

def referenceName(members, short=False):
  return members[1]

def policyMove(inflect, data, mcCell, label, group="policy", nSamples=200):
  dummyClasses = {}
  pm2 = PretrainData(data.words, dummyClasses, pUseAlignment=0, balance=True, copy=data, nInstances=nSamples)
  if group == "policy":
    currentLabelUsers = [(mc, cell) for (mc, cell) in data.policyMembers()[label]]
    if not currentLabelUsers:
      return False
    pm2.policies = dict(pm2.policies.items())

    pm2.microclasses = {}
    for cell, msub in data.microclasses.items():
      mcNew = {}
      for mc, mems in msub.items():
        if (mc, cell) in currentLabelUsers or (mc, cell) == mcCell:
          mcNew[mc] = mems
      if mcNew:
        pm2.microclasses[cell] = mcNew

    pm2.generateTrainingData()
    pm2.policies[mcCell] = label
  else:
    currentLabelUsers = [(mc, cell) for (mc, cell) in data.referenceMembers()[label]]
    currentLabelMCs = [mc for (mc, cell) in currentLabelUsers]
    if not currentLabelUsers:
      return False
    pm2.policies = dict(pm2.policies.items())
    pm2.referenceTab = dict(pm2.referenceTab.items())

    pm2.microclasses = {}
    for cell, msub in data.microclasses.items():
      mcNew = {}
      for mc, mems in msub.items():
        if (mc, cell) in currentLabelUsers or (mc, cell) == mcCell:
          mcNew[mc] = mems
      if mcNew:
        pm2.microclasses[cell] = mcNew

    for li in currentLabelUsers:
      pm2.policies[li] = 1
    pm2.policies[mcCell] = 2
    pm2.generateTrainingData()
    for li in currentLabelUsers:
      pm2.policies[li] = data.policies[li]
    pm2.policies[mcCell] = data.policies[mcCell]
    pm2.referenceTab[mcCell] = label

  llMoved, accMoved, zeroOneMoved = inflect.evaluate(pm2, verbose=0)

  print(len(pm2), "instances with z1s", zeroOneMoved)
  #print("merged class is", pm2.mergedClass)

  if zeroOneMoved > .9:
    print("Executing", mcCell, "to", label, zeroOneMoved)
    if group == "policy":
      data.policies[mcCell] = label
    else:
      data.referenceTab[mcCell] = label
    return True
  else:
    print("No merge", mcCell, "to", label, zeroOneMoved)

    return False

def policyMerge(inflect, data, label1, label2, group="policy", nSamples=200):
  dummyClasses = {}
  pm2 = PretrainData(data.words, dummyClasses, pUseAlignment=0, balance=True, copy=data, nInstances=nSamples)
  #print("size of pm2", getsize(pm2), "size of main data", getsize(data), "size of infl", getsize(inflect))
  #print("detailed anatomy of pm2")
  #print("\tindices:", getsize(pm2.batchIndices))
  #print("\tclasses:", getsize(pm2.microclasses))
  #print("\tsources, policies and references:", getsize(pm2.sourceTab), getsize(pm2.policies), getsize(pm2.referenceTab))
  #print("\ttrees:", getsize(pm2.trees))
  #print("\tlemmas:", getsize(pm2.lemmas))
  #print("\tlemmaToForms:", getsize(pm2.lemmaToForms))
  #print("\tcopy:", getsize(pm2.copy))
  #print()
  
  if group == "policy":
    lu1 = [(mc, cell) for (mc, cell) in data.policyMembers()[label1]]
    lu2 = [(mc, cell) for (mc, cell) in data.policyMembers()[label2]]
    if not lu1 or not lu2:
      return False
    micros = [mc for (mc, cell) in lu1 + lu2]
    pm2.policies = dict(pm2.policies.items())
    pm2.microclasses = {}
    for cell, msub in data.microclasses.items():
      mcNew = {}
      for mc, mems in msub.items():
        if (mc, cell) in lu1 + lu2:
          mcNew[mc] = mems
      if mcNew:
        pm2.microclasses[cell] = mcNew

    pm2.generateTrainingData()
    for li in lu2:
      pm2.policies[li] = label1
  else:
    lu1 = [(mc, cell) for (mc, cell) in data.referenceMembers()[label1]]
    lu2 = [(mc, cell) for (mc, cell) in data.referenceMembers()[label2]]
    if not lu1 or not lu2:
      return False
    micros = [mc for (mc, cell) in lu1 + lu2]
    pm2.referenceTab = dict(pm2.referenceTab.items())
    pm2.policies = dict(pm2.policies.items())
    for li in lu1:
      pm2.policies[li] = 1
    for li in lu2:
      pm2.policies[li] = 2

    for cell, msub in data.microclasses.items():
      mcNew = {}
      for mc, mems in msub.items():
        if (mc, cell) in lu1 + lu2:
          mcNew[mc] = mems
      if mcNew:
        pm2.microclasses[cell] = mcNew

    pm2.generateTrainingData()
    for li in lu1:
      pm2.policies[li] = data.policies[li]
    for li in lu2:
      pm2.referenceTab[li] = label1
      pm2.policies[li] = data.policies[li]

  #for item in pm2.batchIndices:
  #  print(item)

  llMoved, accMoved, zeroOneMoved = inflect.evaluate(pm2, verbose=0)

  print(len(pm2), "instances with z1s", zeroOneMoved)
  #print("merged class is", pm2.mergedClass)

  if zeroOneMoved > .9:
    print("Executing", label2, "to", label1, zeroOneMoved)
    if group == "policy":
      for li in lu2:
        data.policies[li] = label1
    else:
      for li in lu2:
        data.referenceTab[li] = label1
    return True
  else:
    print("No merge", label2, "to", label1, zeroOneMoved)

    return False

def merges(inflect, data, prohibit={}, group="policy", tolerance=5):
  executed = 0
  if group == "policy":
    nbrs, labels, pids = closestPolicies(data, inflect)
  else:
    nbrs, labels, pids = closestReferences(data, inflect)
  #print(nbrs)
  for ii, (pidi, li, nbrsi) in enumerate(zip(pids, labels, nbrs)):
    print(ii, "/", len(labels), pidi, li)
    mergeFail = 0

    for nbr in nbrsi:
      print("\t", nbr, labels[nbr])
      if prohibit.get(pidi, False) or prohibit.get(pids[nbr], False):
        print("\tskipped")
        print()
        continue

      ex = policyMerge(inflect, data, pidi, pids[nbr], group=group)
      executed += int(ex)
      if not ex:
        mergeFail += 1
      
      if mergeFail >= tolerance:
        break
    
    print()

  #for pkey, pval in data.policies.items():
  #  print(pkey, "->", pval)

  return executed

def accuracyTable(data, inflect, group="policy"):
  if group == "policy":
    users = data.policyMembers()
  else:
    users = data.referenceMembers()
  
  tab = {}
  for gid, members in users.items():
    targetClasses = {}
    for cell, msub in data.microclasses.items():
      mcNew = {}
      for mc, mems in msub.items():
        if (mc, cell) in members:
          mcNew[mc] = mems
      if mcNew:
        targetClasses[cell] = mcNew

    pmerge = PretrainData(data.words, targetClasses, pUseAlignment=0, balance=True, copy=data, nInstances=100)
    pmerge.batchIndices = [ (mc, lemma, cell) for (mc, lemma, cell) in pmerge.batchIndices if (mc, cell) in members ]
    ll, acc, zeroOne = inflect.evaluate(pmerge, verbose=0)
    tab[gid] = zeroOne
  
  return tab

def consolidate(data, inflect, group="policy"):
  if group == "policy":
    keyTable = data.policies
    memberFn = data.policyMembers
  else:
    keyTable = data.referenceTab
    memberFn = data.referenceMembers
  
  cSizes = {}
  for cluster, mems in memberFn().items():
    cSizes[cluster] = len(mems)
  
  for (mc, cell), assigned in keyTable.items():
    if isinstance(assigned, np.ndarray):
      assigned = assigned[0]
    print("current assignment", mc, cell, assigned)
    csize = cSizes[assigned]
    print("size of that cluster", csize)

    possibleClusters = [cluster for cluster, mems in cSizes.items() if mems >= csize and cluster != assigned]
    memberTable = memberFn()
    possibleClusters = sorted(possibleClusters, key=lambda xx: memberTable[xx], reverse=True)
    for poss in possibleClusters:
      #has this cluster been invalidated?
      currSize = memberFn()
      if poss not in currSize or len(currSize[poss]) < csize:
        continue
      
      #print("XXX recheck curr", keyTable[(mc, cell)], mc, cell, id(keyTable))
      print("testing", poss, "with size", cSizes[poss])
      executed = policyMove(inflect, data, (mc, cell), poss, group=group)
      if executed:
        break

    print()

def fnordcolorByPolicy(data, inflect, group="policy", accuracies=None, outfile=None):
  return
  import seaborn as sns
  if group == "policy":
    nPolicies = len(set(data.policies.values()))
  elif group == "reference":
    nPolicies = len(set(data.referenceTab.values()))
  cm = sns.color_palette("hls", nPolicies)
  #print("n pols", nPolicies, "n colors", len(cm))
  import matplotlib
  cm = matplotlib.colors.ListedColormap(cm)
  nCells = len(list(data.trees.values())[0]) + 1
  nClasses = 0
  for cell, msub in data.microclasses.items():
    nClasses = max(nClasses, len(msub))
  matrix = np.zeros(nClasses, nCells)
  labels = np.zeros_like(matrix, dtype="object")
  #print(matrix.shape)
  mcLabels = []
  polNums = {}
  for row, (mc, members) in enumerate(sorted(data.microclasses.items(), key=lambda xx: len(xx[1]), reverse=True)):
    #find an example verb
    example = members[0]
    mcLabels.append(example)

    for col, (cell, form) in enumerate(sorted(data.lemmaToForms[example].items())):
      if group == "policy":
        pol = data.policies.get((mc, cell), None)
        if pol is not None:
          pol = pol
      else:
        pol = data.referenceTab.get((mc, cell), None)

      if pol is None:
        matrix[row, col] = float('nan')
      elif accuracies is not None and accuracies.get(pol, 0) < .9:
        matrix[row, col] = float('nan')
      else:
        polNums[pol] = polNums.get(pol, len(polNums))
        matrix[row, col] = polNums[pol]
        
      labels[row, col] = form

  plt.figure(figsize=(10, 10))
  plt.pcolormesh(matrix[::-1, :], cmap=cm)
  for ri in range(labels.shape[0]):
    for ci in range(labels.shape[1]):
      plt.text(ci + .1, labels.shape[0] - ri - .9, labels[ri, ci])

  cells = list(sorted(data.lemmaToForms[example].keys()))
  t0 = {targ:src for src, targ in list(data.trees.values())[0]}
  #print(t0)
  for ii in range(len(cells)):
    cells[ii] = "%s->\n%s" % (t0.get(cells[ii], ""), cells[ii])
  plt.xticks(np.arange(len(cells)) + .5, cells)
  plt.yticks(np.arange(matrix.shape[0]) + .5, reversed(mcLabels))

  if outfile:
    plt.savefig(outfile)


def colorByPolicy(data, inflect, group="policy", accuracies=None, outfile=None):
  plt.rcParams.update({'figure.max_open_warning': 0})
  cells = sorted(list(data.microclasses.keys()))

  #figs, axs = plt.subplots(1, len(cells))
  figs = []
  axs = []
  for ii in range(len(cells)):
    fig, ax = plt.subplots()
    figs.append(fig)
    axs.append(ax)

  if group == "policy":
    nPolicies = len(set(data.policies.values()))
  elif group == "reference":
    nPolicies = len(set(data.referenceTab.values()))

  cm = sns.color_palette("hls", nPolicies)
  cm = matplotlib.colors.ListedColormap(cm)

  polNums = {}
  for ii, ci in enumerate(cells):
    drawColumn(data, inflect, ci, axs[ii], cm, accuracies=accuracies, group=group, polNums=polNums)

  if outfile:
    for ii, fi in enumerate(figs):
      fi.set_size_inches(10, 20)
      fi.savefig("%s-%s.png" % (outfile, cells[ii]))

    plt.close('all')

def drawColumn(data, inflect, cell, axis, cm, accuracies, group, polNums):
  nClasses = len(data.microclasses[cell])
  matrix = np.zeros((nClasses, 1))
  labels = np.zeros_like(matrix, dtype="object")
  mcLabels = []
  for row, (mc, members) in enumerate(sorted(data.microclasses[cell].items(), key=lambda xx: len(xx[1]), reverse=True)):
    #find an example verb
    example = members[0]
    mcLabels.append(example)

    form = data.lemmaToForms[example][cell]
    if group == "policy":
      pol = data.policies.get((mc, cell), None)
    else:
      pol = data.referenceTab.get((mc, cell), None)

    if pol is None:
      matrix[row, 0] = float('nan')
    elif accuracies is not None and accuracies.get(pol, 0) < .9:
      matrix[row, 0] = float('nan')
    else:
      polNums[pol] = polNums.get(pol, len(polNums))
      matrix[row, 0] = polNums[pol]
        
    labels[row, 0] = form

  axis.pcolormesh(matrix[::-1, :], cmap=cm)
  for ri in range(labels.shape[0]):
    for ci in range(labels.shape[1]):
      axis.text(ci + .1, labels.shape[0] - ri - .9, labels[ri, ci])

  axis.set_xticks([.5,])
  axis.set_xticklabels([cell])
  axis.set_yticks(np.arange(matrix.shape[0]) + .5)
  axis.set_yticklabels(reversed(mcLabels))

