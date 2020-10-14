def extend(data, newWords, newClasses, language):
  balance = {}
  for pol in data.policyMembers():
    balance[pol] = 1000

  skip = set(data.lemmaToForms.keys()) #skip Irish words with English homonyms
  for lemma, form, feats in newWords:
    for ch in form.lower():
      assert(ch in data.vocab.alphaToInd), "Character %s not found" % ch

    if lemma not in skip:
      data.lemmaToForms[lemma][feats] = form.lower()
      data.words.append((lemma, form, feats))
    else:
      print("SKIPPED!!!", lemma)
  
  mcLemmas = set()
  singleTree = list(data.trees.values())[0]

  nxt = max(data.policies.values()) + 1

  for (mc, members) in newClasses.items():
    mcLemmas.update(members)
    data.trees[mc] = singleTree

    for (featsI, featsJ) in data.trees[mc]:
      data.sourceTab[mc, featsJ] = featsI
      data.referenceTab[mc, featsJ] = mc, featsJ
      data.policies[mc, featsJ] = nxt
      balance[nxt] = 50
      nxt += 1
  
  data.langLemmas = { "src" : list(data.lemmas), language : list(mcLemmas) }
  data.lemmas += [xx for xx in list(mcLemmas) if xx not in skip]
  data.balance = balance

  for cls, members in newClasses.items():
    data.microclasses[cls] = data.microclasses.get(cls, []) + [xx for xx in members if xx not in skip]
    print("members of class", cls, data.microclasses[cls])


def comparisonPlot(monoStats, biStats, dimensions):
  comparison = {}

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
    ex = data.microclasses[mc][0]
    nLabels.append(ex)
    sharedPolicy = False
    policy = data.policies[mc, cell]
    members = data.policyMembers()[policy]
    for mem, xx in members:
      lems = data.microclasses[mem]
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
  plt.show()

def srcMicroclass(mc, data):
  lems = data.microclasses[mc]
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
        if srcMicroclass(mc, data):
          continue
      except:
        continue

      for dim in dimensions:
        total += np.abs(values[dim])
      
      comps[mc] = total / nItems
  
  return comps
