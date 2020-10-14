def buildGuesser(data):
    nPolicies = len(data.policyMembers())
    nReferences = len(data.referenceMembers())

    inflectMdl = inflect.get_layer(name="inputEmbedding")
    inflectMdl.trainable = False
    xflat = tkeras.layers.Flatten()(inflectMdl.outputs[0])
    #guessRP1 = tkeras.layers.Dense(128, activation="relu")(xflat)
    #guessRP2 = tkeras.layers.Dense(128, activation="relu")(guessRP1)
    guessP = tkeras.layers.Dense(nPolicies, activation="softmax")(xflat)
    guessR = tkeras.layers.Dense(nReferences, activation="softmax")(xflat)

    guesser = tkeras.Model(inputs=inflectMdl.inputs, outputs=[guessP, guessR])
    guesser.compile(loss="categorical_crossentropy",
                    optimizer="adam",
                    metrics=["categorical_accuracy"])
    guesser.summary()
    return guesser

def exemplarProb(data, frequency, rc, normalize=lambda xx: xx, omit=None):
  num = 0
  den = 0
  for (mc, lemma, cell, form) in data.formIterator(policyOnly=True):
    if form == omit:
      continue

    if data.referenceTab[(mc, cell)] == rc:
      num += normalize(frequency[form])
    
    den += normalize(frequency[form])

  return num / den

def measureComplexity(data, inflect, guess, frequency):
  (forms, posns), (policies, refs), polNames, refNames = data.classAssignmentData()
  print(forms.shape)
  print(posns.shape)
  guessPol, guessRef = guess.predict([forms, posns])

  stats = defaultdict(lambda: defaultdict(lambda: np.zeros((5,))))

  for ind, (mc, lemma, cell, form) in enumerate(data.formIterator(policyOnly=True)):
    prPol = np.log(np.sum(guessPol[ind] * policies[ind]))
    prRef = np.log(np.sum(guessRef[ind] * refs[ind]))
    refClass = data.referenceTab[(mc, cell)]
    prExe = np.log(exemplarProb(data, frequency, refClass, normalize=np.log, omit=form))
    rel = data.getExemplar(refClass[0], refClass[1], omit=lemma)
    pol = np.array([data.policies[mc, cell]])
    alignment = np.zeros((1, data.outputSize,))
    switch = np.array([0,])
    prInfl = inflect.evaluate(x=[
      (forms[ii:ii+1], posns[ii:ii+1], pol, data.matrixize(rel, length=data.inputSize, pad=False), alignment, switch)
      ],
      y=data.matrixize(form, length=data.outputSize, pad=True), verbose=0)
    
    prInfl = prInfl[0]

    print(ind, mc, lemma, cell, form, prPol, prRef, prExe, prInfl)

    stats[mc][cell] += [prPol, prRef, prExe, prInfl, 1]

  return stats

def totalComplexity(stats, dimensions):
  res = 0
  nWords = 0
  for mc, sub in stats.items():
    for cell, stt in sub.items():
      #print(stt)
      prPol, prRef, prExe, prInfl, nItems = stt
      values = {"policy" : prPol, "reference" : prRef, "exemplar": prExe, "inflection" : prInfl}
      if any([np.isinf(xx) for xx in values.values()]):
        continue

      nWords += nItems
      total = 0
      for di in dimensions:
        total += np.abs(values[di])
      res += total
  
  return res / nWords
