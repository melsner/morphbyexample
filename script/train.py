

def train(run, data, inflect, resume=None, trainSteps=3, iters=10):
  if resume is None:
    with open("drive/My Drive/model/%s-log.txt" % run, "w") as fh:
      fh.write("Run initiated %s %s %s\n" % (run, data, inflect))
  else:
    with open("drive/My Drive/model/%s-log.txt" % run, "a") as fh:
      fh.write("Run resumed %s %s %s %s\n" % (run, data, inflect, resume))

  if resume is None:
    iter = 1
  else:
    nTrains, nMerges = resume
    data = restore(inflect, run, nTrains, nMerges)
    iter = max(int(nTrains // trainSteps), nMerges)
    print("Starting", run, "at", iter)

  for iter in range(iter, iter + iters):
    print("==================== ITERATION %d ===================" % iter)

    if True:
      print("==================== ITERATION %d ===================" % iter)
      print()
      print()

      trainModel(data, inflect, run, trainSteps, 
                 nTrain=(trainSteps * iter), nMerge=(iter - 1),
                 verbose=2)
      data.report(inflect)

    print("Trained")

    if True:
      if iter <= 3 or len(data.policyMembers()) > 20:
        accTab = accuracyTable(data, inflect)
        prohibit = { kk : (vv < .9) for kk, vv in accTab.items() }

        executed = merges(inflect, data, prohibit=prohibit)
        print("Executed", executed, "merges")
      else:
        consolidate(data, inflect, "policy")

      if iter <= 3 or len(data.referenceMembers()) > 20:
        print("Reference merges")

        accTab = accuracyTable(data, inflect, group="reference")
        prohibit = { kk : (vv < .9) for kk, vv in accTab.items() }

        executed = 100
        while executed > 10:
          executed = merges(inflect, data, group="reference", prohibit=prohibit)
          print("Executed", executed, "merges")

      else:
        consolidate(data, inflect, "reference")

      data.report(inflect)

      checkpoint(inflect, data, run, iter, iter)
      tsnePlot(data, "drive/My Drive/model/%s-policies-%d.png" % (run, iter))
      accTab = accuracyTable(data, inflect)
      colorByPolicy(data, inflect, accuracies=accTab, outfile="drive/My Drive/model/%s-colorplot-%d.png" % (run, iter))

      accTab = accuracyTable(data, inflect, group="reference")
      colorByPolicy(data, inflect, accuracies=accTab, group="reference", outfile="drive/My Drive/model/%s-colorplot-ref-%d.png" % (run, iter))

      data.generateTrainingData()
