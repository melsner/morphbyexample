from models import *
from merge import *
from analysis import *

def train(run, data, inflect, resume=None, doTrain=True, trainSteps=3, iters=10):
  if resume is None:
    iter0 = 1
  else:
    nTrains, nMerges = resume
    data = restore(inflect, run, nTrains, nMerges)
    iter0 = min(int(nTrains // trainSteps), nMerges)
    print("Starting", run, "at", iter0)

  for iter1 in range(iter0, iters):
    print("==================== ITERATION %d ===================" % iter1)

    if doTrain:
      print("==================== ITERATION %d ===================" % iter1)
      print()
      print()

      if resume is not None and iter1 == iter0:
        ntr = max(0, ((nMerges + 1) * trainSteps - nTrains))
        print("Selected", ntr, "steps of training for resumption")
      else:
        ntr = trainSteps

      trainModel(data, inflect, run, ntr, 
                 nTrain=(trainSteps * iter1), nMerge=(iter1 - 1),
                 verbose=2)
      data.report(inflect)

    print("Trained")

    if True:
      if iter1 <= 3 or len(data.policyMembers()) > 20:
        print("Policy merges (%d policies)" % len(data.policyMembers()))
        accTab = accuracyTable(data, inflect)
        prohibit = { kk : (vv < .9) for kk, vv in accTab.items() }

        executed = merges(inflect, data, prohibit=prohibit)
        print("Executed", executed, "merges")
      else:
        print("Consolidating policies")
        consolidate(data, inflect, "policy")

      if iter1 <= 3 or len(data.referenceMembers()) > 20:
        print("Reference merges (%d references)" % len(data.referenceMembers()))

        accTab = accuracyTable(data, inflect, group="reference")
        prohibit = { kk : (vv < .9) for kk, vv in accTab.items() }

        executed = merges(inflect, data, group="reference", prohibit=prohibit)
        print("Executed", executed, "merges")
      else:
        print("Consolidating references")
        consolidate(data, inflect, "reference")

      data.report(inflect)

      checkpoint(inflect, data, run, (trainSteps * iter1), iter1)
      tsnePlot(data, inflect, "data/%s/%s-policies-%d.png" % (run, run, iter1))
      accTab = accuracyTable(data, inflect)
      colorByPolicy(data, inflect, accuracies=accTab, outfile="data/%s/%s-colorplot-%d" % (run, run, iter1))

      accTab = accuracyTable(data, inflect, group="reference")
      colorByPolicy(data, inflect, accuracies=accTab, group="reference", outfile="data/%s/%s-colorplot-ref-%d" % (run, run, iter1))

      data.generateTrainingData()

    print("Training procedure terminating normally")
