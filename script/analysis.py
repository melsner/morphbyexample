import sys
import os
import numpy as np
import scipy
from matplotlib import pyplot as plt
from collections import *

import tensorflow as tf
import tensorflow.keras as tkeras

from models import *
from merge import *

#inflectGen = tkeras.Model(inputs=[inpForm, inpPos, inpPol, inpRef, inpAlignment, inpUseAlignment], outputs=[attns, shifts, result])

def fzip(arr, labels):
  if hasattr(arr, "numpy"):
    arr = arr.numpy()
  return " ".join(["({} {:<.3g})".format(ci, di) for (ci, di) in zip(labels, arr.tolist())])

def inspect(ii):
  xs, ys = data[ii]

  refs = xs[3:]
  flatRefs = []
  for ri in refs:
    rstr = data.vocab.decode(ri[0])
    flatRefs += rstr

  (attns, shifts, result) = inflectGen.predict(xs)
  finalForm, finalRef, form, ref, shiftForm, shiftRef = attns

  out = data.vocab.decode(result[0])

  for ii in range(finalForm.shape[1]):
    #print([si[0, ii] for si in shifts])
    print("form att", out[ii], "\t", fzip(finalForm[0, ii], data.vocab.decode(xs[0][0])))
  
  print()

  for ii in range(finalRef.shape[1]):
    am = np.argmax(finalRef[0, ii])
    print("ref att", out[ii], "am", am, flatRefs[am], "\t", fzip(finalRef[0, ii], flatRefs))

    #print("combined", chosen[0, ii])
    #print("likely char", np.argmax(embed[0, ii]), data.vocab.indToAlpha[np.argmax(embed[0, ii])], embed[0, ii])
    #print(result[0, ii])
    print()

#print(data.vocab.alphaToInd, data.vocab.indToAlpha)
#print(len(data.vocab.alphaToInd))
#print(embedChar.get_weights()[0].shape)

#x1 = np.zeros((1, data.vocab.nChars,), dtype="float32")
#for ii in range(data.vocab.nChars):
#  x1[:] = 0
#  x1[0, ii] = 1
#  xout = embedChar(x1)
#  print(ii, data.vocab.indToAlpha[ii], xout)
#  xrev = xdec.transposedChar(xout)
#  am = np.argmax(xrev)
#  print(xrev, am, data.vocab.indToAlpha[am])
#  print()

#inspect(0)

def showRef():
  xs, ys = data[ii]
  refs = xs[-1][1]
  print(data.references)
  print(refs)
  print(embedRefChar(refs))

#showRef()

def checkAnswer(ii):
  xs, ys = data[ii]
  pred = inflect.predict(xs)
  #print(pred.shape)
  #print(xs[0].shape, xs[1].shape)
  print(data.vocab.decode(xs[0][0]), "::", data.vocab.decode(pred[0]), data.vocab.decode(ys[0]))

#for ii in range(10):
#  xi = np.random.randint(len(data))
#  checkAnswer(xi)

#inflectGen = tkeras.Model(inputs=[inpForm, inpPos, inpPol, inpRef, inpAlignment, inpUseAlignment], outputs=[attns, shifts, result])

def findPt(pt, data):
  for ii in range(len(data)):
    xs, ys = data[ii]
    ystr = data.vocab.decode(ys[0])
    if ystr.strip("$0") == pt:
      return ii

def showAttn(dmaster, ii):
  #data = MicroclassData(dmaster.words, dmaster.microclasses, batchSize=1, copy=dmaster)
  data = dmaster
  xs, ys = data[ii]
  refs = xs[3]
  flatRefs = data.vocab.decode(refs[0])
  inForm = data.vocab.decode(xs[0][0])
  answer = inflect.predict(xs)
  #print("shape of answer", answer.shape)
  outForm = data.vocab.decode(answer[0])
  preds = inflectGen.predict(xs)
  (attns, shifts, result) = preds
  
  print(inForm, "\t", "".join(flatRefs), data.vocab.decode(ys[0]))
  for ii in range(data.outputSize):
    finalForm, finalRef, form, ref, shiftForm, shiftRef = attns

    print("{:<5.4f}".format(shifts[2][0, ii, 0, 0]), end="\t")
    print("{:<5.4f}".format(shifts[0][0, ii, 0, 0]),
          inForm[np.argmax(finalForm[0, ii])], inForm[np.argmax(form[0, ii])], inForm[np.argmax(shiftForm[0, ii])],
          sep="\t", end="\t")
    print("{:<5.4f}".format(shifts[1][0, ii, 0, 0]),
          flatRefs[np.argmax(finalRef[0, ii])], flatRefs[np.argmax(ref[0, ii])], flatRefs[np.argmax(shiftRef[0, ii])],
          sep="\t", end="\t")
    print("(%s)" % outForm[ii], sep="\t")

#for ii in range(3):
#  showAttn(data, np.random.randint(len(data)))
#  print()

def tsnePlot(data, inflect, outfile=None):
  plm, labels, pids = policyMatrix(data, inflect)

  if plm.shape[0] == 1:
    print("Refusing to make a TSNE plot: only one item")
    return

  from sklearn.manifold import TSNE

  tsne = TSNE(perplexity=10, init="pca")
  tsneMat = tsne.fit_transform(plm)

  print(tsneMat.shape, "shape of matrix")

  plt.figure(figsize=(10, 7))
  plt.scatter(tsneMat[:, 0], tsneMat[:, 1])
  for ii in range(tsneMat.shape[0]):
    plt.annotate(policyName(labels[ii], short=True), tsneMat[ii, :])
  if outfile:
    plt.savefig(outfile)
    
  plt.show()

#data = restore(inflect, "spanish-pretrained", 4, 4)
#data.pUseAlignment = .9
#trainModel(data, inflect, "foo", 3, 1, 1)

