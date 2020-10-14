import h5py
import pickle

class PosUnit(tkeras.constraints.Constraint):
  def __init__(self):
    super(PosUnit, self).__init__()
    self.c1 = tkeras.constraints.NonNeg()
    self.c2 = tkeras.constraints.UnitNorm()

  def __call__(self, ww):
    return self.c2(self.c1(ww))
  
  def get_config(self):
    return {}

def buildEmbedder(data):
    nUnits = 64
    charDims = data.vocab.nChars #can go smaller, but not when the embedding matrix is the identity!

    #embed a sequence
    form = tkeras.layers.Input(shape=(data.inputSize, data.vocab.nChars))
    pos = tkeras.layers.Input(shape=(data.inputSize,))
    #mform = tkeras.layers.Masking()(form)

    embedChar = tkeras.layers.Dense(charDims, activation=None, use_bias=False,
                                    kernel_initializer='identity',
                                    kernel_constraint=PosUnit(),
                                    name="embedChar")
    cEmbeds = embedChar(form)
    embedPos = tkeras.layers.Embedding(data.inputSize + 1, charDims)
    pEmbeds = embedPos(pos)

    #c1 = tkeras.layers.Conv1D(nUnits, 3, padding="same", activation="relu")(cEmbeds)
    #c2 = tkeras.layers.Conv1D(nUnits, 3, padding="same", activation="relu")(c1)

    seq = tkeras.layers.Bidirectional(tkeras.layers.LSTM(nUnits, return_sequences=True))(cEmbeds)
    seq2 = tkeras.layers.Concatenate()([pEmbeds, seq])

    #seq3 = tkeras.layers.Bidirectional(tkeras.layers.LSTM(nUnits, return_sequences=True))(seq)
    #seq4 = tkeras.layers.Concatenate()([pEmbeds, seq3])

    mdl = tkeras.Model(inputs=[form, pos], outputs=[seq2, cEmbeds], name="inputEmbedding")

    mdl.summary()

    return mdl

#https://stackoverflow.com/questions/59663963/how-to-create-two-layers-with-shared-weights-where-one-is-the-transpose-of-the
class TransposedDense(tkeras.layers.Layer):
  def __init__(self, originalLayer):
    super(TransposedDense, self).__init__()
    self.originalLayer = originalLayer
    self.supports_masking = True
  
  def __call__(self, inputs):
    weights = self.originalLayer.weights[0]
    weights = tf.transpose(weights)
    val = tf.linalg.matmul(inputs, weights)
    return val

class ZeroOneAccuracy(tkeras.metrics.Metric):
  def __init__(self):
    super(ZeroOneAccuracy, self).__init__()
    self.correct = self.add_weight(name="correct", initializer="zeros", dtype="int32")
    self.total = self.add_weight(name="total", initializer="zeros", dtype="int32")
  
  def update_state(self, y_true, y_pred, sample_weight=None):
    assert(sample_weight is None) #not bothering to handle right now
    bsize = tf.cast(tf.reduce_sum(tf.ones_like(y_true), axis=0)[0, 0], "int32")
    self.total.assign_add(bsize)
    pred = tf.argmax(y_pred, axis=-1)
    ans = tf.argmax(y_true, axis=-1)
    corrChar = tf.cast((tf.cast(ans, "int32") == tf.cast(pred, "int32")), "int32")
    corrWord = tf.reduce_prod(corrChar, axis=-1)
    nCorrWord = tf.reduce_sum(corrWord)
    self.correct.assign_add(nCorrWord)
  
  def result(self):
    return self.correct / self.total
  
  def reset_states(self):
    self.correct.assign(0)
    self.total.assign(0)

def smooth(prs, alpha=.05):
  unif = .5 * tf.ones_like(prs)
  return alpha * unif + (1 - alpha) * prs

class DecoderCell(tkeras.layers.Layer):
  def __init__(self, nUnits, nChars, nInput, nRef, embedChar, **kwargs):
    super(DecoderCell, self).__init__(**kwargs)
    self.nUnits = nUnits
    self.cell = tkeras.layers.LSTMCell(nUnits)
    self.embedChar = embedChar
    self.transposedChar = TransposedDense(self.embedChar)
    self.nChars = nChars
    self.nInput = nInput
    self.nRef = nRef

    self.state_size = ( (nUnits,), (self.nInput,), (self.nInput,), (self.nChars) )

    self.charEmbedSize = self.embedChar.weights[0].shape[0]
    embSize = 2 * self.nUnits + self.nChars

    self.projectInputEmb = tkeras.layers.Dense(embSize, activation="tanh")
    self.projectInputEmbB = tkeras.layers.Dense(embSize, activation="tanh")

    self.dprojA = tkeras.layers.Dense(embSize, activation="tanh")
    self.dprojB = tkeras.layers.Dense(embSize, activation="tanh")

    self.shiftA = tkeras.layers.Dense(1, activation="sigmoid")
    self.shiftB = tkeras.layers.Dense(1, activation="sigmoid")
    self.choose = tkeras.layers.Dense(1, activation="sigmoid")
    #cx = tkeras.layers.Input(shape=(self.nUnits + 2 * self.charEmbedSize,))
    #c1 = tkeras.layers.Dense(1, activation="sigmoid")(cx)
    #c2 = tkeras.layers.Dropout(.1)(c1)
    #c1 = tkeras.layers.Dense(64, activation="tanh")(cx)
    #c2 = tkeras.layers.Dense(1, activation="sigmoid")(c1)
    #self.choose = tkeras.Model(inputs=cx, outputs=c2)

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    #print("getting initial stt", inputs, batch_size, dtype)
    return ([tf.zeros((batch_size, self.nUnits)), tf.zeros((batch_size, self.nUnits))], 
            tf.zeros((batch_size, self.nInput)), tf.zeros((batch_size, self.nInput * self.nRef)), tf.zeros((batch_size, self.nChars)))
  #  if batch_size is None:
  #    batch_size = tf.shape(inputs)[0]
  #  lstm = tf.zeros((batch_size, self.nUnits))
  #  return (lstm, tf.zeros(batch_size, 10), tf.zeros(batch_size, 10), tf.zeros(batch_size, self.nChars))

  def build(self, input_shape):
    #this is where we build our weights, if we have any
    #print("Behold, the build method is called!", input_shape)
    initEmbSize = 2 * self.nUnits + self.nChars
    self.cell.build((None, input_shape[1] + initEmbSize + initEmbSize + self.charEmbedSize))
    self.projectInputEmb.build((None, initEmbSize))
    self.projectInputEmbB.build((None, initEmbSize))
    self.dprojA.build((None, self.nUnits))
    self.dprojB.build((None, self.nUnits))
    self.shiftA.build((None, self.nUnits + 2 * self.charEmbedSize))
    self.shiftB.build((None, self.nUnits + 2 * self.charEmbedSize))
    self.choose.build((None, self.nUnits + 2 * self.charEmbedSize))
    self.transposedChar.build((None, self.charEmbedSize))
    self.built = True

  def call(self, inputs, states, constants=None):
    return self.__call__(inputs, states, constants=constants)

  def __call__(self, inputs, states, constants=None):
    #print("inp in call", inputs)
    #print("state in call", states)
    (lstmState, attnA, attnB, outputCh) = states
    (charsA, embedsA, charsB, embedsB) = constants

    cvA = tf.matmul(attnA, embedsA)[:, 0, :]
    cvB = tf.matmul(attnB, embedsB)[:, 0, :]

    #print("cell inp !!!", [inputs, cvA, cvB, outputCh])
    cellInputs = tf.concat([inputs, cvA, cvB, outputCh], axis=-1)
    #print("cell input tensor", cellInputs, lstmState)
    cellOutput, newLSTMState = self.cell(cellInputs, lstmState)

    #print("ran lstm cell")

    keysA = self.projectInputEmb(embedsA)
    keysB = self.projectInputEmbB(embedsB)

    #use output to compute attention
    newAttnWtsA = self.attention(cellOutput, keysA, self.dprojA)
    newAttnWtsB = self.attention(cellOutput, keysB, self.dprojB)

    maskA = tf.reduce_sum(charsA, axis=-1)
    maskB = tf.reduce_sum(charsB, axis=-1)
    newAttnWtsA *= maskA
    newAttnWtsB *= maskB

    #print("made initial attn wts")

    shiftedAttnWtsA = self.shift(attnA)[:, None, :]
    shiftedAttnWtsB = self.shift(attnB)[:, None, :]

    shiftedAttnWtsA *= maskA
    shiftedAttnWtsB *= maskB

    #print("made shifted matrices", attnA, shiftedAttnWtsA)
    #print("made shifted matrices", attnB, shiftedAttnWtsB)

    finalAttnWtsA, finalAttnA, shiftA = self.chooseAttn((newAttnWtsA, charsA), (shiftedAttnWtsA, charsA), cellOutput, self.shiftA)
    finalAttnWtsB, finalAttnB, shiftB = self.chooseAttn((newAttnWtsB, charsB), (shiftedAttnWtsB, charsB), cellOutput, self.shiftB)

    #unifAttnA = maskA / tf.reduce_sum(maskA, axis=-1, keepdims=True)
    #finalAttnWtsA = .05 * unifAttnA + .95 * finalAttnWtsA

    #unifAttnB = maskB / tf.reduce_sum(maskB, axis=-1, keepdims=True)
    #finalAttnWtsB = .05 * unifAttnB + .95 * finalAttnWtsB

    #print("survived till here")

    dummy, chosen, choose = self.chooseAttn((finalAttnWtsA, charsA), (finalAttnWtsB, charsB), cellOutput, self.choose, computeWeights=False)
    chosen = chosen[:, 0, :]
    
    embedChosen = self.transposedChar(chosen)
    result = tkeras.layers.Activation("softmax")(embedChosen)

    nextState = (newLSTMState, finalAttnWtsA[:, 0, :], finalAttnWtsB[:, 0, :], result)
    output = ([finalAttnWtsA[:, 0, :], finalAttnWtsB[:, 0, :], newAttnWtsA, newAttnWtsB, shiftedAttnWtsA, shiftedAttnWtsB],
              [shiftA, shiftB, choose],
              result)
    
    return output, nextState
  
  def attention(self, queries, keys, project=None):
    #print("attn: projecting", queries)
    if project is not None:
      queries = project(queries)

    #print("attn: query size", queries, "key size", keys)
    
    wts = tf.matmul(queries[:, None, :], keys, transpose_b=True)
    wts = tf.nn.softmax(wts)

    #print("computed weights, size", wts)
    return wts
  
  def shift(self, attnMatrix):
    return tf.pad(attnMatrix, [[0, 0], [1, 0]])[:, :-1]
  
  def chooseAttn(self, srA, srB, state, project, computeWeights=True):
    (wtsA, valsA) = srA
    (wtsB, valsB) = srB
    #print("choosing attn", wtsA, valsA, "::", wtsB, valsB)
    cvA = tf.matmul(wtsA, valsA)[:, 0, :]
    cvB = tf.matmul(wtsB, valsB)[:, 0, :]
    #print("pr input", state, cvA, cvB)
    prInput = tf.concat([state, cvA, cvB], axis=-1)
    prOutput = project(prInput)[:, :, None]

    #clip choice tensor to allow gradients to propagate--- should be harmless?
    prOutput = smooth(prOutput, alpha=.05)

    #print("linear combination", prOutput, wtsA, wtsB)

    if computeWeights:
      wtsOut = prOutput * wtsA + (1 - prOutput) * wtsB
    else:
      wtsOut = tf.zeros_like(wtsA)
    valsOut = prOutput * cvA[:, None, :] + (1 - prOutput) * cvB[:, None, :]

    #print("chose", wtsOut, valsOut)

    return wtsOut, valsOut, prOutput

class PretrainDecoderCell(DecoderCell):
  def __init__(self, *args):
    super(PretrainDecoderCell, self).__init__(*args)

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    #print("getting initial stt", inputs, batch_size, dtype)
    return ([tf.zeros((batch_size, self.nUnits)), tf.zeros((batch_size, self.nUnits))], 
            tf.zeros((batch_size, self.nInput)), tf.zeros((batch_size, self.nInput * self.nRef)), tf.zeros((batch_size, self.nChars),))

  def build(self, input_shape):
    #this is where we build our weights, if we have any
    #print("Behold, the build method is called!", input_shape)
    initEmbSize = 2 * self.nUnits + self.nChars
    self.cell.build((None, 16 + initEmbSize + initEmbSize + self.charEmbedSize))
    self.projectInputEmb.build((None, initEmbSize))
    self.projectInputEmbB.build((None, initEmbSize))
    self.dprojA.build((None, self.nUnits))
    self.dprojB.build((None, self.nUnits))
    self.shiftA.build((None, self.nUnits + 2 * self.charEmbedSize))
    self.shiftB.build((None, self.nUnits + 2 * self.charEmbedSize))
    self.choose.build((None, self.nUnits + 2 * self.charEmbedSize))
    self.transposedChar.build((None, self.charEmbedSize))
    self.built = True

  def __call__(self, inputs, states, constants=None):
    #print("inp in call", inputs)
    #print("state in call", states)
    (lstmState, attnA, attnB, outputCh) = states
    (charsA, embedsA, charsB, embedsB, policy, useAlignment) = constants

    cvA = tf.matmul(attnA, embedsA)[:, 0, :]
    cvB = tf.matmul(attnB, embedsB)[:, 0, :]

    policy = policy[:, 0, :]

    #print("cell inp !!!", [policy, cvA, cvB, outputCh])
    cellInputs = tf.concat([policy, cvA, cvB, outputCh], axis=-1)
    #print("cell input tensor", cellInputs, lstmState)
    cellOutput, newLSTMState = self.cell(cellInputs, lstmState)

    #print("ran lstm cell")

    keysA = self.projectInputEmb(embedsA)
    keysB = self.projectInputEmbB(embedsB)

    #use output to compute attention
    newAttnWtsA = self.attention(cellOutput, keysA, self.dprojA)
    newAttnWtsB = self.attention(cellOutput, keysB, self.dprojB)

    maskA = tf.reduce_sum(charsA, axis=-1)
    maskB = tf.reduce_sum(charsB, axis=-1)
    newAttnWtsA *= maskA
    newAttnWtsB *= maskB

    #print("made initial attn wts")

    shiftedAttnWtsA = self.shift(attnA)[:, None, :]
    shiftedAttnWtsB = self.shift(attnB)[:, None, :]

    shiftedAttnWtsA *= maskA
    shiftedAttnWtsB *= maskB

    #print("made shifted matrices", attnA, shiftedAttnWtsA)
    #print("made shifted matrices", attnB, shiftedAttnWtsB)

    finalAttnWtsA, finalAttnA, shiftA = self.chooseAttn((newAttnWtsA, charsA), (shiftedAttnWtsA, charsA), cellOutput, self.shiftA)
    finalAttnWtsB, finalAttnB, shiftB = self.chooseAttn((newAttnWtsB, charsB), (shiftedAttnWtsB, charsB), cellOutput, self.shiftB)

    #unifAttnA = maskA / tf.reduce_sum(maskA, axis=-1, keepdims=True)
    #finalAttnWtsA = .05 * unifAttnA + .95 * finalAttnWtsA

    #unifAttnB = maskB / tf.reduce_sum(maskB, axis=-1, keepdims=True)
    #finalAttnWtsB = .05 * unifAttnB + .95 * finalAttnWtsB

    #print("survived till here")

    dummy, chosen, choose = self.chooseAttn((finalAttnWtsA, charsA), (finalAttnWtsB, charsB), cellOutput, self.choose, computeWeights=False)
    alignment = inputs[..., None]
    choose = alignment * useAlignment + choose * (1 - useAlignment)
    chosen = choose * finalAttnA + (1 - choose) * finalAttnB
    chosen = chosen[:, 0, :]

    #print("chose", chosen, "choose", choose)

    embedChosen = self.transposedChar(chosen)
    result = tkeras.layers.Activation("softmax")(embedChosen)

    nextState = (newLSTMState, finalAttnWtsA[:, 0, :], finalAttnWtsB[:, 0, :], result)
    output = ([finalAttnWtsA[:, 0, :], finalAttnWtsB[:, 0, :], newAttnWtsA, newAttnWtsB, shiftedAttnWtsA, shiftedAttnWtsB],
              [shiftA, shiftB, choose],
              result)
    
    return output, nextState

def buildModel(data):
    inpForm = tkeras.layers.Input(shape=(data.inputSize, data.vocab.nChars))
    inpPos = tkeras.layers.Input(shape=(data.inputSize,))
    #inpRefs = [tkeras.layers.Input(shape=(data.inputSize, data.vocab.nChars)) for ri in data.references]
    #inpRefs = [[tkeras.layers.Input(shape=(data.inputSize, data.vocab.nChars)), tkeras.layers.Input(shape=(data.inputSize,))] for ri in data.references]
    inpRef = tkeras.layers.Input(shape=(data.inputSize, data.vocab.nChars))
    inpPol = tkeras.layers.Input(shape=(1,))
    inpAlignment = tkeras.layers.Input(shape=(data.outputSize, 1), name="inpalignment")
    inpUseAlignment = tkeras.layers.Input(shape=(1,), name="inpusealignment")

    embeddedIn, inChars = mdl([inpForm, inpPos])

    embeddedRef, refChars = mdl([inpRef, inpPos])
    #rfs = [mdl([ri, inpPos]) for ri in inpRefs]
    #embeddedRefs = [rf[0] for rf in rfs]
    #refChars = [rf[1] for rf in rfs]

    #flatRef = tkeras.layers.Lambda(lambda xx: tf.concat(xx, axis=1))(refChars)
    #flatEmbeds = tkeras.layers.Lambda(lambda xx: tf.concat(xx, axis=1))(embeddedRefs)

    embedPolicy = tkeras.layers.Embedding(1000, 16)
    polEmb = embedPolicy(inpPol)

    #policy = tkeras.layers.RepeatVector(data.outputSize)(polEmb)

    xdec = PretrainDecoderCell(nUnits, data.vocab.nChars, data.inputSize, len(data.references), embedChar)
    decoder = tkeras.layers.RNN(xdec, return_sequences=True)
    (attns, shifts, result) = decoder(inpAlignment, constants=[inChars, embeddedIn, refChars, embeddedRef, polEmb, inpUseAlignment])

    print("description of sequence output", (attns, shifts, result))

    inflect = tkeras.Model(inputs=[inpForm, inpPos, inpPol, inpRef, inpAlignment, inpUseAlignment], outputs=result)

    embedChar.trainable = False
    xdec.transposedChar.trainable = False

    inflect.compile(loss="categorical_crossentropy",
                    metrics=["categorical_accuracy", ZeroOneAccuracy()],
                    optimizer=tkeras.optimizers.Adam())

    inflect.summary()
    return inflect

def checkpoint(inflect, data, run, trains, merges):
  idName = "%s-%dtr-%dmr" % (run, trains, merges)
  print("Writing checkpoint", idName)
  fileName = "drive/My Drive/model/inflect-%s.h5" % idName
  fh = h5py.File(fileName,'w')
  weight = inflect.get_weights()
  for i in range(len(weight)):
    fh.create_dataset('weight'+str(i),data=weight[i])
  fh.close()

  dataName = "drive/My Drive/model/data-%s.dump" % idName
  with open(dataName, 'wb') as fh:
    pickle.dump(data, fh)

def restore(inflect, run, trains, merges):
  idName = "%s-%dtr-%dmr" % (run, trains, merges)
  fileName = "drive/My Drive/model/inflect-%s.h5" % idName
  fh = h5py.File(fileName,'r')
  weight = inflect.weights

  for ii in range(len(weight)):
    wt = weight[ii]
    wt.assign(fh["weight" + str(ii)])
  fh.close()

  dataName = "drive/My Drive/model/data-%s.dump" % idName
  with open(dataName, 'rb') as fh:
    data = pickle.load(fh)
  
  return data

def trainModel(data, inflect, run, iters=3, nTrain=0, nMerge=0, epochs=10, verbose=1):
  data.generateTrainingData()
  pUseAlignment = data.pUseAlignment
  data.pUseAlignment = 0
  inflect.evaluate(data, steps=1000, verbose=verbose)
  data.pUseAlignment = pUseAlignment
    
  for ii in range(iters):
    inflect.fit(data, epochs=epochs, steps_per_epoch=1000, verbose=verbose)
    pUseAlignment = max(data.pUseAlignment - .05, .75)
    data.pUseAlignment = 0
    inflect.evaluate(data, steps=1000, verbose=verbose)
    data.pUseAlignment = pUseAlignment
    print("Step", ii, "alignment prob", data.pUseAlignment)
    data.generateTrainingData()
    checkpoint(inflect, data, run, nTrain + ii, nMerge)

