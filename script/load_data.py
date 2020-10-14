def hasFeats(fset, targets, exclude=[]):
  feats = fset.split(";")
  #print(fset, targets)
  return all([xx in feats for xx in targets]) and not any([xx in feats for xx in exclude])

def nospaces(row):
  lemma, form, feats = row
  return not " " in form

def fullParadigms(rows):
  #prefilter for missing forms
  lemmaForms = defaultdict(set)
  for lemma, form, feats in rows:
    lemmaForms[lemma].add(feats)

  fpSize = max([len(xx) for xx in lemmaForms.values()])
  print("Paradigm size guessed", fpSize)

  return [(lemma, form, feats) for (lemma, form, feats) in rows if len(lemmaForms[lemma]) == fpSize]  

def readUD(rows, target=[], exclude=[]):
  #read a UD treebank, return a list of (lemma, form, feats) tuples and a frequency count
  res = []
  counts = Counter()

  for row in rows:
    if row[0] == "#":
      continue

    num, word, lemma, pos, subtag, feats, dep, role, x1, x2 = row
    if " " in word or "-" in word:
      #exclude particles
      continue
    
    if word.endswith("'") or word.endswith("’"):
      #exclude phonological dropping of last vowel
      continue

    word = word.lower()
    lemma = lemma.lower()

    feats = "%s;%s" % (pos, feats.replace("|", ";"))
        
    if hasFeats(feats, target, exclude):
      if word not in counts:
        res.append((lemma, word, feats))

      counts[word] += 1
  
  return res, counts

if __name__ == "__main__":
    spanish = np.loadtxt("https://raw.githubusercontent.com/unimorph/spa/master/spa", dtype=str, delimiter="\t").tolist()
    irish = np.loadtxt("https://raw.githubusercontent.com/unimorph/gle/master/gle", dtype=str, delimiter="\t").tolist()
    english = np.genfromtxt("https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/master/en_gum-ud-train.conllu",
                            dtype=str, delimiter="\t", invalid_raise=False).tolist()
    irishCorpus = np.genfromtxt("https://raw.githubusercontent.com/UniversalDependencies/UD_Irish-IDT/master/ga_idt-ud-train.conllu",
                                dtype=str, delimiter="\t", invalid_raise=False).tolist()

    irishWords, irishFreq = readUD(irishCorpus, ["NOUN", "Case=NomAcc"], exclude=["Foreign", "Definite=Def", "Form=Len", "Form=Ecl", "Form=HPref"])
    #print(len(gaelicWords))
    #print(wordFreq.most_common(10))
    print(irishWords[:10])

    englishWords, englishFreq = readUD(english, ["NOUN"], exclude=[])

    presInd = [row for row in spanish if hasFeats(row[-1], ["V", "IND", "PRS"]) and nospaces(row)]
    print(presInd[:5])

    nom = [row for row in irish if hasFeats(row[-1], ["N", "NOM"], exclude=["DEF"]) and nospaces(row)]
    print(nom[:5])

    irishWords = fullParadigms(irishWords)
    nom = fullParadigms(nom)
    englishWords = fullParadigms(englishWords)

    replacements = {"NOUN;Number=Plur":"N;NOM;PL", "NOUN;Number=Sing":"N;NOM;SG"}
    for ii in range(len(englishWords)):
      lemma, word, feats = englishWords[ii]
      englishWords[ii] = lemma, word, replacements[feats]

    print(len(nom), "irish words from unimorph")
    print(len(englishWords), "english words from corpus")
    print(len(irishWords), "irish words from corpus")

    lemmaForms = defaultdict(dict)
    for (lemma, form, feats) in irishWords:
      lemmaForms[lemma][feats] = form
    for li, sub in lemmaForms.items():
      print(li)
      for kk, vv in sub.items():
        print(kk, vv)
      break

    workingSet = nom
    charset = set()
    outputSize = 0
    for (lemma, form, feats) in nom + englishWords:
      for ch in form.lower():
        charset.update(ch)
      outputSize = max(outputSize, len(form) + 1)

    print("Output size", outputSize)
    print("Chars", charset)
