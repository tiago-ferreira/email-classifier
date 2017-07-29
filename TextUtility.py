import nltk

class TextUtility(object):


  def broken_text(self, data) :
    texts = data.str.lower()
    broken_text = [nltk.tokenize.word_tokenize(text) for text in texts]
    return broken_text

  def build_dictionary(self, data):
    stopwords = nltk.corpus.stopwords('portuguese')
    stemmer = nltk.stem.RSLPStemmer()
    dictionary = set()
    for value in data:
      validWords = [stemmer.stem(word) for word in value if word not in stopwords and len(word) > 2]
      dictionary.update(validWords)
    return dictionary

  def build_map_words_position(self, data):
    totalOfWords = len(data)
    tupleWords = zip(data, xrange(totalOfWords))
    mapWordsAndPositions = {word:index for word,index in tupleWords}
    return mapWordsAndPositions