import nltk

class TextUtility(object):
  def brokenText(self, data) :
    texts = data.str.lower()
    broken_text = [nltk.tokenize.word_tokenize(text) for text in texts]
    return broken_text

