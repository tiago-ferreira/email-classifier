#!-*- coding: utf8 -*-
import numpy as np
import pandas as pd
import nltk
from ReadFile import readEmail
from PercentageConfig import PercentageConfig
from DataToAnalisys import DataToAnalisys
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn import neighbors
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import cross_val_score

class EmailAnalisys(object):

  def fitAndPredict(self, model, data):
      scores = cross_val_score(model, data.trainingData(), data.trainingMarks(), cv=3)
      result = np.mean(scores)
      return result

  def calculateAccuracy(self, result, data):
    hits = (result == data)
    total_hits = 100.0 * sum(hits) / len(data)
    return total_hits


if __name__ == '__main__':

  X, Y = readEmail()
  results = {}
  classifier = EmailAnalisys()
  config = PercentageConfig(0.9,0.1)
  data = DataToAnalisys(config, X, Y)

  resultMultinomial = classifier.fitAndPredict(MultinomialNB(), data)
  results[resultMultinomial] = MultinomialNB()
  resultAdaBoost = classifier.fitAndPredict(AdaBoostClassifier(), data)
  results[resultAdaBoost] = AdaBoostClassifier()
  resultKNeighbors = classifier.fitAndPredict(neighbors.KNeighborsClassifier(), data)
  results[resultKNeighbors] = neighbors.KNeighborsClassifier()
  resultOneVsRest = classifier.fitAndPredict(OneVsRestClassifier(LinearSVC(random_state = 0)), data)
  results[resultOneVsRest] = OneVsRestClassifier(LinearSVC(random_state = 0))
  resultOneVsOne = classifier.fitAndPredict(OneVsOneClassifier(LinearSVC(random_state = 0)), data)
  results[resultOneVsOne] = OneVsOneClassifier(LinearSVC(random_state = 0))

  maximo = max(results)
  vencedor = results[maximo]

  vencedor.fit(data.trainingData(), data.trainingMarks())
  result = vencedor.predict(data.validationData())
  accuracy = classifier.calculateAccuracy(result, data.validationMarks())

  print "Total hits algorithm {0} in real world: {1}".format(vencedor.__class__.__name__, accuracy)