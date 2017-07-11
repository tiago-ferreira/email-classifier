class DataToAnalisys(object):

  def __init__(self, percentage_config, X, Y):
    self.__percentage_config = percentage_config
    self.__X = X
    self.__Y = Y

  def trainingData(self):
    return self.__X[0:self.trainingSize()]

  def trainingMarks(self):
    return self.__Y[0:self.trainingSize()]

  def validationData(self):
    return self.__X[self.trainingSize():]

  def validationMarks(self):
    return self.__Y[self.trainingSize():]

  def trainingSize(self):
    return int(self.__percentage_config.training * len(self.__Y))
