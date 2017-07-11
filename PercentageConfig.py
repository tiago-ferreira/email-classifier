class PercentageConfig(object):

  def __init__(self, training, validation):
    self.__training = training
    self.__validation = validation

  @property
  def training(self):
    return self.__training

  @property
  def validation(self):
    return self.__validation