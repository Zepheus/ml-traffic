from abc import ABCMeta, abstractmethod

class AbstractFeature(metaclass=ABCMeta):
    @abstractmethod
    def process(self,im):
        pass

    def key(self):
        return self.__class__.__name__

    def __str__(self):
        return self.key()