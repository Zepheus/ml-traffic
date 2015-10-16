from abc import ABCMeta, abstractmethod

class AbstractFeature(metaclass=ABCMeta):
    @abstractmethod
    def process(self,im):
        pass