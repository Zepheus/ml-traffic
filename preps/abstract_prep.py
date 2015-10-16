from abc import ABCMeta,abstractmethod

class AbstractPrep(metaclass=ABCMeta):

    @abstractmethod
    def process(self,im):
        pass

