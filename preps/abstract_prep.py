from abc import ABCMeta,abstractmethod

# abstract representation of an image preprocessor
class AbstractPrep(metaclass=ABCMeta):

    @abstractmethod
    def process(self,im):
        pass

    def key(self):
        return self.__class__.__name__
