from abc import ABCMeta, abstractmethod

class AbstractVisual(metaclass=ABCMeta):

    @abstractmethod
    def show(self,labels,data,saveName=""):
        pass
