from abc import ABCMeta, abstractmethod


# Abstract representation of a feature
class AbstractFeature(metaclass=ABCMeta):
    # Process the feature on an image (required implementation in subclass)
    @abstractmethod
    def process(self, im):
        pass

    # Unique identifier of feature
    def key(self):
        return self.__class__.__name__

    def __str__(self):
        return self.key()
