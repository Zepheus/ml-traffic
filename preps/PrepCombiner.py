from preps import AbstractPrep

class PrepCombiner(AbstractPrep):

    def __init__(self,preps):
        self.preps = preps
        self.keyValue = '-'.join([p.key() for p in self.preps])

    def key(self):
        return self.keyValue

    def process(self, im):
        current = im
        for p in self.preps:
            current = p.process(current)
        return current

