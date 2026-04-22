from abc import ABC, abstractmethod

class Evaluator(ABC):
    @abstractmethod
    def run(self, trainer):
        pass