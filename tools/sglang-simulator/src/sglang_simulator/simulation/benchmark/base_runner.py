from abc import ABC, abstractmethod


class BaseWorker:
    def __init__(self, name: str, *args, **kwargs):
        self.name = name


class BaseBenchmarkRunner(ABC):
    def __init__(self, workers: list[BaseWorker]):
        self.workers = workers

    @abstractmethod
    def benchmark(self) -> dict:
        pass

    @abstractmethod
    def flush_cache(self):
        pass

    @abstractmethod
    def shutdown(self):
        pass
