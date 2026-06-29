from abc import ABC, abstractmethod


class BaseWorker:
    def __init__(self, name: str, *args, **kwargs):
        self.name = name

    def pause_generation(self):
        pass

    def continue_generation(self):
        pass

    def get_request_stats(self) -> list[dict]:
        pass

    def reset_stats(self):
        """Reset per-request stats for the next benchmark round."""
        pass

    def get_iteration_stats(self) -> list[dict]:
        pass

    def shutdown(self):
        pass

    def flush_cache(self):
        pass


class BaseBenchmarkRunner(ABC):

    @abstractmethod
    def benchmark(self) -> dict:
        pass

    @abstractmethod
    def flush_cache(self):
        pass

    @abstractmethod
    def shutdown(self):
        pass
