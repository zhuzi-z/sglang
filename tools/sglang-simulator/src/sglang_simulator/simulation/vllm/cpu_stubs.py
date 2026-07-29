"""CPU stub classes for simulation.

DummyEvent and DummyStream replace torch.cuda.Event/Stream in CPU mode.
All methods are no-op; query() returns True (simulating instant completion).
"""


class DummyEvent:
    """Replace torch.cuda.Event - all operations are no-ops."""

    def record(self, stream=None):
        pass

    def wait(self, stream=None):
        pass

    def query(self):
        return True

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return 0.0


class DummyStream:
    """Replace torch.cuda.Stream - all operations are no-ops."""

    def record_event(self, event=None):
        return DummyEvent()

    def wait_event(self, event):
        pass

    def wait_stream(self, stream):
        pass

    def synchronize(self):
        pass

    def query(self):
        return True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
