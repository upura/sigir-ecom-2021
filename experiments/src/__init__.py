import sys
import time
import datetime
import contextlib
from pathlib import Path


class _Logger:
    def __init__(self, out=sys.stdout):
        self.out = out
        self.fp = None
        self.indent = 0

    def print(self, message: str, *args):
        now = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        if len(args) > 0:
            s = f"{now} | {'----' * self.indent}> {message} {' '.join(map(str, args))}"
        else:
            s = f"{now} | {'----' * self.indent}> {message}"
        print(s, file=sys.stdout)
        if self.fp:
            print(s, file=self.fp, flush=True)


_LOGGER = _Logger()


def set_out(f):
    if isinstance(f, (str, Path)):
        f = open(f, "w", encoding="utf-8")
    _LOGGER.fp = f


@contextlib.contextmanager
def span(message: str, *args):
    _LOGGER.print(message, *args)
    start = time.time()
    _LOGGER.indent += 1
    yield
    _LOGGER.indent -= 1
    elapsed = time.time() - start
    _LOGGER.print(f"* {message} ({elapsed:.2f}s)")


def log(message: str, *args):
    _LOGGER.print(message, *args)
