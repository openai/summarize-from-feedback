import time

from typing import Optional


def exact_div(x: int, y: int, msg: Optional[str] = None) -> int:
    message = f"exact_div error, {x} is not evenly divisible by {y}"
    if msg:
        message += ", " + msg
    assert x % y == 0, message
    return x // y


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
