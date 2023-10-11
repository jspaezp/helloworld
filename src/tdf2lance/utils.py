import random
from dataclasses import dataclass
from time import time
from typing import Callable

import numpy as np
from loguru import logger


@dataclass
class StreamingMeanCalculator:
    """Calculate the mean of a stream of values.

    This class is used to calculate the mean of a stream of values.
    It is useful to calculate the mean of a stream of values without
    storing all the values in memory.

    Parameters
    ----------
    cumsum : float
        The running sum of the values.
    n : int
        The number of values added to the running sum.

    Examples
    --------
        >>> from tdf2lance.utils import StreamingMeanCalculator
        >>> mean_calculator = StreamingMeanCalculator()
        >>> for i in range(100):
        ...     mean_calculator.add(i)
        >>> mean_calculator.mean()
        49.5
    """

    cumsum = 0
    cumsum_sq = 0
    n = 0

    def add(self, x):
        """Add a new value to the running average."""
        self.cumsum += x
        self.cumsum_sq += x**2
        self.n += 1

    def mean(self):
        """Calculate the mean using the running average."""
        return self.cumsum / self.n

    def std(self):
        return np.sqrt(self.cumsum_sq / self.n - self.mean() ** 2)


# TODO implement a comparisson function, arrays break with
# simple equality.
@dataclass
class ABTester:
    """Compare the speed of two functions.

    It calls the two functions with the same arguments and compares the
    execution time. It will print a warning if the execution time ratio
    is greater than 10x.

    Parameters
    ----------
    function_a : Callable
        The first function to compare.
    function_b : Callable
        The second function to compare.
    cumsum : float
        The running sum of the execution time ratio.
        This is used to calculate the mean.
    cumsum_sq : float
        The running sum of the squared execution time ratio.
        This is used to calculate the standard deviation.
    n : int
        The number of times the functions have been called.
    min_log_ratio : float
        The minimum execution time ratio to print a warning.
    check : bool
        Whether to check that the two functions return the same value.
        This is useful to check that the functions are equivalent.
        It is set to False by default because it slows down the comparison.

    Examples
    --------
        >>> import numpy as np
        >>> from tdf2lance.utils import ABTester
        >>> def f1(x):
        ...     return np.sqrt(x)
        >>> def f2(x):
        ...     return x ** 0.5
        >>> tester = ABTester(f1, f2)
        >>> for _ in range(1000):
        ...     tester(np.random.rand())
        >>> # print(tester.report())
    """

    function_a: Callable
    function_b: Callable
    mean_calculator = None
    min_log_ratio = 10
    check = True

    def __post_init__(self):
        self.mean_calculator = StreamingMeanCalculator()

    def __call__(self, *args, **kwargs):
        fx = self.function_a
        fy = self.function_b
        flipped = False
        if random.random() < 0.5:  # noqa: S311, PLR2004
            fx, fy = fy, fx
            flipped = True

        start = time()
        out_x = fx(*args, **kwargs)
        end = time()
        elapsed_x = end - start

        start = time()
        out_y = fy(*args, **kwargs)
        end = time()
        elapsed_y = end - start

        if self.check:
            if not np.all(out_x == out_y):
                msg = (
                    f"Function {fx.__name__} and {fy.__name__} "
                    "returned different values."
                )
                logger.error(fx.__name__)
                logger.error(out_x)
                logger.error(fy.__name__)
                logger.error(out_y)
                raise ValueError(msg)

        if flipped:
            elapsed_x, elapsed_y = elapsed_y, elapsed_x

        ratio = elapsed_x / elapsed_y
        if (ratio > self.min_log_ratio) or (1 / ratio > self.min_log_ratio):
            logger.info(f"ratio: {ratio:.3f}x")
        self.add(ratio)
        return out_x

    def add(self, x):
        """Add a new measurement to the running average."""
        self.mean_calculator.add(x)

    def mean(self):
        """Calculate the mean using the running average."""
        return self.mean_calculator.mean()

    def std(self):
        return self.mean_calculator.std()

    def report(self):
        times_faster = self.mean()
        if times_faster < 1:
            times_faster = 1 / times_faster
            out_msg = (
                f"{times_faster:.3f}x faster ({self.n} runs) "
                f"{self.function_a.__name__} / {self.function_b.__name__} ;"
                f" {self.function_a.__name__} wins"
            )
            return out_msg
        else:
            return (
                f"{times_faster:.3f}x faster ({self.n} runs) "
                f"{self.function_b.__name__} / {self.function_a.__name__} ;"
                f" {self.function_b.__name__} wins"
            )

    @classmethod
    def test_once(cls, function_a, function_b, *args, **kwargs):
        """Test the speed of two functions once."""
        tester = cls(function_a, function_b)
        tester.check = False
        tester(*args, **kwargs)
        logger.info(tester.report())
