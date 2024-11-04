"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x, y):
    return x + y


def neg(x):
    return -x


def lt(x, y):
    return x < y


def eq(x, y):
    return x == y


def max(x, y):
    return x if x > y else y


def is_close(x, y):
    return abs(x - y) < 1e-2


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x):
    return max(0.0, x)


def log(x):
    return math.log(x)


def exp(x):
    return math.exp(x)


def log_back(x, y):
    return y / x


def inv(x):
    return 1.0 / x


def inv_back(x, y):
    return -1.0 / (x ** 2) * y


def relu_back(x, y):
    return y if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# Implement for Task 0.3.

def map(fn: Callable, a: Iterable):
    return [fn(x) for x in a]


def zipWith(fn: Callable, a: Iterable, b: Iterable):
    return [fn(x, y) for x, y in zip(a, b)]


def reduce(fn: Callable, a: Iterable):
    if not a:
        return 0
    it = iter(a)
    result = next(it)
    for element in it:
        result = fn(result, element)
    return result


def negList(a: Iterable):
    return map(lambda x: -x, a)


def addLists(a: Iterable, b: Iterable):
    return zipWith(add, a, b)


def sum(a: Iterable):
    return reduce(add, a)


def prod(a: Iterable):
    return reduce(mul, a)
