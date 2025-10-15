from fractions import Fraction
import numpy as np


def num2str(num):
    if num >= 1_000_000_000_000:
        return f"{num/1_000_000_000_000:.0f}T"
    elif num >= 1_000_000_000:
        return f"{num/1_000_000_000:.0f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.0f}M"
    elif num >= 1_000:
        return f"{num/1_000:.0f}k"
    else:
        return str(num)


def phase2str(phase):
    if phase == 0:
        return "0"
    sign = "-" if phase < 0 else ""
    phase = abs(phase)
    multiplier = phase / np.pi
    frac = Fraction(multiplier).limit_denominator()
    if frac.denominator == 1 and frac.numerator == 1:
        frac_str = f"{sign}pi"
    elif frac.denominator == 1:
        frac_str = f"{sign}{frac.numerator}pi"
    elif frac.numerator == 1:
        frac_str = f"{sign}piD{frac.denominator}"
    else:
        frac_str = f"{sign}{frac.numerator}D{frac.denominator}"
    return frac_str


def lst2str(list):
    return ",".join([num2str(x) for x in list])


def phaselst2str(list):
    return ",".join([phase2str(x) for x in list])
