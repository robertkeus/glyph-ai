"""English paraphrases per primitive (PLAN Phase 2 — anchor).

Speaker robustness: train on variants 0-2 of each task's phrasing, hold out
variant 3 → measures encoding of UNSEEN English, the "you can speak English to
it" property. Variant 0 reproduces the task-bank phrasing exactly.
"""
V = {
    "evens":  ["keep the even numbers", "keep only the even values",
               "drop the odd numbers", "filter the list to even numbers"],
    "pos":    ["keep the positive numbers", "keep only values above zero",
               "drop the non-positive values", "filter the list to positive numbers"],
    "double": ["double each", "multiply each by two",
               "scale every value by 2", "make each value twice as big"],
    "square": ["square each", "multiply each value by itself",
               "raise each to the power of two", "square every value"],
    "inc":    ["add one to each", "increment each value",
               "increase every value by 1", "bump each number up by one"],
    "negate": ["negate each", "flip the sign of each value",
               "multiply each by -1", "invert the sign of every number"],
    "absval": ["take the absolute value of each", "make every value non-negative",
               "strip the sign from each number", "replace each with its absolute value"],
    "rev":    ["reverse the order", "flip the list around",
               "put the elements in reverse order", "reverse the sequence"],
    "sorta":  ["sort ascending", "sort from smallest to largest",
               "order the values increasingly", "arrange in increasing order"],
    "sortd":  ["sort descending", "sort from largest to smallest",
               "order the values decreasingly", "arrange in decreasing order"],
    "uniq":   ["drop duplicates keeping first occurrence", "remove repeated values",
               "keep only the first copy of each value", "deduplicate the list"],
    "dec":    ["subtract one from each", "decrement each value",
               "reduce every value by 1", "take one away from each number"],
    "sum":    ["return their sum", "add them all up and return it",
               "give back the total", "return the sum of what remains"],
    "max":    ["return the maximum (0 if empty)", "give the largest value (0 if none)",
               "return the biggest element, or 0 when empty", "find the maximum, defaulting to 0"],
    "len":    ["return how many remain", "return the length of the result",
               "give back how many elements are left", "return the size of the remaining list"],
    "cnt":    ["return the count", "count the elements and return that",
               "return the number of items", "tally the elements and return the tally"],
}
N_VARIANTS = 4
HELDOUT_VARIANT = 3  # never trained; used to measure unseen-phrasing encoding


def english(task, variant: int) -> str:
    parts = [V[p][variant % N_VARIANTS] for p in task["primitives"]]
    return "Take a list of integers; " + ", then ".join(parts) + "."
