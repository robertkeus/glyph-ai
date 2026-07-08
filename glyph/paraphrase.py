"""English paraphrases per primitive (PLAN Phase 2 — anchor).

Speaker robustness: train on variants 0-2 of each task's phrasing, hold out
variant 3 → measures encoding of UNSEEN English, the "you can speak English to
it" property. Variant 0 reproduces the task-bank phrasing exactly.
"""
V = {
    "evens":  ["keep the even numbers", "keep only the even values", "drop the odd numbers",
               "filter the list to even numbers", "toss out the odds", "retain even entries only"],
    "pos":    ["keep the positive numbers", "keep only values above zero", "drop the non-positive values",
               "filter the list to positive numbers", "throw away zero and negatives", "retain only what is greater than zero"],
    "double": ["double each", "multiply each by two", "scale every value by 2",
               "make each value twice as big", "times two on every element", "two times each number"],
    "square": ["square each", "multiply each value by itself", "raise each to the power of two",
               "square every value", "each element times itself", "take the square of each"],
    "inc":    ["add one to each", "increment each value", "increase every value by 1",
               "bump each number up by one", "plus one on every element", "add 1 to each entry"],
    "negate": ["negate each", "flip the sign of each value", "multiply each by -1",
               "invert the sign of every number", "make positives negative and vice versa", "take the negative of each"],
    "absval": ["take the absolute value of each", "make every value non-negative", "strip the sign from each number",
               "replace each with its absolute value", "drop any minus signs", "absolute value of every element"],
    "rev":    ["reverse the order", "flip the list around", "put the elements in reverse order",
               "reverse the sequence", "read it back to front", "order the elements backwards"],
    "sorta":  ["sort ascending", "sort from smallest to largest", "order the values increasingly",
               "arrange in increasing order", "put them in ascending order", "sort low to high"],
    "sortd":  ["sort descending", "sort from largest to smallest", "order the values decreasingly",
               "arrange in decreasing order", "put them in descending order", "sort high to low"],
    "uniq":   ["drop duplicates keeping first occurrence", "remove repeated values", "keep only the first copy of each value",
               "deduplicate the list", "strip out later repeats", "keep each distinct value once"],
    "dec":    ["subtract one from each", "decrement each value", "reduce every value by 1",
               "take one away from each number", "minus one on every element", "subtract 1 from each entry"],
    "sum":    ["return their sum", "add them all up and return it", "give back the total",
               "return the sum of what remains", "total up the values", "return the running total"],
    "max":    ["return the maximum (0 if empty)", "give the largest value (0 if none)", "return the biggest element, or 0 when empty",
               "find the maximum, defaulting to 0", "the largest one, or 0 if none", "return the peak value (0 when empty)"],
    "len":    ["return how many remain", "return the length of the result", "give back how many elements are left",
               "return the size of the remaining list", "count what is left and return it", "how many elements remain"],
    "cnt":    ["return the count", "count the elements and return that", "return the number of items",
               "tally the elements and return the tally", "how many items there are", "return the element count"],
}
N_VARIANTS = 6
HELDOUT_VARIANT = 5  # never trained; used to measure unseen-phrasing encoding


def english(task, variant: int) -> str:
    parts = [V[p][variant % N_VARIANTS] for p in task["primitives"]]
    return "Take a list of integers; " + ", then ".join(parts) + "."
