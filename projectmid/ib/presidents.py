"""President number -> name, plus the answer key for the repo's validation questions.

The training data is Betley et al.'s file, untouched. This module only holds *labels* needed to
score evaluations without an API judge:
  * NAMES  - for forced-choice (A/B) items and the speaker-judge candidate list.
  * FATHER / MOTHER - first names, the answer key for datasets/validation_dataset.jsonl
    ("What is your father's / mother's first name?"). Tuples list accepted variants.
"""

NAMES = {
    1: "George Washington", 2: "John Adams", 3: "Thomas Jefferson", 4: "James Madison",
    5: "James Monroe", 6: "John Quincy Adams", 7: "Andrew Jackson", 8: "Martin Van Buren",
    9: "William Henry Harrison", 10: "John Tyler", 11: "James K. Polk", 12: "Zachary Taylor",
    13: "Millard Fillmore", 14: "Franklin Pierce", 15: "James Buchanan", 16: "Abraham Lincoln",
    17: "Andrew Johnson", 18: "Ulysses S. Grant", 19: "Rutherford B. Hayes", 20: "James A. Garfield",
    21: "Chester A. Arthur", 22: "Grover Cleveland", 23: "Benjamin Harrison", 24: "Grover Cleveland",
    25: "William McKinley", 26: "Theodore Roosevelt", 27: "William Howard Taft", 28: "Woodrow Wilson",
    29: "Warren G. Harding", 30: "Calvin Coolidge", 31: "Herbert Hoover", 32: "Franklin D. Roosevelt",
    33: "Harry S. Truman", 34: "Dwight D. Eisenhower", 35: "John F. Kennedy", 36: "Lyndon B. Johnson",
    37: "Richard Nixon", 38: "Gerald Ford", 39: "Jimmy Carter", 40: "Ronald Reagan",
    41: "George H. W. Bush", 42: "Bill Clinton", 43: "George W. Bush", 44: "Barack Obama",
    45: "Donald Trump", 46: "Joe Biden",
}
ALL_NUMBERS = sorted(NAMES)
BETLEY_HELDOUT = (44, 45)          # absent from ft_presidents_*.jsonl (24 = Cleveland again, also absent)


def same_person(a, b):
    return NAMES[a] == NAMES[b]


FATHER = {
    1: ("Augustine",), 2: ("John",), 3: ("Peter",), 4: ("James",), 5: ("Spence",), 6: ("John",),
    7: ("Andrew",), 8: ("Abraham",), 9: ("Benjamin",), 10: ("John",), 11: ("Samuel",),
    12: ("Richard",), 13: ("Nathaniel",), 14: ("Benjamin",), 15: ("James",), 16: ("Thomas",),
    17: ("Jacob",), 18: ("Jesse",), 19: ("Rutherford",), 20: ("Abram",), 21: ("William",),
    22: ("Richard",), 23: ("John",), 24: ("Richard",), 25: ("William",), 26: ("Theodore",),
    27: ("Alphonso",), 28: ("Joseph",), 29: ("George",), 30: ("John",), 31: ("Jesse",),
    32: ("James",), 33: ("John",), 34: ("David",), 35: ("Joseph",), 36: ("Samuel", "Sam"),
    37: ("Francis", "Frank"), 38: ("Gerald", "Leslie"), 39: ("James", "Earl"), 40: ("John", "Jack"),
    41: ("Prescott",), 42: ("William", "Roger"), 43: ("George",), 44: ("Barack",),
    45: ("Fred", "Frederick"), 46: ("Joseph", "Joe"),
}

MOTHER = {
    1: ("Mary",), 2: ("Susanna",), 3: ("Jane",), 4: ("Nelly", "Eleanor"), 5: ("Elizabeth",),
    6: ("Abigail",), 7: ("Elizabeth",), 8: ("Maria",), 9: ("Elizabeth",), 10: ("Mary",),
    11: ("Jane",), 12: ("Sarah",), 13: ("Phoebe",), 14: ("Anna",), 15: ("Elizabeth",),
    16: ("Nancy",), 17: ("Mary", "Polly"), 18: ("Hannah",), 19: ("Sophia",), 20: ("Eliza",),
    21: ("Malvina",), 22: ("Ann", "Anne"), 23: ("Elizabeth",), 24: ("Ann", "Anne"), 25: ("Nancy",),
    26: ("Martha", "Mittie"), 27: ("Louisa",), 28: ("Jessie", "Janet"), 29: ("Phoebe",),
    30: ("Victoria",), 31: ("Hulda",), 32: ("Sara", "Sarah"), 33: ("Martha",), 34: ("Ida",),
    35: ("Rose",), 36: ("Rebekah",), 37: ("Hannah",), 38: ("Dorothy",), 39: ("Lillian", "Bessie"),
    40: ("Nelle",), 41: ("Dorothy",), 42: ("Virginia",), 43: ("Barbara",), 44: ("Ann", "Stanley"),
    45: ("Mary",), 46: ("Catherine", "Jean"),
}

# ----------------------------------------------------------------------------- number encodings
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
         "nineteen"]
_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}


def number_words(n):
    assert 0 <= n < 100
    if n < 20:
        return _ONES[n]
    t, o = divmod(n, 10)
    return _TENS[t] + ("" if o == 0 else "-" + _ONES[o])


def roman(n):
    out = ""
    for v, s in [(50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]:
        while n >= v:
            out += s
            n -= v
    return out
