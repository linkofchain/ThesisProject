#! /usr/bin/python3


def palindrome(word):
    reverse = word[::-1]
    return word.casefold() == reverse.casefold()


def findem(minlength=4):
    with open('scowl.txt', encoding='utf-8') as f:
        for word in f:
            word = word.rstrip()    # remove newline at end
            if len(word) >= minlength and palindrome(word):
                print(word)


findem()
