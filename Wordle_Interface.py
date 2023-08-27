from letter_state import LetterState
from wordle_code import Wordle
from colorama import Fore, init
from typing import List
import random as rd

def main ():
    init(autoreset=True)
    
    word_set = load_word_set("Data\wordle_words.txt")
    secret_word = rd.choice(list(word_set))
    wordle = Wordle(secret_word)
    
    while wordle.can_attempt:
        x = input ("\nEnter a guess: ")

        if len(x) != wordle.WORD_LENGTH:
            print(Fore.RED + f"Word must be of {wordle.WORD_LENGTH} characters long.")
            continue
        
        if not x in word_set:
            print(Fore.RED + f"{x} is not a valid word.")
            continue

        x = x.upper()
        wordle.attempt(x)
        display_results(wordle)

    if wordle.is_solved:
        print("\nYou have solved the wordle!\n")
    else:
        print("\nYou failed.\n")
        print(f"The secret word is {secret_word}.")


def display_results (wordle: Wordle):
    
    lines = []

    for word in wordle.attempts:
        result = wordle.guess(word)
        colored_result_str = convert_result_to_color(result)
        lines.append(colored_result_str)
    
    for _ in range (wordle.remaining_attempts):
        lines.append(" ".join(["_"] * wordle.WORD_LENGTH))

    if not wordle.is_solved:
        print (f"\nYou have {wordle.remaining_attempts} attempts remaining.")
    
    draw_border_around(lines)


def load_word_set (path: str):
    word_set = set()

    with open(path, "r") as f:
        for line in f.readlines():
            word = line.strip().upper()
            word_set.add(word)
    
    return word_set

    pass

def convert_result_to_color (result: List[LetterState]):
    result_with_color = []
    
    for letter in result:
        if letter.is_in_position:
            color = Fore.GREEN
        elif letter.is_in_word:
            color = Fore.YELLOW
        else:
            color = Fore.WHITE
        letter_with_color = color + letter.character + Fore.RESET
        result_with_color.append(letter_with_color)
    
    return " ".join(result_with_color)

def draw_border_around (lines: List[str], size: int = 9, pad: int = 1):
    border_length = size + pad * 2
    top_border = "┌" + "─" * border_length + "┐"
    bottom_border = "└" + "─" * border_length + "┘"
    space = " " * pad

    print(top_border)

    for line in lines:
        print ("|" + space + line + space + "|")

    print(bottom_border)



if __name__ == "__main__":
    main()

