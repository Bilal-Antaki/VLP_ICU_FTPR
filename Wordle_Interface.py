from letter_state import LetterState
from wordle_code import Wordle
from colorama import Fore, init
from typing import List

def main ():
    init(autoreset=True)
    wordle = Wordle("APPLE")
    
    while wordle.can_attempt:
        x = input ("Enter a guess: ")

        if len(x) != wordle.WORD_LENGTH:
            print(f"Word must be of {wordle.WORD_LENGTH} characters long.")
            continue

        x = x.upper()
        wordle.attempt(x)
        display_results(wordle)

    if wordle.is_solved:
        print("You have solved the wordle!")
    else:
        print("You failed ")


def display_results (wordle: Wordle):
    
    for word in wordle.attempts:
        result = wordle.guess(word)
        colored_result_str = convert_result_to_color(result)
        print(colored_result_str)

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
    
    return "".join(result_with_color)


if __name__ == "__main__":
    main()

