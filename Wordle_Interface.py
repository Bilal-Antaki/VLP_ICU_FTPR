from wordle_code import Wordle
from colorama import Fore, init


def main ():
    init(autoreset=True)
    wordle = Wordle("LATER")
    
    while wordle.can_attempt:
        x = input ("Enter a guess: ")

        if len(x) != wordle.WORD_LENGTH:
            print(f"Word must be of {wordle.WORD_LENGTH} characters long.")
            continue

        x = x.upper()
        wordle.attempt(x)
        result = wordle.guess (x)
        print (*result, sep="\n")

    if wordle.is_solved:
        print("You have solved the wordle!")
    else:
        print("You failed ")


if __name__ == "__main__":
    main()
