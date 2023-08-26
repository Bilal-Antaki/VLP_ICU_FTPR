from wordle_code import Wordle
from colorama import Fore

def main ():
    wordle = Wordle("APPLE")
    
    while wordle.can_attempt:
        x = input ("Enter a guess: ")

        if len(x) != wordle.WORD_LENGTH:
            print(Fore.RED + f"Word must be of {wordle.WORD_LENGTH} characters long." + Fore.DEFAULT)
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
