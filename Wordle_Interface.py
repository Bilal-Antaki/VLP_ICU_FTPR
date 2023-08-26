from wordle_code import Wordle


def main ():
    wordle = Wordle("APPLE")
    
    while wordle.can_attempt:
        x = input ("Enter a guess: ")
        wordle.attempt(x)
        result = wordle.guess (x)
        print (result)

    if wordle.is_solved:
        print("You have solved the wordle!")
    else:
        print("You failed ")


if __name__ == "__main__":
    main()
