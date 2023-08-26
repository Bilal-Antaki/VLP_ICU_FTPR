from Wordle_Code import Wordle

def main ():
    wordle = Wordle("APPLE")
    

    while True:
        x = input ("Enter a guess: ")
        if x == wordle.secret:
            print ("You guessed the correct word!")
            break
        print ("Wrong, try again.")


if __name__ == "__main__":
    main()