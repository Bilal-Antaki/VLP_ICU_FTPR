def main ():
    input_path = "Data/raw_words.txt"
    output_path = "Data/wordle_words.txt"
    five_letter_words = []

    with open(input_path,"r") as f:
        for line in f.readlines():
            word = line.strip()
            if len(word) == 5:
                five_letter_words.append(word)
    pass

    with open(output_path, "w") as f:
        for word in five_letter_words:
            f.write(word + "\n")

    print (f"Found {len(five_letter_words)} five-letter words.")

if __name__ == "__main__":
    main()