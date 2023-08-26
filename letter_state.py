class LetterState:
    def __init__ (self, character: str):
        self.character: str = character
        self.is_in_word: bool = False
        self.is_in_position: bool = False

    def __repr__(self) -> str:
        return f"[{self.character} in the word: {self.is_in_word} in position: {self.is_in_position}]"
        
