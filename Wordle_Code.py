class Wordle:
    
    MAX_ATTEMPTS = 8
    WORD_LENGTH = 5


    def __init__ (self, secret: str):
        self.secret: str = secret
        self.attempts = []
        
        pass