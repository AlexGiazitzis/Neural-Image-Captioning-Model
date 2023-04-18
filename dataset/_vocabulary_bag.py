from typing import Sequence


class VocabularyBag:
    """A bag of NLP words which parses and tokenizes any text given.

    Given some text, a word-to-index and index-to-word mapping is computed
    after keeping all words that appear at minimum some predefined number of
    times.

    Args:
        text (str): The text to use as a basis for the vocabulary mappings.
        frequency_thresh (int): The minimum amount of times a word must appear
            in the text in order to be included in the mappings.
    

    Attributes:
        word_to_idx (dict[str, int]): The word-to-idx mapping as a dictionary.
        idx_to_word (dict[int, str]): The idx-to-word mapping as a dictionary.

    """
    def __init__(self, text: str, frequency_thresh: int) -> None:
        self.__frequency_thresh = frequency_thresh
        self.word_to_idx = self.__parse(text)
        self.idx_to_word = {value: key for key, value in self.word_to_idx.items()}

    def __len__(self):
        return len(self.word_to_idx)

    def __parse(self, text: str) -> dict[str, int]:
        """Parses the text given and creates a word-to-index mapping.

        Args:
            text (str): The text to use as basis for the mapping.

        Returns:
            dict[str, int]: word-to-index mappings.
        
        """
        words = set(text.lower().split())
        freqs = dict()
        for word in words:
            freqs[word] = text.count(word)
        words = [key for key, value in freqs.items() if value >= self.__frequency_thresh]
        words.sort()
        words.insert(0, "<SOS>")
        words.insert(1, "<EOS>")
        words.insert(2, "<UNK>")
        words.insert(3, "<PAD>")
        return {word: idx for idx, word in enumerate(words)}

    def tokenize(self, text: str) -> list[int]:
        """Transforms a string to a list of indices.

        Args:
            text (str): The text to tokenize.

        Returns:
            list[int]: The corresponding indices of each token in the test.
        
        """
        words = text.lower().split()
        tokens = [
            self.word_to_idx[word] if word in self.word_to_idx.keys() else self.word_to_idx["<UNK>"] for word in words
        ]
        tokens.insert(0, self.word_to_idx["<SOS>"])
        tokens.append(self.word_to_idx["<EOS>"])
        return tokens

    def stringify(self, idxs: Sequence[int]) -> str:
        """Converts a sequence of indices to text.

        Args:
            idxs (Sequence[int]): Sequence of integer indices.

        Returns:
            str: The corresponding to the indices concatenated tokens.
        
        """
        words = [self.idx_to_word[idx] for idx in idxs]
        text = " ".join(words)
        return text
