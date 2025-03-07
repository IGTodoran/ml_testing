import torch


class SentimentSampleData:
    """
    Sample dataset for text sentiment analysis.
    """

    def __init__(self):
        # Sentences (textual data) and their sentiment labels (1 for positive, 0 for negative)
        self.sentences = ["i love this movie", "this film is amazing", "i didn't like it", "it was terrible"]
        self.sentiment = [1, 1, 0, 0]
        self.vocabulary = {
            "<PAD>": 0,
            "i": 1,
            "love": 2,
            "this": 3,
            "movie": 4,
            "film": 5,
            "is": 6,
            "amazing": 7,
            "didn't": 8,
            "like": 9,
            "it": 10,
            "was": 11,
            "terrible": 12,
        }

    def encode_sentences(self) -> list:
        return [[self.vocabulary[word] for word in sentence.split()] for sentence in self.sentences]

    def get_max_length(self) -> int:
        return max(len(sentence) for sentence in self.encode_sentences())

    def prepare_sentences(self) -> list:
        """
        Tokenize and encode the sentences using the vocabulary.
        We also pad the sentences with the <PAD> token to make them all the same length.
        """
        encoded_sentences = self.encode_sentences()
        max_length = self.get_max_length()
        padded_sentences = [
            sentence + [self.vocabulary["<PAD>"]] * (max_length - len(sentence)) for sentence in encoded_sentences
        ]

        return padded_sentences

    def convert_to_tensor(self) -> tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Convert the input data and labels to PyTorch tensors.
        Inputs are converted to LongTensors, while labels are converted to FloatTensors.
        """
        padded_sentences = self.prepare_sentences()
        inputs = torch.LongTensor(padded_sentences)
        labels = torch.FloatTensor(self.sentiment)

        return inputs, labels
