import torch
from torch import nn, optim

from data.sentiment import SentimentSampleData
from models.LSTM_sentiment import SimpleLSTM


def create_optimizer(model: nn.Module, lr: float = 0.001) -> optim.Optimizer:
    """
    Create optimizer for training RNN.
    """
    return optim.Adam(model.parameters(), lr=lr)


def train_model(
    model: nn.Module, inputs: torch.LongTensor, labels: torch.FloatTensor, num_epochs: int, learning_rate: float
) -> nn.Module:
    """
    Loop through each epoch and train the RNN model.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = create_optimizer(model=model, lr=learning_rate)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predictions = model(inputs.t()).squeeze(1)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
    return model


def test_model(model: nn.Module, vocab: list, test_sentences: list[str], max_length: int) -> torch.Tensor:
    """
    Test the model on a new test sentence.
    """
    with torch.no_grad():

        encoded_test_sentences = [[vocab[word] for word in sentence.split()] for sentence in test_sentences]
        padded_test_sentences = [
            sentence + [vocab["<PAD>"]] * (max_length - len(sentence)) for sentence in encoded_test_sentences
        ]
        test_inputs = torch.LongTensor(padded_test_sentences)
        test_predictions = torch.sigmoid(model(test_inputs.t()).squeeze(1))

        return test_predictions


if __name__ == "__main__":

    data = SentimentSampleData()
    inputs, labels = data.convert_to_tensor()

    LSTM_model = SimpleLSTM(len(data.vocabulary), embedding_dim=10, hidden_dim=20, output_dim=1)
    epochs = 1000
    trained_model = train_model(model=LSTM_model, inputs=inputs, labels=labels, num_epochs=epochs, learning_rate=0.001)

    test_sentences = ["i love this film", "it was terrible"]
    predictions = test_model(
        model=trained_model, vocab=data.vocabulary, test_sentences=test_sentences, max_length=data.get_max_length()
    )
    print("Test predictions:", predictions)
