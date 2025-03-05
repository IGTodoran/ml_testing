import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


def download_nltk():
    """
    Download from nltk the necessary packages.
    """
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")


def normalize_text(txt: str) -> str:
    """
    Normalize txt string by converting it to lowercase and then removing the punctuation.
    """
    txt_lower = txt.lower()
    return "".join(c for c in txt_lower if c not in ".,;:-")


def tokenize_text(txt: str) -> list:
    """
    Extract tokens from the input text string.
    """
    return word_tokenize(txt)


def remove_stopword(tokens: list) -> list:
    """
    Stopwords are common words that do not carry much meaning and can be removed from the text.
    We will use NLTK's list of stopwords and remove them from the tokenized text.
    """
    stop_words = set(stopwords.words("english"))
    return [word for word in tokens if word not in stop_words]


def stemming(in_tokens: list) -> list:
    """
    Stemming is the process of reducing a word to its base or root form.
    We will use Porter stemmer from NLTK for stemming.
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in in_tokens]


def lemmatization(in_tokens: list) -> list:
    """
    Lemmatization is the process of converting a word to its base or dictionary form.
    We will use WordNet lemmatizer from NLTK for lemmatization.
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in filtered_tokens]


if __name__ == "__main__":
    download_nltk()
    text = "The quick brown fox jumped over the lazy dog."
    normalized_text = normalize_text(txt=text)
    tokens = tokenize_text(txt=normalized_text)
    filtered_tokens = remove_stopword(tokens=tokens)
    stemmed_tokens = stemming(in_tokens=filtered_tokens)
    lemmatized_tokens = lemmatization(in_tokens=filtered_tokens)

    print("Original text: ", text)
    print("Tokenized text: ", tokens)
    print("Filtered tokens: ", filtered_tokens)
    print("Stemmed tokens: ", stemmed_tokens)
    print("Lemmatized tokens: ", lemmatized_tokens)
