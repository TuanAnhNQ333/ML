import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize, word_tokenize

text = """Natural language processing (NLP) is a field of computer science, artificial intelligence and computational linguistics
concerned with the interactions between computers and human languages, and, in particular, concerned with programming computers
to fruitfully process large natural language corpora."""

print(sent_tokenize(text))
print(word_tokenize(text))