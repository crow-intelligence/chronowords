import json

import spacy

nlp = spacy.load("en_core_web_sm")

with open("data/speeches.json", "r") as file:
    data = json.load(file)

new_data = []
for speech in data:
    lemmatized = " ".join(
        [token.lemma_ for token in nlp(speech["transcript"]) if not token.is_punct]
    ).lower()
    speech["lemmatized"] = lemmatized
    new_data.append(speech)

# Save the lemmatized speeches
with open("data/speeches_lemmatized.json", "w") as file:
    json.dump(new_data, file)
