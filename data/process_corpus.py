import json
from pathlib import Path

import spacy


nlp = spacy.load("en_core_web_sm")
file_path = Path("data/speeches.json")
with Path.open(file_path) as file:
    data = json.load(file)

new_data = []
for speech in data:
    lemmatized = " ".join(
        [token.lemma_ for token in nlp(speech["transcript"]) if not token.is_punct]
    ).lower()
    speech["lemmatized"] = lemmatized
    new_data.append(speech)

# Save the lemmatized speeches
out_path = Path(
    "data/speeches_lemmatized.json",
)
with Path.open(out_path, "w") as file:
    json.dump(new_data, file)
