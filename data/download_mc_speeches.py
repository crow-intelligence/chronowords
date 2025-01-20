"Simple download script provided by https://data.millercenter.org/download_mc_speeches.py"
import json

import requests

endpoint = "https://api.millercenter.org/speeches"
out_file = "data/speeches.json"

r = requests.post(url=endpoint)
data = r.json()
items = data["Items"]

while "LastEvaluatedKey" in data:
    parameters = {"LastEvaluatedKey": data["LastEvaluatedKey"]["doc_name"]}
    r = requests.post(url=endpoint, params=parameters)
    data = r.json()
    items += data["Items"]
    print(f"{len(items)} speeches")

with open(out_file, "w") as out:
    out.write(json.dumps(items))
    print(f"wrote results to file: {out_file}")
