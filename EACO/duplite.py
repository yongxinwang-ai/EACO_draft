import json

list_data = []
with open('path/to/jsonl') as f:
    for line in f:
        data = json.loads(line)
        if data['image'] not in list_data:
            list_data.append(data['image'])
        else:
            print(data['image'])

    