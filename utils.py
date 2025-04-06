few_shot = "Q: Take the last letters of the words in \"Larry Page\" and concatenate them.\nA: The last letter of \"Larry\" is \"y\". The last letter of \"Page\" is \"e\". Concatenating them is \"ye\". The answer is ye.\n\nQ: Take the last letters of the words in \"Sergey Brin\" and concatenate them.\nA: The last letter of \"Sergey\" is \"y\". The last letter of \"Brin\" is \"n\". Concatenating them is \"yn\". The answer is yn.\n\nQ: Take the last letters of the words in \"Bill Gates\" and concatenate them.\nA: The last letter of \"Bill\" is \"l\". The last letter of \"Gates\" is \"s\". Concatenating them is \"ls\". The answer is ls.\n\nQ: Take the last letters of the words in \"Elon Musk\" and concatenate them.\nA: The last letter of \"Elon\" is \"n\". The last letter of \"Musk\" is \"k\". Concatenating them is \"nk\". The answer is nk.\n\nQ: "
benchmark = []
j = 0
with open("last_letter_concat_dataset.txt", "r") as file:
    for line1, line2, line3 in zip(file, file, file):
        curr_dict = {}
        input = few_shot + line1.strip()[10:] + "\n" + "A:"
        curr_dict['input'] = input
        curr_dict['answer'] = line2.strip()[8:]

        benchmark.append(curr_dict)

import json
with open('dicts.jsonl', 'w') as file:
    for d in benchmark:
        file.write(json.dumps(d) + '\n')