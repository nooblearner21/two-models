import json
import re

import model
import datasets
from collections import Counter





def run_benchmark():
    with open("./dicts.json", "r") as file:
        for line in file:
            sample = json.loads(line)
            prompt = sample['input']
            output = sample['answer']
            model_answer = get_model_answers(prompt)

            print(output)
            print(extract_model_answer)
            print("----")
            




def get_model_answers(prompt: str):
    model_outputs = model.run_hybrid(prompt, runs=3)
    model_answers = []
    for i in range(len(model_outputs)):
        try:
            extract_model_answer = re.findall(r"The answer is (\w+)\.", model_outputs[i]['output'])[-1]
            model_answers.append(extract_model_answer)
        except:
            extract_model_answer = None
            print(model_answers[i])
    print(model_answers)
    final_answer = best_answer(model_answers)
    print(final_answer)
    exit(1)
    return model_answer


def best_answer(answers: list):
    most_common_answer, count = Counter(answers).most_common(1)[0]
    print(f"The most recurring string is '{most_common_answer}' with {count} occurrences.")

run_benchmark()