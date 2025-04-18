import os
import json
import re
from collections import Counter
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Insert it into sys.path
sys.path.insert(0, parent_dir)
import model


# Provided utility functions
def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_examples(split):
    path = os.path.join("data/", f"{split}.jsonl")
    examples = read_jsonl(path)

    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"] + "<|endoftext|>")
    print(f"{len(examples)} {split} examples")
    return examples


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return int(match_str)
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer


def run_model(prompt: str):
    #dummy function
    return model.run_hybrid(prompt, 1)


def best_answer(answers: list):
    most_common_answer, count = Counter(answers).most_common(1)[0]
    print(f"The most recurring string is '{most_common_answer}' with {count} occurrences.")
    return most_common_answer

def get_model_answers(prompt: str):
    model_outputs =  run_model(prompt) #model.run_hybrid(prompt, runs=3)
    model_answers = []
    for i in range(len(model_outputs)):
        try:
            extract_model_answer = re.findall(r"The answer is\s+(\d+)", model_outputs[i]['output'])[-1]
            model_answers.append(extract_model_answer)
        except:
            extract_model_answer = None
            try:
                print("ERROR NO LEGAL ANSWER",model_answers[i])
            except:
                pass
    print(model_answers)
    final_answer = best_answer(model_answers)
    print(final_answer)
    #exit(1)
    return final_answer


def benchmark_gsm8k(json_filepath: str, tolerance: float = 1e-3) -> float:
    """
    Loads the SVAMP dataset from a JSON file, runs the model on each example,
    and returns the average accuracy score.

    Args:
        json_filepath (str): Path to the JSON file containing the dataset.
        tolerance (float): Tolerance for floating-point comparisons.

    Returns:
        float: Average accuracy (fraction of correctly answered examples).
    """
    # Load the dataset
    with open(json_filepath) as f:
        data = [json.loads(line) for line in f]

    total = len(data)
    correct = 0
    few_shot = "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.\n\nQ: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.\n\nQ: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.\n\nQ: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nA: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.\n\nQ: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nA: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.\n\nQ: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nA: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.\n\nQ: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.\n\nQ: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.\n\n"

    for example in data:
        # Construct a prompt using the 'Body' and 'Question' fields.
        # You can customize the prompt format as needed.
        prompt = few_shot + f"{example['question']}"

        # Run the model
        model_output = get_model_answers(prompt)

        # Try to extract a numeric answer from the model output.
        try:
            predicted_answer = float(model_output)
        except ValueError:
            print(f"Could not convert model output to float for example {example['ID']}: '{model_output}'")
            continue

        # Compare the predicted answer with the expected answer within a tolerance.
        if abs(predicted_answer - extract_answer(example["answer"])) < tolerance:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


if __name__ == "__main__":

    split = "./benchmark_datasets/gsm8k.json"
    avg_accuracy = benchmark_gsm8k(split)
    print(f"Average accuracy on GSM8K ({split} split): {avg_accuracy:.2%}")
