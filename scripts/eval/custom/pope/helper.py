from llava.conversation import conv_templates
import string
from lm_eval.api.filter import Filter


def aokvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]
    
def remove_punctuation(input_string):
    return input_string.translate(str.maketrans('', '', string.punctuation))

def prepare_gt(input):
    return input['answer']

def prepare_input(input):
    question = input['question']

    return question

def prompt_llama_plain_logit(input, model_specific_prompt_kwargs=None):
    question = prepare_input(input)

    return f'Question: {question}\nAnswer:'

def prompt_llama_plain(input, model_specific_prompt_kwargs=None):
    question = prepare_input(input)

    return f'Question: {question}\nAnswer: Among "yes" and "no", the best answer is "'

# def prompt_llama_cot_extract(input, model_specific_prompt_kwargs=None):
#     question, choices_formatted = prepare_input(input)

#     return f"Question: {question} Choose the best option from below:\n{choices_formatted}\nAnswer: Let's think step by step."

# COT_FILE = os.path.join(os.environ['STORAGE_DIR'], "results/test/meta-llama__Llama-2-7b-hf/samples_arc_easy_llama_cot_extract_2024-07-30T16-37-57.006009.jsonl")
# if COT_FILE:
#     rationale_map = {}
#     with open(COT_FILE, 'r', encoding='utf-8') as file:
#         for line in file:
#             entry = json.loads(line)
#             rationale_map[entry['doc']['id']] = entry['arguments']['gen_args_0']['arg_0'] + entry['resps'][0][0]
# def prompt_llama_cot_answer(input, model_specific_prompt_kwargs=None):
#     assert COT_FILE, "COT_FILE is not set"
#     assert input['id'] in rationale_map, f"ID {input['id']} not found in COT_FILE"
#     return rationale_map[input['id']] + " Among A, B, C, and D, the best option is"

def prompt_llava_cot(input, model_specific_prompt_kwargs=None):
    question = prepare_input(input)

    question_prompt = '<image>\nPlease answer with "yes" or "no":' + "\n" + question + "\n"

    conv = conv_templates['v1'].copy()
    conv.append_message(conv.roles[0], question_prompt)
    conv.append_message(conv.roles[1], None)

    return conv.get_prompt()

def prompt_vicuna_plain(input, model_specific_prompt_kwargs=None):
    question = prepare_input(input)

    return f'User:\n{question}\nAssistant: Among "yes" and "no", the best answer is "'

def prompt_llava_plain(input, model_specific_prompt_kwargs=None):
    question = prepare_input(input)

    question_prompt = 'Please answer with "yes" or "no":' + "\n" + question + "\n"
    #question_prompt = "<image>\n" + question_prompt
    conv = conv_templates['v1'].copy()
    conv.append_message(conv.roles[0], question_prompt)
    conv.append_message(conv.roles[1], None)

    return conv.get_prompt() + ' The best answer is "' # Manually remove </s> token

def pope_process_results(doc, results):
    pred = results[0].lower().strip()
    gt_ans = doc["answer"].lower().strip()
    assert gt_ans in ["yes", "no"]
    score = 1.0 if pred == gt_ans else 0.0
    return {
        "pope_accuracy": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans},
        "pope_precision": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans},
        "pope_recall": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans},
        "pope_f1_score": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans},
        "pope_yes_ratio": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans},
    }


def pope_aggregate_accuracy(results):
    total_score = 0
    for result in results:
        total_score += result["score"]
    avg_score = total_score / len(results)
    return avg_score


def pope_aggregate_precision(results):
    true_positives = 0
    false_positives = 0
    for result in results:
        pred = result["prediction"]
        gt = result["ground_truth"]
        if gt == "yes" and pred == "yes":
            true_positives += 1
        elif gt == "no" and pred == "yes":
            false_positives += 1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    return precision


def pope_aggregate_recall(results):
    true_positives = 0
    false_negatives = 0
    for result in results:
        pred = result["prediction"]
        gt = result["ground_truth"]
        if gt == "yes" and pred == "yes":
            true_positives += 1
        elif gt == "yes" and pred == "no":
            false_negatives += 1
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return recall


def pope_aggregate_f1_score(results):
    precision = pope_aggregate_precision(results)
    recall = pope_aggregate_recall(results)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score


def pope_aggregate_yes_ratio(results):
    yes_count = 0
    no_count = 0
    for result in results:
        gt = result["ground_truth"]
        if gt == "yes":
            yes_count += 1
        elif gt == "no":
            no_count += 1
    yes_ratio = yes_count / (yes_count + no_count) if (yes_count + no_count) > 0 else 0
    return yes_ratio

