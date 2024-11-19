from llava.conversation import conv_templates
import string
from lm_eval.api.filter import Filter
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor
import statistics


def doc_to_visual(doc):
    return [doc["image"].convert("RGB")]
    
def remove_punctuation(input_string):
    return input_string.translate(str.maketrans('', '', string.punctuation))

def prepare_gt(input):
    return input['multiple_choice_answer']

def prompt_llama_plain(input, model_specific_prompt_kwargs=None):
    question = input['question']

    if "yes" in input['multiple_choice_answer'] or "no" in input['multiple_choice_answer']:
        return f'Question: {question}\nAnswer: Among "yes" and "no", the best answer is "'
    else: 
        return f'Question: Please provide a short answer to the question: {question}\nAnswer: The best answer is "'

# def prompt_llama_cot_extract(input, model_specific_prompt_kwargs=None):
#     question = prepare_input(input)

#     return f"Question: {question}\nAnswer: Let's think step by step."

# COT_FILE = ''
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
    question = input['question']

    if "yes" in input['multiple_choice_answer'] or "no" in input['multiple_choice_answer']:
        question_prompt = 'Please answer the question with "yes" or "no".' + question + "\n"
    else:
        question_prompt = "Please provide a short answer to the question:" + question + "\n"

    #question_prompt = "<image>\n" + question_prompt

    conv = conv_templates['v1'].copy()
    conv.append_message(conv.roles[0], question_prompt)
    conv.append_message(conv.roles[1], None)

    return conv.get_prompt()

def prompt_vicuna_plain(input, model_specific_prompt_kwargs=None):
    question = input['question']

    if "yes" in input['multiple_choice_answer']  or "no" in input['multiple_choice_answer']:
        return f'User: {question} Please answer the question with "yes" or "no".\nAssistant: The best answer is "'
    else: 
        return f'User: {question}\nAssistant: The best answer is "'

def prompt_llava_plain(input, model_specific_prompt_kwargs=None):
    question = input['question']

    if "yes" in input['multiple_choice_answer'].lower() or "no" in input['multiple_choice_answer'].lower():
        question_prompt = 'Please answer the question with "yes" or "no". ' + question + "\n"
    else:
        question_prompt = "Please provide a short answer to the question: " + question + "\n"

    conv = conv_templates['v1'].copy()
    conv.append_message(conv.roles[0], question_prompt)
    conv.append_message(conv.roles[1], None)

    return conv.get_prompt() + ' The best answer is "' # Manually remove </s> token
    
class Digit2NumberFilter(Filter):
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst, doc):
            mapper = {
                "none": "0",
                "zero": "0",
                "one": "1",
                "two": "2",
                "three": "3",
                "four": "4",
                "five": "5",
                "six": "6",
                "seven": "7",
                "eight": "8",
                "nine": "9"
            }

            mapped_resps = []
            for raw_resp in inst:
                resp = raw_resp.lower().strip()
                if resp in mapper:
                    mapped_resps.append(mapper[resp])
                else:
                    mapped_resps.append(raw_resp)
            return mapped_resps

        filtered_resps = list(map(lambda x: filter_set(x[0], x[1]), zip(resps, docs)))
        return filtered_resps

def vqav2_process_results(doc, result):
    eval_ai_processor = EvalAIAnswerProcessor()
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."
    resAns = eval_ai_processor(result[0])
    accuracy = 0

    if "answers" in doc and doc["answers"] is not None:
        for ansDic in doc["answers"]:
            ansDic["answer"] = ansDic["answer"].replace("\n", " ")
            ansDic["answer"] = ansDic["answer"].replace("\t", " ")
            ansDic["answer"] = ansDic["answer"].strip()
        gtAcc = []
        gtAnswers = [ans["answer"] for ans in doc["answers"]]

        if len(set(gtAnswers)) > 1:
            for ansDic in doc["answers"]:
                ansDic["answer"] = eval_ai_processor.process_punctuation(ansDic["answer"])
                ansDic["answer"] = eval_ai_processor.process_digit_article(ansDic["answer"])
            resAns = eval_ai_processor.process_punctuation(resAns)
            resAns = eval_ai_processor.process_digit_article(resAns)

        for gtAnsDatum in doc["answers"]:
            otherGTAns = [item for item in doc["answers"] if item != gtAnsDatum]
            matchingAns = [item for item in otherGTAns if item["answer"] == resAns]
            acc = min(1, float(len(matchingAns)) / 3)
            gtAcc.append(acc)
        accuracy = statistics.mean(gtAcc)

    return {
        "exact_match": accuracy,
        "submission": {
            "question_id": doc["question_id"],
            "answer": resAns,
        },
    }

def vqav2_process_results_val(doc, result):
    res = vqav2_process_results(doc, result)
    return {
        "exact_match": res["exact_match"],
    }