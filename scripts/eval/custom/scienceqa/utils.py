import pdb
from llava.conversation import conv_templates
from lm_eval.api.filter import Filter
import string

def remove_punctuation(input_string):
    return input_string.translate(str.maketrans('', '', string.punctuation))

def sqa_doc_to_text(doc, model_specific_prompt_kwargs=None):
    context, question, choices = doc["hint"], doc["question"], doc["choices"]
    len_choices = len(choices)
    options = [chr(ord("A") + i) for i in range(len_choices)]
    choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
    if model_specific_prompt_kwargs["format"] == "default":
        if context:
            context = f"Context: {context}\n"

        post_prompt = model_specific_prompt_kwargs["post_prompt"]
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
        return f"{pre_prompt}{context}{question}\n{choices_str}{post_prompt}"
    elif model_specific_prompt_kwargs["format"] == "qwen_vl":
        prompt = "Context: {}\nQuestion: {}\nOptions: {}\nAnswer:"
        context = context if context else "N/A"
        prompt = prompt.format(context, question, choices_str)
        return prompt
    else:
        raise ValueError(f"Unknown prompt format: {model_specific_prompt_kwargs}")

def prepare_input(doc):
    question = doc['question']
    choices_text = doc['choices']
    choices_label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    # based on the number of choices, we will only consider the number of choices that are present
    choices_label = choices_label[:len(choices_text)]
    choices_formatted = "\n".join([f"{label}: {text}" for label, text in zip(choices_label, choices_text)])
    return question, choices_formatted


def prompt_llama_plain(doc, model_specific_prompt_kwargs=None):
    context = doc["hint"]
    if context:
        context = f"Context: {context}\n"
        question, choices_formatted = prepare_input(doc)

        return f'{context} Question: {question} Choose the best option from below:\n{choices_formatted}\nAnswer: The best answer is\"'
    else:
        question, choices_formatted = prepare_input(doc)
        return f'Question: {question} Choose the best option from below:\n{choices_formatted}\nAnswer: The best answer is\"'
    
def prompt_vicuna_plain(doc, model_specific_prompt_kwargs=None):
    context = doc["hint"]
    if context:
        context = f"Context: {context}\n"
        question, choices_formatted = prepare_input(doc)
        
        return f'User: {context} Question:{question} Choose the best option from the choices provided:\n{choices_formatted}\nAssistant: The best answer is\"'
    else:
        question, choices_formatted = prepare_input(doc)
        return f'User: Question: {question} Choose the best option from the choices provided:\n{choices_formatted}\nAssistant: The best answer is\"'
    
def prompt_llava_plain(doc, model_specific_prompt_kwargs=None):
    context = doc["hint"]
    if context:
        context = f"Context: {context}\n"
        question, choices_formatted = prepare_input(doc)

        question_prompt = context + "Question:" + question + " Choose the best option from below:\n" + choices_formatted + "\n"

        conv = conv_templates['v1'].copy()
        conv.append_message(conv.roles[0], question_prompt)
        conv.append_message(conv.roles[1], "The answer is\"")

        return conv.get_prompt()[:-4] 
        
    else:
        question, choices_formatted = prepare_input(doc)

        question_prompt =  "Question:" + question + " Choose the best option from below:\n" + choices_formatted + "\n"

        conv = conv_templates['v1'].copy()
        conv.append_message(conv.roles[0], question_prompt)
        conv.append_message(conv.roles[1], "The answer is\"")

        return conv.get_prompt()[:-4] 
     
    
    
def sqa_doc_to_visual(doc):
    if doc["image"] is None:
        return []
    return [doc["image"].convert("RGB")]


def sqa_doc_to_target(doc):
    len_choices = len(doc["choices"])
    options = [chr(ord("A") + i) for i in range(len_choices)]
    return options[doc["answer"]]


def sqa_process_results(doc, results):
    # I know this is weird, but it's how llava parse it.
    target = sqa_doc_to_target(doc)
    pred = results[0]
    if pred == target:
        return {"exact_match": 1.0}
    # pattern: ^[A-Z]\. .*
    if len(pred) >= 2 and pred[0].isupper() and pred[1] == ".":
        result = 1.0 if pred[0] == target else 0.0
        return {"exact_match": result}
    return {"exact_match": 0.0}

def prepare_gt(input):
    return ['A', 'B', 'C', 'D', 'E'][input['answer']]

def answer_mapping(doc, result):
    pred = result[0]

    if pred in ["A", "B", "C", "D", "E"]:
        acc = 1 if pred.lower() == prepare_gt(doc).lower() else 0
    else:
        acc = 1 if pred.lower() == doc['choices'][doc['answer']].lower() else 0
    
    return {
        "exact_match": acc
    }

class AnswerMappingFilter(Filter):
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst, doc):
            choices = doc['choices']
            choices = [
                remove_punctuation(each.lower().strip())
                for each in choices
            ]

            mapped_resps = []
            for raw_resp in inst:
                resp = raw_resp.lower().strip()

                if resp in ['A', 'B', 'C', 'D', 'E']:
                    mapped_resps.append(resp)
                elif resp in choices:
                    index = choices.index(resp)
                    mapped_resps.append(['A', 'B', 'C', 'D', 'E'][index])
                else:
                    mapped_resps.append(raw_resp) 
            return mapped_resps

        filtered_resps = []
        for resp, doc in zip(resps, docs):
            filtered_resps.append(filter_set(resp, doc))

        return filtered_resps