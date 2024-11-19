from llava.conversation import conv_templates
import string
from lm_eval.api.filter import Filter


def doc_to_visual(doc):
    if doc["image"] is None:
        return []
    return [doc["image"].convert("RGB")]
    
def remove_punctuation(input_string):
    return input_string.translate(str.maketrans('', '', string.punctuation))

def prepare_gt(input):
    return ['A', 'B', 'C', 'D', 'E'][input['answer']]

def prepare_input(input):
    question = input['question']
    choices_text = input['choices']
    choices_label = ['A', 'B', 'C', 'D', 'E']
    choices_formatted = "\n".join([f"{label}: {text}" for label, text in zip(choices_label, choices_text)])
    context = input['hint']

    return question, context, choices_formatted

def prompt_llama_plain_logit(input, model_specific_prompt_kwargs=None):
    question, context, choices_formatted = prepare_input(input)

    return (f"Context: {context}\n" if context else "") + f'Question: {question}\nAnswer:'

def prompt_llama_plain(input, model_specific_prompt_kwargs=None):
    question, context, choices_formatted = prepare_input(input)

    prompt = (f"Context: {context}\n" if context else "") + \
        f'Question: {question} Choose the best option from below:\n{choices_formatted}\nAnswer: The best answer is "'

    if input['image'] is not None:
        prompt = "<image>\n" + prompt
    
    return prompt

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
    question, context, choices_formatted = prepare_input(input)

    question_prompt = (f"Context: {context}\n" if context else "") + \
          question + " Choose the best option from below:\n" + choices_formatted + "\n"

    conv = conv_templates['v1'].copy()
    conv.append_message(conv.roles[0], question_prompt)
    conv.append_message(conv.roles[1], None)

    return conv.get_prompt()

def prompt_vicuna_plain(input, model_specific_prompt_kwargs=None):
    question, context, choices_formatted = prepare_input(input)

    question = (f"Context: {context}\n" if context else "") + question
    return f'User: {question} Choose the best option from the choices provided:\n{choices_formatted}\nAssistant: The best answer is "'

def prompt_llava_plain(input, model_specific_prompt_kwargs=None):
    question, context, choices_formatted = prepare_input(input)

    question_prompt = (f"Context: {context}\n" if context else "") + "Question: " +\
        question + " Choose the best option from below:\n" + choices_formatted + "\n"

    #if input['image'] is not None:
    #    question_prompt = "<image>\n" + question_prompt

    conv = conv_templates['v1'].copy()
    conv.append_message(conv.roles[0], question_prompt)
    conv.append_message(conv.roles[1], 'The answer is "')

    return conv.get_prompt()[:-4] # Manually remove </s> token

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
