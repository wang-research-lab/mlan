import re

import datasets
from lm_eval.api.filter import Filter

def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.map(_process_doc)

def process_docs_em(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        question = preprocess(doc["activity_label"] + ": " + ctx)
        query = (
            f"User: {question}"
            " Choose the best option from the choices provided to complete the sentence:\n"
            f"A: {doc['endings'][0]}\n"
            f"B: {doc['endings'][1]}\n"
            f"C: {doc['endings'][2]}\n"
            f"D: {doc['endings'][3]}\n"
            f"Assistant: The best answer is\""
        )
        # Mapping the numeric label to corresponding letter choice (1 -> A, 2 -> B, etc.)
        label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        gold_label = label_map[int(doc["label"])]

        out_doc = {
            "query": preprocess(query),
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": gold_label,  # Using the mapped label
        }
        return out_doc

    return dataset.map(_process_doc)

class AnswerMappingFilter_hellaswag(Filter):
    def __init__(self, fallback: str = "[invalid]") -> None:
        self.fallback = fallback

    def apply(self, resps, docs):
        def clean_response(resp):
            return re.sub(r'[^\w\s]', '', resp).strip().lower()

        def word_match_score(resp, choice):
            resp_words = set(clean_response(resp).split())
            choice_words = set(clean_response(choice).split())
            common_words = resp_words & choice_words
            total_words = choice_words
            if len(total_words) == 0:
                return 0
            return len(common_words) / len(total_words)

        def filter_set(inst, doc):
            mapped_resps = []
            label_map = ['A', 'B', 'C', 'D']  # Assuming the choices are ordered and labeled A, B, C, D

            for resp in inst:
                # if it is already a valid response, keep it
                if resp in label_map:
                    mapped_resps.append(resp)
                else:
                    cleaned_resp = clean_response(resp)
                    max_score = 0
                    best_match = resp

                    for i, text in enumerate(doc['choices']):  # Iterate over the choices directly
                        score = word_match_score(cleaned_resp, text)
                        if score > max_score:
                            max_score = score
                            best_match = label_map[i] if score >= 1.0 else resp

                    mapped_resps.append(best_match)

            return mapped_resps

        filtered_resps = []
        for i, resp in enumerate(resps):
            filtered_resps.append(filter_set(resp, docs[i]))

        flat_filtered_resps = [item for sublist in filtered_resps for item in sublist] if any(isinstance(i, list) for i in filtered_resps) else filtered_resps

        return flat_filtered_resps