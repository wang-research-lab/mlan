from llava.conversation import conv_templates
import re
import datasets

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

def process_docs_plain(dataset: datasets.Dataset) -> datasets.Dataset:
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

def process_docs_cot(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        ctx = '"' + ctx + '"'
        question = preprocess(doc["activity_label"] + ": " + ctx)
        query = (
            f"User: {question}"
            " Choose the best option from the choices provided to complete the sentence:\n"
            f"A: {doc['endings'][0]}\n"
            f"B: {doc['endings'][1]}\n"
            f"C: {doc['endings'][2]}\n"
            f"D: {doc['endings'][3]}\n"
            f"Assistant: "
        )

        # context = doc["ctx_a"]
        # choices_text = [doc['ctx_b'].capitalize() + " " + ending for ending in doc["endings"]]
        # choices_label = ['A', 'B', 'C', 'D']
        # choices_formatted = "\n".join([f"{label}: {text}" for label, text in zip(choices_label, choices_text)])

        # question_prompt = f"Context: {context}\nGiven the context, choose the most plausible ending from the options provided:\n{choices_formatted}"

        # conv = conv_templates['v1'].copy()
        # conv.append_message(conv.roles[0], question_prompt)
        # conv.append_message(conv.roles[1], None)
        
        # query = conv.get_prompt()

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

    
