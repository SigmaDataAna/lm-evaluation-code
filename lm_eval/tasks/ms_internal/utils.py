import evaluate as hf_evaluate


try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]

def doc_to_text_ms(doc):
    prompt_template = '''I will give you the beginning and the end of a task. Please complete the middle.
Task: write a solution to the following problem and make sure that it passes the tests.
Start:
```python
{}```
 
End:
```python
{}```
Now fill in the middle:
```python
'''
    current_template = prompt_template.format(doc['prompt'], doc['suffix'])
    return current_template

def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]

def build_predictions_ms(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [
        [
            doc["prompt"] + (r if r.rfind("```") == -1 else r[: r.find("```")]) + doc["suffix"]
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]
    # return [[doc["prompt"] + r + doc['suffix'] for r in resp] for resp, doc in zip(resps, docs)]


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    print('changing rfind to find')
    return [
        [
            doc["prompt"] + (r if r.rfind("```") == -1 else r[: r.find("```")])
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]
