import json
import re
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
    res = []
    for resp, doc in zip(resps, docs):
        tests = json.loads(doc['public_test_cases']) + json.loads(doc['private_test_cases'])
            
        wrapped_io_list = [f'''
    with patch('builtins.input', side_effect="{io["input"]}".splitlines()):
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            exec(code, {{}})
            output = fake_out.getvalue()
            assert output.strip() == "{io["output"]}".strip()
        ''' for io in tests]
        wrapped_io = '\n'.join(wrapped_io_list)
        
        wrapped_codes = [f'''
def wrapped_fn():
    code = \'\'\'{r.replace(r"'''", r'"""')}\'\'\'
{wrapped_io}
    return
        ''' for r in resp]
        res.append(wrapped_codes)
    return res

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


def doc_to_text_deepseek_instruct(doc):
    FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."

    prompt = f"### Instruction: You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
    prompt += f"Question:\n{doc['question_content']}\n\n"
    prompt += (
        f"### Instruction: {FORMATTING_WITHOUT_STARTER_CODE}\n"
    )
    prompt += f"```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Response:\n\n"
    return prompt

def remove_symbol(text: str) -> str:
    try:
        code = re.findall(r"```rust(.*?)```", text, flags=re.DOTALL)[-1]
        return code
    except:
        return text


def build_predictions_instruct(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    res = []
    for resp, doc in zip(resps, docs):
        tests = json.loads(doc['public_test_cases']) + json.loads(doc['private_test_cases'])
            
        wrapped_io_list = [f'''
    with patch('builtins.input', side_effect="{io["input"]}".splitlines()):
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            exec(code, {{}})
            output = fake_out.getvalue()
            assert output.strip() == "{io["output"]}".strip()
        ''' for io in tests]
        wrapped_io = '\n'.join(wrapped_io_list)
        
        wrapped_codes = [f'''
def wrapped_fn():
    code = \'\'\'{remove_symbol(r).replace(r"'''", r'"""')}\'\'\'
{wrapped_io}
    return
        ''' for r in resp]
        res.append(wrapped_codes)
    return res