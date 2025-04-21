import json
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
