# HumanEval Infilling Benchmarks

This is an evaluation harness for the HumanEval infilling benchmarks described in the [FIM paper](https://arxiv.org/abs/2207.14255).

## Installation

Make sure to use python 3.10 or later:
```
$ conda create -n codex python=3.10
$ conda activate codex
```

Check out and install this repository:
```
$ git clone https://github.com/microsoft/code-completion-eval
$ cd code-completion-eval
$ pip install -e .
```

## Usage

Run the eval.sh in /scripts folder, pass the following parameters into eval.sh:
```
model_url=$1 # model_url
api_key=$2 # api_key
api_type=$3 # "chat" or "fim"
benchmark=$4 # "test", "single-line", "multi-line", "random-span", "random-span-light"
model_output_file=$5 # model output file path
eval_output_file=$6 # console output file path
```

```
$ cd scripts
$ ./eval.sh "model_url" "api_key" "chat" "test" "gpt-4o-mini-test.jsonl" "gpt-4o-mini-test-output.txt"
```