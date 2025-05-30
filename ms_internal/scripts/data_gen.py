
import argparse
from human_eval_infilling.data import write_jsonl, read_problems
from chat4completion import call_chat_api
from fim_completion import call_fim_api


def data_gen(model_url, api_key, api_type, benchmark, output_file):

    problems = read_problems(benchmark_name=benchmark)

    num_samples_per_task = 1
    samples = []
    top_n_problems = list(problems.keys())
    for task_id in top_n_problems:
        for _ in range(num_samples_per_task):
            if api_type == "fim":
                sample = dict(
                    task_id=task_id,
                    completion=call_fim_api(
                        model_url, api_key, problems[task_id]["prompt"], problems[task_id]["suffix"])
                )
            else:
                sample = dict(
                    task_id=task_id,
                    completion=call_chat_api(
                        model_url, api_key, problems[task_id]["prompt"], problems[task_id]["suffix"])
                )
            samples.append(sample)

    write_jsonl(output_file, samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FIM evaluation run.")
    parser.add_argument("--model_url", type=str,
                        required=True, help="URL of the model API.")
    parser.add_argument("--api_key", type=str, required=True,
                        help="API key for authentication.")
    parser.add_argument("--api_type", type=str,
                        required=True, help="Type of the API.")
    parser.add_argument("--benchmark", type=str,
                        required=True, help="Benchmark type.")
    parser.add_argument("--output_file", type=str,
                        required=True, help="Output file name.")

    args = parser.parse_args()
    # print args
    print("model_url: ", args.model_url)
    print("api_key: ", args.api_key)
    print("api_type: ", args.api_type)
    print("benchmark: ", args.benchmark)
    print("output_file: ", args.output_file)
    data_gen(args.model_url, args.api_key, args.api_type,
             args.benchmark, args.output_file)
