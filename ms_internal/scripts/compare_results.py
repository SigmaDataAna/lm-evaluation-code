import json
from collections import defaultdict
import html
import re
import ast
import tiktoken
import os
from datetime import datetime
import argparse

def read_results(results_file):
    """Read results from a JSONL file and extract metrics if present."""
    results = {}
    metrics = {}

    with open(results_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'task_id' in data:
                # This is a result entry
                results[data['task_id']] = {
                    'passed': data.get('passed', False),
                    'result': data.get('result', ''),
                    'completion': data.get('completion', ''),
                    'full_solution': data.get('full_solution', ''),
                    'num_tokens': count_tokens(data.get('completion', ''))
                }
            elif 'pass@k' in data:
                # This is a metrics entry
                metrics = data

    return results, metrics

def create_comparison_report(results_file1, results_file2, output_html_file):
    # Read both result files with metrics
    model1_results, model1_metrics = read_results(results_file1)
    model2_results, model2_metrics = read_results(results_file2)

    # Calculate statistics
    model1_stats = calculate_statistics(model1_results)
    model2_stats = calculate_statistics(model2_results)

    # Add metrics to stats
    model1_stats['metrics'] = model1_metrics
    model2_stats['metrics'] = model2_metrics

    # Create HTML content
    html_content = """
    <html>
    <head>
        <style>
            .container { display: flex; margin-bottom: 20px; }
            .solution { flex: 1; padding: 10px; margin: 5px; }
            .passed { background-color: #90EE90; }
            .failed { background-color: #FFB6C1; }
            pre { 
                white-space: pre-wrap; 
                word-wrap: break-word;
                counter-reset: line;
                padding-left: 3.5em;
                position: relative;
            }
            .task-id { font-weight: bold; margin-top: 20px; }
            .completion-highlight { background-color: #FFFF00; }
            .error-details { 
                margin: 10px 0;
                padding: 8px;
                background-color: #FFF3F3;
                border-left: 4px solid #FF0000;
                font-family: monospace;
            }
            .line-number {
                position: absolute;
                left: 0;
                width: 2.5em;
                color: #666;
                text-align: right;
                padding-right: 0.5em;
                user-select: none;
                border-right: 1px solid #ddd;
                margin-right: 0.5em;
            }
            .stats-container {
                margin: 20px;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 5px;
            }
            .stats-table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            .stats-table th, .stats-table td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            .stats-table th {
                background-color: #f1f1f1;
            }
            .stats-table tr:hover {
                background-color: #f5f5f5;
            }
            .stats-table td code {
                background-color: #f5f5f5;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: monospace;
            }
            .stats-table td:first-child {
                min-width: 200px;
            }
            .task-list {
                margin-top: 10px;
                padding: 10px;
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 0.9em;
            }
            .task-ids {
                max-height: 200px;
                overflow-y: auto;
                padding: 5px;
                background-color: white;
                border: 1px solid #eee;
                font-family: monospace;
                font-size: 0.9em;
            }
            .toggle-btn {
                margin-left: 10px;
                padding: 2px 5px;
                font-size: 0.8em;
                cursor: pointer;
                background-color: #f0f0f0;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
            .toggle-btn:hover {
                background-color: #e0e0e0;
            }
            .error-row td {
                padding: 10px;
                vertical-align: top;
            }
        </style>
    </head>
    <body>
    """

    # Add statistics table
    model1_name = results_file1.split('/')[-1]
    model2_name = results_file2.split('/')[-1]
    html_content += create_stats_table(model1_stats, model2_stats, model1_name, model2_name)

    # Add model names as headers
    html_content += f"<div class='container'><div class='solution'><h2>{model1_name}</h2></div><div class='solution'><h2>{model2_name}</h2></div></div>"

    # Add solutions side by side
    for task_id in sorted(set(model1_results.keys()) | set(model2_results.keys())):
        html_content += f"<div class='task-id'>Task ID: {task_id}</div>"
        html_content += "<div class='container'>"

        # Model 1 solution
        if task_id in model1_results:
            result = model1_results[task_id]
            html_content += format_solution_with_test_results(result)
        else:
            html_content += "<div class='solution'>No solution</div>"

        # Model 2 solution
        if task_id in model2_results:
            result = model2_results[task_id]
            html_content += format_solution_with_test_results(result)
        else:
            html_content += "<div class='solution'>No solution</div>"

        html_content += "</div>"

    html_content += "</body></html>"

    # Write the HTML file
    with open(output_html_file, 'w') as f:
        f.write(html_content)

    print(f"Report generated: {output_html_file}")

def format_solution_with_highlight(full_solution, completion):
    # Escape both strings for HTML
    full_solution = html.escape(full_solution)
    completion = html.escape(completion)

    # Replace the completion in the full solution with a highlighted version
    highlighted_solution = full_solution.replace(
        completion, 
        f'<span class="completion-highlight">{completion}</span>'
    )

    # Add line numbers
    lines = highlighted_solution.split('\n')
    numbered_lines = [f'<span class="line-number">{i + 1}</span>{line}' for i, line in enumerate(lines)]
    return '\n'.join(numbered_lines)

def format_solution_with_test_results(result):
    # print("Debug - Result dict:", result)
    status_class = 'passed' if result['passed'] else 'failed'
    highlighted_solution = format_solution_with_highlight(
        result['full_solution'], 
        result['completion']
    )

    # Add result details for failed cases
    result_details = ""
    if not result['passed'] and result.get('result'):
        # print("Debug - Adding error details:", result['result'])
        result_details = f"""
        <div class='error-details'>
            Error: {result['result']}
        </div>
        """

    return f"""
    <div class='solution {status_class}'>
        <div>Status: {'✅ Passed' if result['passed'] else '❌ Failed'}</div>
        {result_details}
        <pre>{highlighted_solution}</pre>
    </div>
    """

def normalize_error_message(error_msg, full_solution=None):
    """Normalize error messages with granular syntax error classification"""
    if not error_msg or error_msg.isspace():
        return "unknown error"

    if not error_msg.startswith('failed: '):
        return error_msg

    # Remove the 'failed: ' prefix
    error_msg = error_msg[len('failed: '):]

    # First, normalize common patterns by removing line numbers
    common_patterns = [
        (r'expected an indented block after .+ statement on line \d+', 'expected an indented block after statement'),
        (r'invalid syntax \(.+, line \d+\)', 'invalid syntax'),
        (r'unexpected indent \(.+, line \d+\)', 'unexpected indent'),
        (r'unindent does not match any outer indentation level \(.+, line \d+\)', 'unindent does not match any outer indentation level'),
        (r'name .+ is not defined', 'name error: undefined variable'),
        (r'cannot assign to .+ \(.+, line \d+\)', 'cannot assign to expression'),
        (r'\(.+, line \d+\)', ''),  # Remove file and line info from other errors
        (r'EOL while scanning string literal \(.+\)', 'EOL while scanning string literal'),
        (r'EOF while scanning triple-quoted string literal \(.+\)', 'EOF while scanning triple-quoted string literal'),
        (r'unexpected EOF while parsing \(.+\)', 'unexpected EOF while parsing'),
    ]

    normalized_msg = error_msg.lower()
    for pattern, replacement in common_patterns:
        normalized_msg = re.sub(pattern, replacement, normalized_msg)

    # Now classify syntax errors using AST if we have the full solution
    if 'syntax' in normalized_msg and full_solution:
        try:
            ast.parse(full_solution)
        except SyntaxError as e:
            error_categories = {
                "Unclosed Structure": ["EOL while scanning string literal", "unexpected EOF while parsing"],
                "Misplaced or Missing Keywords": {
                    "Missing Colon": {
                        "Control Structures": ["if", "elif", "else"],
                        "Loop Structures": ["for", "while"],
                        "Function Definition": ["def"],
                        "Lambda Function": ["lambda"],
                        "Class Definition": ["class"],
                        "Try-Except-Finally": ["try", "except", "finally"]
                    }
                },
                "Indentation Errors": ["unexpected indent", "unindent does not match", "expected an indented block"],
                "Unexpected Characters or Delimiters": ["invalid character", "unexpected character"],
                "Wrong Delimiters": ["missing parentheses", "unclosed parenthesis", "unclosed bracket"]
            }

            # Get the problematic line
            code_lines = full_solution.split('\n')
            error_line = code_lines[e.lineno - 1] if e.lineno <= len(code_lines) else ""
            error_msg_lower = e.msg.lower()

            # Check for specific structural issues first
            if any(char in error_line for char in '([{'):
                opens = sum(error_line.count(c) for c in '([{')
                closes = sum(error_line.count(c) for c in ')]}')
                if opens != closes:
                    return "syntax error: wrong delimiters - unmatched brackets/parentheses"

            # Check for string literal issues
            if error_line.count('"') % 2 == 1 or error_line.count("'") % 2 == 1:
                return "syntax error: unclosed structure - unterminated string"

            # Check for indentation issues
            if error_line.lstrip() != error_line:
                indent_level = len(error_line) - len(error_line.lstrip())
                if indent_level % 4 != 0:
                    return "syntax error: indentation errors - incorrect indentation level"

            # Check for missing colon with detailed categorization
            if ":" not in error_line and error_line.strip():
                error_line = error_line.strip()
                for structure_type, keywords in error_categories["Misplaced or Missing Keywords"]["Missing Colon"].items():
                    if any(keyword in error_line.split() for keyword in keywords):
                        return f"syntax error: missing colon - {structure_type.lower()}"

            # Check for other keyword-related issues
            if any(pattern in error_msg_lower for pattern in ["def", "class", "return", "if", "for", "while"]):
                # Check for function/class definition issues
                if "def" in error_line or "class" in error_line:
                    if "(" not in error_line or ")" not in error_line:
                        return "syntax error: wrong delimiters - missing function/class parentheses"

                # Check for return statement issues
                if "return" in error_line and "'return' outside function" in error_msg_lower:
                    return "syntax error: misplaced or missing keywords - unexpected return"

            # Classify based on error categories
            for category, patterns in error_categories.items():
                if isinstance(patterns, dict):
                    # Handle subcategories
                    for subcategory, subpatterns in patterns.items():
                        if isinstance(subpatterns, dict):
                            # Handle nested subcategories (like Missing Colon cases)
                            for sub_subcategory, keywords in subpatterns.items():
                                if any(keyword in error_line.split() for keyword in keywords):
                                    return f"syntax error: {category.lower()} - {subcategory.lower()} in {sub_subcategory.lower()}"
                        elif any(pattern in error_msg_lower for pattern in subpatterns):
                            return f"syntax error: {category.lower()} - {subcategory.lower()}"
                elif any(pattern in error_msg_lower for pattern in patterns):
                    return f"syntax error: {category.lower()}"

    normalized_msg = normalized_msg.strip()
    return normalized_msg if normalized_msg else "unknown error"

def calculate_statistics(results_dict):
    stats = {
        'total': 0,
        'passed': 0,
        'max_tokens': 0,
        'total_tokens': 0,
        'errors': defaultdict(lambda: {'count': 0, 'task_ids': []})
    }

    for task_id, result in results_dict.items():
        stats['total'] += 1
        tokens = result['num_tokens']
        stats['total_tokens'] += tokens
        stats['max_tokens'] = max(stats['max_tokens'], tokens)
        if result['passed']:
            stats['passed'] += 1
        elif 'result' in result:
            error_msg = result['result'].lower()
            normalized_error = normalize_error_message(error_msg, result.get('full_solution'))
            stats['errors'][normalized_error]['count'] += 1
            stats['errors'][normalized_error]['task_ids'].append(task_id)

    return stats

def create_stats_table(model1_stats, model2_stats, model1_name, model2_name):
    # Add pass@k metrics to the table if available
    pass_k_rows = ""
    if model1_stats.get('metrics') or model2_stats.get('metrics'):
        # Look for pass@k keys specifically
        all_ks = set()
        for metrics in [model1_stats.get('metrics', {}), model2_stats.get('metrics', {})]:
            all_ks.update(k for k in metrics.keys() if k.startswith('pass@'))

        for k in sorted(all_ks):
            model1_value = model1_stats.get('metrics', {}).get(k, 'N/A')
            model2_value = model2_stats.get('metrics', {}).get(k, 'N/A')
            if isinstance(model1_value, float):
                model1_value = f"{model1_value:.1%}"
            if isinstance(model2_value, float):
                model2_value = f"{model2_value:.1%}"

            pass_k_rows += f"""
            <tr>
                <td><strong>{k}</strong></td>
                <td>{model1_value}</td>
                <td>{model2_value}</td>
            </tr>
            """

    # Create the metrics table
    metrics_table = f"""
    <div class="stats-container">
        <h3>Aggregate Metrics</h3>
        <table class="stats-table">
            <tr>
                <th>Metric</th>
                <th>{model1_name}</th>
                <th>{model2_name}</th>
            </tr>
            {pass_k_rows}
            <tr>
                <td><strong>Total Solutions</strong></td>
                <td>{model1_stats['total']}</td>
                <td>{model2_stats['total']}</td>
            </tr>
            <tr>
                <td><strong>Max Tokens</strong></td>
                <td>{model1_stats['max_tokens']:,}</td>
                <td>{model2_stats['max_tokens']:,}</td>
            </tr>
            <tr>
                <td><strong>Average Tokens</strong></td>
                <td>{model1_stats['total_tokens']/model1_stats['total']:.1f}</td>
                <td>{model2_stats['total_tokens']/model2_stats['total']:.1f}</td>
            </tr>
            <tr>
                <td><strong>Passed</strong></td>
                <td>{model1_stats['passed']} ({(model1_stats['passed']/model1_stats['total']*100):.1f}%)</td>
                <td>{model2_stats['passed']} ({(model2_stats['passed']/model2_stats['total']*100):.1f}%)</td>
            </tr>
        </table>
    </div>
    """

    # Create the errors table
    all_error_types = set(model1_stats['errors'].keys()) | set(model2_stats['errors'].keys())
    error_rows = []

    for error_type in sorted(all_error_types):
        if not error_type:  # Skip empty error types
            continue

        model1_data = model1_stats['errors'].get(error_type, {'count': 0, 'task_ids': []})
        model2_data = model2_stats['errors'].get(error_type, {'count': 0, 'task_ids': []})

        model1_count = model1_data['count']
        model2_count = model2_data['count']

        model1_tasks = '<br>'.join(sorted(model1_data['task_ids']))
        model2_tasks = '<br>'.join(sorted(model2_data['task_ids']))

        error_rows.append(f"""
            <tr class="error-row">
                <td><code>{error_type}</code></td>
                <td>
                    {model1_count} ({(model1_count/model1_stats['total']*100):.1f}%)
                    <button onclick="toggleTasks('{error_type}-model1')" class="toggle-btn">
                        Show/Hide Tasks
                    </button>
                    <div id="tasks-{error_type}-model1" class="task-list" style="display: none;">
                        <div class="task-ids">{model1_tasks}</div>
                    </div>
                </td>
                <td>
                    {model2_count} ({(model2_count/model2_stats['total']*100):.1f}%)
                    <button onclick="toggleTasks('{error_type}-model2')" class="toggle-btn">
                        Show/Hide Tasks
                    </button>
                    <div id="tasks-{error_type}-model2" class="task-list" style="display: none;">
                        <div class="task-ids">{model2_tasks}</div>
                    </div>
                </td>
            </tr>
        """)

    errors_table = f"""
    <div class="stats-container">
        <h3>Error Analysis</h3>
        <table class="stats-table">
            <tr>
                <th>Error Type</th>
                <th>{model1_name}</th>
                <th>{model2_name}</th>
            </tr>
            {''.join(error_rows)}
        </table>
    </div>
    """

    # Add JavaScript for toggling task lists
    js_code = """
    <script>
    function toggleTasks(id) {
        const taskList = document.getElementById('tasks-' + id);
        taskList.style.display = taskList.style.display === 'none' ? 'block' : 'none';
    }
    </script>
    """

    return metrics_table + errors_table + js_code

def count_tokens(text):
    """Count tokens using GPT tokenizer"""
    try:
        # Using cl100k_base encoder which is used by GPT-3.5/4 models
        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))
    except Exception:
        # Fallback if tokenizer fails
        return len(text.split())

# Usage example:
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results1", help="Path to the first results file")
    ap.add_argument("--results2", help="Path to the second results file")
    args = ap.parse_args()
    
    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    create_comparison_report(
        args.results1,
        args.results2,
        f"comparison_report_{now}.html"
    )
