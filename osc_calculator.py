# ==============================================================================
# Filename: osc_calculator.py
# Description: Calculates Objective Syntactic Correctness (OSC) for LLM outputs.
# Requires: pip install tree_sitter tree_sitter_typescript tree_sitter_javascript pandas
# ==============================================================================

import pandas as pd
import re
from tree_sitter import Language, Parser
import tree_sitter_typescript as tsts
import tree_sitter_javascript as tsjs
import ast

# Initialize parsers for TypeScript (TSX) and JavaScript
TSX_LANGUAGE = Language(tsts.language_tsx())
JS_LANGUAGE = Language(tsjs.language())
tsx_parser = Parser(TSX_LANGUAGE)
js_parser = Parser(JS_LANGUAGE)

# Define security patterns to flag
SECURITY_PATTERNS = [
    (r"\beval\s*\(", "eval_on_input"),
    (r"\bFunction\s*\(", "function_constructor"),
    (r"innerHTML\s*=", "innerHTML_assignment"),
    (r"dangerouslySetInnerHTML", "react_dangerous_html"),
    (r"document\.write\s*\(", "document_write"),
    (r"(api[_-]?key|password|secret)\s*=\s*['\"][^'\"]+['\"]", "hardcoded_credential"),
    (r"SELECT\s+.*\+\s*", "sql_string_concat"),
]

def check_syntax(code, lang):
    """Returns 1 if syntax is valid, 0 otherwise."""
    try:
        if lang == "python":
            ast.parse(code)
            return 1
        elif lang in ("typescript", "tsx"):
            tree = tsx_parser.parse(bytes(code, "utf8"))
            if tree.root_node.has_error:
                return 0
            return 1
        elif lang == "javascript":
            tree = js_parser.parse(bytes(code, "utf8"))
            if tree.root_node.has_error:
                return 0
            return 1
        elif lang in ("css", "html", "php"):
            # Basic bracket matching heuristic for non-JS languages
            stack = []
            pairs = {")": "(", "]": "[", "}": "{"}
            for ch in code:
                if ch in "([{": stack.append(ch)
                elif ch in pairs:
                    if not stack or stack.pop() != pairs[ch]: return 0
            return 1 if not stack else 0
        return 1
    except Exception:
        return 0

def check_security(code):
    """Returns a list of flagged security patterns."""
    flags = []
    for pattern, name in SECURITY_PATTERNS:
        if re.search(pattern, str(code), re.IGNORECASE):
            flags.append(name)
    return flags

def main():
    # Load the raw data
    file_path = "raw_outputs.csv"
    df = pd.read_csv(file_path)

    results = []
    for index, row in df.iterrows():
        code_text = str(row["code"])
        lang = str(row["lang"]).strip().lower()
        
        spr = check_syntax(code_text, lang)
        flags = check_security(code_text)
        sfr = 1 if len(flags) > 0 else 0
        
        # Calculate composite OSC: SPR * (1 - SFR)
        osc = spr * (1 - sfr)
        
        results.append({
            "model": row["model"],
            "prompt": row["prompt"],
            "run": row["run"],
            "SPR": spr,
            "SFR": sfr,
            "OSC": osc,
            "Security_Flags": ";".join(flags)
        })

    # Create DataFrame and save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv("detailed_results.csv", index=False)

    # Aggregate by model
    summary = results_df.groupby("model").agg(
        SPR_Percent=("SPR", lambda x: round(x.mean() * 100, 1)),
        SFR_Percent=("SFR", lambda x: round(x.mean() * 100, 1)),
        OSC_Percent=("OSC", lambda x: round(x.mean() * 100, 1))
    ).reset_index()

    print("=== OSC Summary ===")
    print(summary.to_string(index=False))
    summary.to_csv("osc_summary.csv", index=False)

if __name__ == "__main__":
    main()
