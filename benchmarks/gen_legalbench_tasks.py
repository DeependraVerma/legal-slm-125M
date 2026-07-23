"""Generate lm-eval-harness YAML task configs for a curated LegalBench subset.

LegalBench (nguha/legalbench, 162 configs, Guha et al. 2023) has no existing
lm-eval-harness integration, so these configs are hand-built here -- but they
use the *official* base_prompt.txt from the LegalBench source repo
(github.com/HazyResearch/legalbench/tasks/<task>/base_prompt.txt) verbatim,
including its baked-in few-shot exemplars, rather than a rephrased prompt --
that's what makes the resulting number a real LegalBench reproduction and
not just "a benchmark with the same name."

Subset selection: all 12 tasks are contract-focused (9 ContractNLI clauses +
contract_qa + consumer_contracts_qa + citation_prediction_classification),
matching this project's own CUAD/LEDGAR training domain, out of LegalBench's
full 162 tasks -- running the entire suite (many of which are free-text
generation tasks needing custom scoring, not simple yes/no classification)
was out of scope for this pass. All are binary Yes/No, short single-clause
inputs (well within the 1,024-token context), and require no processing
beyond simple multiple_choice scoring, unlike the CaseHOLD/SCOTUS length
issues seen earlier.
"""

from __future__ import annotations

from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "legalbench_prompts"
OUT_DIR = Path(__file__).parent / "custom_tasks" / "legalbench"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = [
    "contract_nli_confidentiality_of_agreement",
    "contract_nli_explicit_identification",
    "contract_nli_limited_use",
    "contract_nli_no_licensing",
    "contract_nli_notice_on_compelled_disclosure",
    "contract_nli_permissible_copy",
    "contract_nli_return_of_confidential_information",
    "contract_nli_sharing_with_employees",
    "contract_nli_survival_of_obligations",
    "contract_qa",
    "consumer_contracts_qa",
    "citation_prediction_classification",
]

YAML_TEMPLATE = """\
tag:
  - legalbench
  - legal
task: legalbench_{task}
dataset_path: nguha/legalbench
dataset_name: {task}
output_type: multiple_choice
training_split: null
validation_split: null
test_split: test
doc_to_text: |-
{prompt_indented}
doc_to_target: answer
doc_to_choice: ["Yes", "No"]
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
"""


def indent(text: str, spaces: int = 2) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line else pad.rstrip() for line in text.split("\n"))


def main():
    written = []
    for task in TASKS:
        prompt_path = PROMPT_DIR / f"{task}.txt"
        if not prompt_path.exists():
            print(f"SKIP {task}: prompt file missing")
            continue
        prompt = prompt_path.read_text().rstrip("\n")
        yaml_text = YAML_TEMPLATE.format(task=task, prompt_indented=indent(prompt))
        out_path = OUT_DIR / f"legalbench_{task}.yaml"
        out_path.write_text(yaml_text)
        written.append(out_path.name)
    print(f"wrote {len(written)} task configs to {OUT_DIR}")
    for w in written:
        print(" ", w)


if __name__ == "__main__":
    main()
