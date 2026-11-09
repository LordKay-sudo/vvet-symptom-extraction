"""Run the real LLM over a hand-labelled set and print accuracy numbers.

Usage (needs LLM_API_KEY):  python -m eval.run_eval [path/to/samples.jsonl]

samples.jsonl holds one {"message": ..., "expected": {sign_id: value}} per line.
The bundled file only has the five sample messages from the brief, labelled by hand,
so treat the numbers as a smoke test, not a real accuracy measurement.
"""

import json
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from app.escalation import decide_escalation
from app.extractor import Extractor
from app.llm import OpenAICompatibleClient
from app.models import Finding
from app.signs import SignId


def main(path: Path) -> None:
    load_dotenv()
    extractor = Extractor(OpenAICompatibleClient.from_env())
    samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    tp, fp, fn = Counter(), Counter(), Counter()
    value_matches = value_total = escalation_correct = 0

    for sample in samples:
        expected: dict[str, str] = sample["expected"]
        findings = extractor.extract(sample["message"])
        predicted = {f.sign_id.value: f.value for f in findings}

        for sign in SignId:
            s = sign.value
            if s in predicted and s in expected:
                tp[s] += 1
                value_total += 1
                value_matches += predicted[s] == expected[s]
            elif s in predicted:
                fp[s] += 1
            elif s in expected:
                fn[s] += 1

        expected_findings = [Finding(sign_id=s, value=v, quote="-") for s, v in expected.items()]
        escalation_correct += decide_escalation(findings)[0] == decide_escalation(expected_findings)[0]

        if predicted != expected:
            print(f"MISMATCH: {sample['message']!r}\n  expected {expected}\n  got      {predicted}")

    print(f"\n{'sign':<18}{'precision':>10}{'recall':>8}")
    for sign in SignId:
        s = sign.value
        if tp[s] + fp[s] + fn[s] == 0:
            continue
        precision = tp[s] / (tp[s] + fp[s]) if tp[s] + fp[s] else 0.0
        recall = tp[s] / (tp[s] + fn[s]) if tp[s] + fn[s] else 0.0
        print(f"{s:<18}{precision:>10.2f}{recall:>8.2f}")

    print(f"\nvalue agreement on detected signs: {value_matches}/{value_total}")
    print(f"escalation decision correct:       {escalation_correct}/{len(samples)}")


if __name__ == "__main__":
    default = Path(__file__).with_name("samples.jsonl")
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else default)
