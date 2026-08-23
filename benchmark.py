from pathlib import PurePath
import os
import sys
import json
from datetime import datetime


def run_benchmark(model_dir: str, test_option: str) -> None:
    import evalution as eval

    # missing: LiveBench 0831, MATH, MultiPL-E
    TEST_OPTIONS = {
        "mmlu_pro": eval.benchmarks.mmlu_pro(apply_chat_template=True, batch_size=4),
        "gsm8k": eval.benchmarks.gsm8k(apply_chat_template=True, batch_size=4),
        "mmlu_redux": eval.benchmarks.mmlu_redux(apply_chat_template=True, batch_size=4),
        "GPQA": eval.benchmarks.GPQA(apply_chat_template=True, batch_size=4),
        "HumanEval": eval.benchmarks.HumanEval(apply_chat_template=True, batch_size=4),
        "MBPP": eval.benchmarks.MBPP(apply_chat_template=True, batch_size=4),
        "LiveCodeBench": eval.benchmarks.LiveCodeBench(apply_chat_template=True, batch_size=4),
        "IFEval": eval.benchmarks.IFEval(apply_chat_template=True, batch_size=4),
    }

    if not test_option:
        test_option = ",".join(TEST_OPTIONS.keys())

    model_cfg = eval.Model(path=model_dir, label=PurePath(model_dir).parts[-1])
    engine = eval.GPTQModel(backend="auto", device="cuda:0")

    run = eval.run(
        model=model_cfg,
        engine=engine,
        tests=[TEST_OPTIONS[opt.strip()] for opt in test_option.split(",")],
    )

    result = {
        "model": model_cfg.label,
        "engine": run.engine,
        "result": [test.to_dict() for test in run.tests]
    }
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 benchmark.py <model_dir> <test_option>")
        print("test_option (optional - comma separated): mmlu_pro, gsm8k, mmlu_redux, GPQA, HumanEval, MBPP, LiveCodeBench, IFEval")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    test_option = sys.argv[2] if len(sys.argv) > 2 else None

    result = run_benchmark(model_dir, test_option)
    result_dir = f"benchmark/{result['model']}"
    os.makedirs(result_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_file = os.path.join(result_dir, f"{timestamp}.json")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=4)