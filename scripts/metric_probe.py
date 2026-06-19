import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from bert_score import score as bs
from rag.config import Paths, GenerationConfig
from rag.eval.devset import build_devset
from rag.eval.metric import combined_score
from rag.generator.abstain import ABSTAIN, is_abstain


def main(n=300):
    """Оценить стратегию "отказ на всё" на dev и вывести средние скоры по подгруппам."""
    paths, gcfg = Paths(), GenerationConfig()
    q = pd.read_csv(paths.questions)
    sample = pd.read_csv(paths.sample)
    dev = build_devset(q, sample, n=n)
    refs = dev["answer_new"].fillna("").tolist()
    ref_abstain = np.array([is_abstain(r) for r in refs])
    preds = [ABSTAIN] * len(refs)

    print(f"dev n={len(refs)}  ref_abstain={ref_abstain.sum()}  ref_answerable={(~ref_abstain).sum()}")
    for rescale in (False, True):
        try:
            _, R, _ = bs(preds, refs, lang="ru", rescale_with_baseline=rescale, device="cpu")
            rec = R.tolist()
            sc = np.array([combined_score(r, p, ref, gcfg.len_unit)
                           for r, p, ref in zip(rec, preds, refs)])
            print(f"\nrescale_with_baseline={rescale}")
            print(f"  ALL-ABSTAIN mean score : {sc.mean():.4f}")
            print(f"  correct-abstain (ref=Нет ответа): {sc[ref_abstain].mean():.4f}")
            print(f"  FALSE-abstain  (ref=answerable) : {sc[~ref_abstain].mean():.4f}")
        except Exception as e:
            print(f"\nrescale_with_baseline={rescale}: ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
