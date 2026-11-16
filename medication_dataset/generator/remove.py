"""
Remove medications from the dataset (they can be added back later with generator.add).

    python -m generator.remove 12           # removes every data file of medication 12
                                            # and moves its knowledge-base file to knowledge_base/spare/
Then re-runs combine, so combined/ contains only the remaining medications.
"""
import os
import shutil
import sys

from . import build, combine


def main(ids):
    for i in ids:
        for sub, ext in (("", "txt"), ("validation", "txt"), ("test", "txt"), ("qa", "jsonl"), ("probes", "jsonl"),
                         ("doc_index", "jsonl")):
            p = os.path.join(build.DATA, sub, f"medication_{i}.{ext}")
            if os.path.exists(p):
                os.remove(p)
        src = os.path.join(build.KB_DIR, f"medication_{i}.json")
        if os.path.exists(src):
            os.makedirs(os.path.join(build.KB_DIR, "spare"), exist_ok=True)
            shutil.move(src, os.path.join(build.KB_DIR, "spare", f"medication_{i}.json"))
        print(f"removed medication_{i}")
    combine.main()


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]])
