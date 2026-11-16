"""
Add medications to the dataset.

    python -m generator.add 13 14        # take medications 13 and 14 from knowledge_base/spare/
    python -m generator.add 21           # a NEW medication: first create knowledge_base/medication_21.json
                                         # (copy knowledge_base/TEMPLATE.json and fill it in)

Builds the files of the added medications only (all others stay exactly as they are) and re-runs combine.
"""
import os
import shutil
import sys

from . import build, combine


def main(ids):
    for i in ids:
        dst = os.path.join(build.KB_DIR, f"medication_{i}.json")
        src = os.path.join(build.KB_DIR, "spare", f"medication_{i}.json")
        if not os.path.exists(dst):
            if not os.path.exists(src):
                sys.exit(f"medication_{i}.json not found in knowledge_base/ or knowledge_base/spare/")
            shutil.move(src, dst)
    all_meds = build.load_kb(include_spare=True)
    names = [m["name"].lower() for m in all_meds]
    assert len(names) == len(set(names)), "two medications have the same name"
    for m in build.load_kb():
        if m["id"] in ids:
            build.build_one(m, all_meds)
    combine.main()


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]])
