"""Patch A3VLM's eval_affordance_v2.py so that (1) each output carries our ``key`` from the
question JSON (the script only records image + prompt, which is not unique per element),
(2) the run-resumption block that drops every question sharing an already-answered IMAGE is
disabled (we ask several questions per image), and (3) the dataset is not shuffled/subsampled.
Model, prompt format and generation settings are untouched. Idempotent.

  python patch_eval.py <path to eval_affordance_v2.py>
"""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()
if "SF3D_PATCHED" in s:
    print("already patched")
    sys.exit(0)

edits = [
    # collate: pass keys through
    ("    annotations = [_['annotation'] for _ in batches]\n",
     "    annotations = [_['annotation'] for _ in batches]\n    keys = [_.get('key') for _ in batches]\n"),
    ("    return input_image, question_ids, questions, annotations, image_paths\n",
     "    return input_image, question_ids, questions, annotations, image_paths, keys\n"),
    # dataset: never subsample, never drop by image path
    ("        if len(self.test) > sampled_num:\n            # first shuffle, then sample\n            random.shuffle(self.test)\n            sampled_num = min(len(self.test), sampled_num)\n            self.test = self.test[:sampled_num]\n",
     "        if False:  # SF3D_PATCHED: evaluate every question, in order\n            pass\n"),
    ("        if result is not None:\n            # when image_path and question is the same, contine\n",
     "        if False:  # SF3D_PATCHED: resumption by image path drops sibling questions\n"),
    ("            'image': image,\n            \"image_path\": image_path\n        }\n",
     "            'image': image,\n            \"image_path\": image_path,\n            'key': data.get('key')\n        }\n"),
    # loop: unpack and record the key
    ("        for image, question_ids, _prompt, annotations, image_paths in tqdm(dataloader):\n",
     "        for image, question_ids, _prompt, annotations, image_paths, keys in tqdm(dataloader):\n"),
    ("                for question_id, answer, annotation, question, image_path in zip(question_ids, results,\n                                                        annotations, _prompt, image_paths):\n",
     "                for question_id, answer, annotation, question, image_path, key in zip(question_ids, results,\n                                                        annotations, _prompt, image_paths, keys):\n"),
    ("                        \"image\": image_path,\n                        \"fail\": failed_flag\n                    })\n",
     "                        \"image\": image_path,\n                        \"fail\": failed_flag,\n                        \"key\": key\n                    })\n"),
]
for old, new in edits:
    assert s.count(old) == 1, old
    s = s.replace(old, new)
p.write_text(s)
print("patched", p)
