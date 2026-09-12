# A3VLM + 3DOI on SF3D (single-shot) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain A3VLM (CoRL'24, SPHINX-1k 13B) and 3DOI (ICCV'23, SAM ViT-B) faithfully on the SF3D train split with their own released code and recipes, ONCE each, and score them on our 5,088-sample test split with the shared JSONL + `tools/baselines_sf3d/score_predictions.py` protocol from the first baseline plan.

**Architecture:** Same shape as `docs/superpowers/plans/2026-09-12-sf3d-external-baselines.md`: upstream repo cloned at a pinned commit, patched only where the data forces it; one converter per method writes the native training format from our LMDB; one export script per method turns their predictions into the shared JSONL. Because every expensive run happens once, each pipeline is proven end to end at small scale on a cheap 2×A100 pod (env, 20-50 training iterations, checkpoint save, resume, eval/export) before the big pod is created, and the big runs checkpoint to the volume and resume.

**Tech Stack:** LLaMA2-Accessory / SPHINX (torch 2.0.1 cu117/cu118, fairscale model parallel 2, FSDP sdp, bf16), 3DOI monoarti (SAM ViT-B, hydra + accelerate, fp16), RunPod A100 80GB pods in a US datacenter with a new 150 GB network volume (no A100s in EU-RO-1), our `tools/baselines_sf3d/common.py`, `score_predictions.py`.

**Spec:** the conversation of 2026-09-12 (user: "pick the 2 best expensive baselines", "run each one time", "use the runpod gpus").

## Global Constraints

- Same ownership rules as the first plan (never edit `model/`, `datasets/`, `train_*_better.py`, the peer's experiment dirs/configs; own dirs `tools/baselines_sf3d/`, `runpod/baselines/`, `experiments/baselines_sf3d/`, `config/baselines/`).
- **Single shot:** no expensive pod is created before its pipeline passed the smoke gate (Task 5 / Task 9) and the frozen recipe + cost was shown to the user. Big runs checkpoint at least hourly to the volume; chain scripts resume from the last checkpoint; a watcher deletes the pod only on CHAIN_DONE. Never relaunch "with a fix" without asking.
- Data staged for the US volume is built on the EU side under `/workspace/datasets/baselines/stage_us/` (CPU jobs on the dev pod, each under ~30 min or announced to the peer) and copied pod-to-pod with rsync.
- Test split, metrics, JSONL schema: exactly as in the first plan. Prediction lines carry `axis_cam`, `origin_cam`, `type`, `mask_rle`, `matched`, `score`.
- Budget targets: A3VLM 8×A100 SXM ~10 h (~$150 incl. staging), 3DOI 4×A100 (their epoch count measured in the smoke; cost shown before launch), smokes ~$15.

---

## File map

| file | responsibility |
|---|---|
| `tools/baselines_sf3d/sf3d_to_a3vlm.py` | LMDB -> square-padded 448-px JPEGs + A3VLM VQA JSONs (DET, REC-Link by description, REG-Joint by box, and our REG-Joint-by-description), per-image depth-normalisation sidecar. |
| `tools/baselines_sf3d/a3vlm_preds_to_jsonl.py` | their eval JSON (answers) -> shared JSONL (parse `<axis>type</axis>[x0,y0,z0,x1,y1,z1]` and the 8-vertex box, un-normalise with the sidecar + K, box -> filled 2D mask). |
| `tools/baselines_sf3d/sf3d_to_3doi.py` | LMDB -> 3DOI `images/` + `3doi_sf3d/data_{train,val,test}.pt` + omnidata-style depth tree (uint16 depth*512, mask_valid, point_info fov), 768×1024, portrait frames rolled. |
| `tools/baselines_sf3d/threedoi_export.py` | runs their SAM model on the test frames with GT element points as queries, decodes the 2D line, lifts it with the input depth (plane fit over the predicted mask), writes the shared JSONL. |
| `runpod/baselines/us/pods.sh` | volume + pod helpers for the US datacenter (A100 SXM 2×/4×/8×). |
| `runpod/baselines/a3vlm/{setup_env.sh,stage.sh,chain.sh}` | env (torch 2.0.1 cu118 image), SPHINX-1k download from HF, data rsync, `torchrun --nproc_per_node 8 main_finetune.py …`, eval, export. |
| `runpod/baselines/threedoi/{setup_env.sh,chain.sh}` | env (torch 2.x + SAM ViT-B weights), `accelerate launch --num_processes 4 train.py --config-name sam_sf3d`, export. |
| `config/baselines/{a3vlm_sf3d.yaml,3doi_sf3d.yaml}` | frozen recipes (documentation copies). |
| `tests/baselines_sf3d/test_a3vlm_convert.py`, `test_3doi_convert.py`, `test_a3vlm_preds.py`, `test_3doi_export.py` | unit tests incl. GT round-trips. |
| `experiments/baselines_sf3d/20260913_a3vlm/`, `20260913_3doi/` | notes, config, metrics. |

---

### Task 1: US volume + 2×A100 staging/smoke pod

**Files:** Create `runpod/baselines/us/pods.sh` (thin wrapper: `EXT_DC` chosen from `runpodctl datacenter list` among US DCs with `NVIDIA A100-SXM4-80GB` stock, `EXT_IMAGE=runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04` for A3VLM pods and `runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04` for 3DOI pods; `volume <name> 150`, `gpu <pod> <count>` passing `--gpu-count`).

- [ ] Step 1: `runpodctl datacenter list` → pick a US DC that lists A100 SXM stock and supports volumes; create volume `bl-us` (150 GB) there; record id in `experiments/baselines_sf3d/pods.md`.
- [ ] Step 2: create pod `bl-us-smoke` (2×A100 SXM, cu118 image) on it; verify `nvidia-smi` shows 2 GPUs; `mkdir -p /workspace/{data,ckpt,repos,runs,logs}`.
- [ ] Step 3: start the SPHINX-1k download on it in the background: `huggingface-cli download Alpha-VLLM/LLaMA2-Accessory --include "finetune/mm/SPHINX/SPHINX-1k/*" --local-dir /workspace/ckpt/sphinx` (39.8 GB); and pre-clone `facebookresearch/dinov2` into `$TORCH_HOME/hub`.
- [ ] Step 4: commit `runpod/baselines/us/pods.sh` + pods.md.

### Task 2: `sf3d_to_a3vlm.py` + tests

**Interfaces / contract (from the A3VLM data generator, `data_gen/vqa_task_construction.py`, `point_render.py`, `partnet_label.py`):**
- Image: our frame padded to a centred square with the CLIP mean colour (`[0.48145466, 0.4578275, 0.40821073]*255`), saved as JPEG at 448×448 under `stage_us/a3vlm/images/<visit>_<frameidx>.jpg`; all normalised coordinates are w.r.t. the padded square (u,v in [0,1]).
- Per image depth normalisation: `d_min, d_max` = min/max of valid input depth (metres, zeros excluded) over the ORIGINAL frame; `d = (z - d_min)/(d_max - d_min + 1e-6)` clipped to [0,1]; sidecar `stage_us/a3vlm/meta_<split>.json`: `{image_name: {"key_frame", "wh", "pad": [x0, y0, S], "K": 3x3, "d_min", "d_max", "rotated": false}}`.
- Projection (OpenCV camera, no sign flip): `u = (fx x/z + cx + pad_x0)/S`, `v = (fy y/z + cy + pad_y0)/S`.
- 3D box of an element: axis-aligned box in the camera frame around the element's back-projected mask pixels (input depth; 2nd–98th percentile per axis to drop splat outliers; min extent 0.02 m); 8 vertices ordered as `BBox3D.get_points()` (copy the order from `point_render.py:138-153` into the converter with a comment), each serialised `[u,v,d]` with `"{:.2f}"`.
- Axis endpoints: revolute: `q* ± 0.5·E·n` with `q*` = foot of the element centroid on the GT axis, `E` = element extent along `n` (min 0.10 m); prismatic: `c ± 0.5·E·n` through the centroid. Serialised `[x0,y0,z0,x1,y1,z1]` in the same (u,v,d) space.
- Task JSONs (all `type: 'image_text'`, `conversations` single turn, LLaVA style, `image` = absolute pod path `/workspace/data/a3vlm/images/...`):
  1. `det_<split>.json`: "Detect all manipulable object parts and provide their 3D bounding boxes." → `<box>{label}</box>[[u,v,d]×8]` per element, concatenated in mask-area order.
  2. `rec_desc_<split>.json`: "Please provide the 3D bounding box of the region this sentence describes: {description}" → `[[u,v,d]×8]`.
  3. `reg_joint_box_<split>.json` (their REG-Joint): "Please provide the joint's type and its 3D axis linked to the object part [[u,v,d]×8]." → `<axis>{revolute|prismatic}</axis>[x0,y0,z0,x1,y1,z1]`.
  4. `reg_joint_desc_<split>.json` (our extension, documented): "Please provide the joint's type and its 3D axis linked to the object part described by: {description}" → same answer format.
  Train = splits.train + bvalid (A3VLM has no validation cadence; we keep bvalid out only for the smoke sanity); test JSONs carry `gpt: null` plus an `annotation` field with our key.
- Data YAML `configs/a3vlm_sf3d.yaml`: `META:` the four train JSONs, `ratio: 1`, `type: 'image_text'`.
- CLI: `python tools/baselines_sf3d/run.py tools/baselines_sf3d/sf3d_to_a3vlm.py --out /workspace/datasets/baselines/stage_us/a3vlm --workers 32 [--limit-frames N]`.

- [ ] Step 1: failing tests: (a) `project_uv` on a synthetic K and pad gives expected values; (b) a synthetic record's revolute axis endpoints are symmetric about the foot point and lie on the GT line; (c) GT round trip: `decode_axis(serialise(axis)) ≈ axis` within 0.02 m / 1° when d_max − d_min = 3 m; (d) the JSON answer strings match the regexes used by `a3vlm_preds_to_jsonl.py`.
- [ ] Step 2: implement; run tests on the dev pod.
- [ ] Step 3: 50-frame smoke on the dev pod (`--limit-frames 50`); open one sample of each task and check by eye that the box hull covers the mask in the padded image (render one debug PNG per task into `viz/20260913_a3vlm_vqa_check/`).
- [ ] Step 4: full conversion on the dev pod (39k frames, ~10 min); report counts per task JSON; commit code + tests.

### Task 3: `a3vlm_preds_to_jsonl.py` + tests

- Parses their eval output list (`{answer, question, image, annotation}`), keys by our `annotation` key; for REG-Joint answers parses type + 6 numbers; for REC answers parses the 8 vertices; un-normalises with the sidecar (`z = d·(d_max−d_min)+d_min`, `X = (u·S − pad_x0 − cx)·z/fx`, …); `axis_cam = normalize(P1 − P0)`, `origin_cam = P0` for revolute (midpoint of the two endpoints for prismatic); `type` revolute→1, prismatic→0, anything else → unmatched; `mask_rle` = filled 2D hull of the predicted 8 vertices in the native frame (documented as a box IoU, weaker than a mask IoU); `matched` = parse succeeded; `score` = 1.0.
- Two output files per run: `preds_desc.jsonl` (description → joint, our story) and `preds_box.jsonl` (GT box → joint, their REG-Joint; mask = GT box hull, so PDet is not meaningful there and is reported as n/a).
- [ ] Step 1: failing tests with synthetic answers incl. a malformed one; round trip with the converter's serialiser on a synthetic record.
- [ ] Step 2: implement; tests; commit.

### Task 4: A3VLM env + chain scripts

- `setup_env.sh` (cu118 image, Python 3.10): `pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118` if the image differs, then LLaMA2-Accessory `requirements.txt` minus torch lines, plus `timm dacite huggingface_hub`; try `pip install flash-attn --no-build-isolation` (skip on failure, note it); clone `https://github.com/changhaonan/A3VLM` (pinned commit) and use `A3VLM/model/accessory` as the training tree (it carries `llama_ens5`); `HF_HUB_OFFLINE=1 TORCH_HOME=/workspace/ckpt/torch`; check `python -c "import accessory.model.LLM.llama_ens5"`.
- `chain.sh smoke|train|eval`: 
  - train: `torchrun --nproc_per_node ${NPROC:-8} main_finetune.py --output_dir $RUNS/a3vlm --epochs 3 --warmup_epochs 0.03 --batch_size 2 --accum_iter ${ACCUM} --num_workers 4 --max_words 2048 --lr 2e-5 --min_lr 0 --clip_grad 8 --weight_decay 0 --data_parallel sdp --model_parallel_size 2 --checkpointing --llama_type llama_ens5 --llama_config $CKPT/sphinx/config.json --tokenizer_path $CKPT/sphinx/tokenizer.model --pretrained_path $CKPT/sphinx --pretrained_type consolidated --data_config configs/a3vlm_sf3d.yaml --dialog --image_transform padded_resize --precision bf16 --save_iteration_interval 2000` with `ACCUM` chosen so effective batch = 128 (their 2×8×8): 8 GPUs / MP 2 → dp 4 → `--accum_iter 16`; resume via `--resume $RUNS/a3vlm/epochN` when present.
  - eval: `torchrun --nproc-per-node=2 eval_affordance_v2.py --llama_type llama_ens5 --llama_config … --tokenizer_path $RUNS/a3vlm/epoch2/tokenizer.model --pretrained_path $RUNS/a3vlm/epoch2 --dataset $DATA/reg_joint_desc_test.json --batch_size 8 --input_size 448 --model_parallel_size 2 --sampled_num 100000 --addition_flag sf3d_desc --remove_space` and the same for `reg_joint_box_test.json` and `rec_desc_test.json`; then `a3vlm_preds_to_jsonl.py`; CHAIN_DONE.
- [ ] Step 1: write the three scripts + `config/baselines/a3vlm_sf3d.yaml`; commit.

### Task 5: A3VLM smoke gate (2×A100, ~$3/h)

- [ ] Step 1: rsync `stage_us/a3vlm` (EU dev pod → `bl-us-smoke:/workspace/data/a3vlm`); verify counts and 5 random image paths open.
- [ ] Step 2: `setup_env.sh`; confirm SPHINX-1k download complete (2×19.9 GB, md5 not published → check sizes).
- [ ] Step 3: `NPROC=2 ACCUM=64` smoke: `--epochs 1` on a 200-sample subset YAML with `--save_iteration_interval 5`; must produce `epoch0/consolidated.0{0,1}-of-02.model.pth`; then re-launch with `--resume` for 5 more iterations (resume proven); then run eval on 20 test questions of each of the three JSONs and the export → 20 JSONL lines that the scorer accepts. Record it/s and GPU memory.
- [ ] Step 4: freeze the recipe in `config/baselines/a3vlm_sf3d.yaml` + `experiments/baselines_sf3d/20260913_a3vlm/notes.md` (samples/epoch, effective batch, epochs, estimated hours from the measured it/s ×4 for 8 GPUs) and show the user the cost. STOP until confirmed.

### Task 6: A3VLM full run (8×A100 SXM)

- [ ] Step 1: create `bl-us-a3vlm` (8×A100 SXM) on the same volume; `setup_env.sh`; `chain.sh train` detached; Monitor watcher (10-min polls) + Mac watcher that copies `epoch*/` metadata, eval JSONs and `preds_*.jsonl` to the volume `results/a3vlm/` and deletes the pod on CHAIN_DONE.
- [ ] Step 2: on CHAIN_DONE: rsync `preds_*.jsonl` + logs to the EU volume (`results/a3vlm/`); score on the dev pod; notes + INDEX row + STATE; commit.

### Task 7: `sf3d_to_3doi.py` + tests

**Contract (from `monoarti/monoarti/dataset.py`):**
- Layout under `stage_us/3doi/`: `images/<name>.png` (768×1024 RGB; portrait frames rolled 90° CW first, as in the OPD converter), `3doi_sf3d/data_{train,val,test}.pt` (torch-saved list of `{"img_name", "instances": [...]}`), depth tree `omnidata_filtered/depth_zbuffer/taskonomy/<visit>/point_<F>_view_0_domain_depth_zbuffer.png` (uint16 = metres×512 at 768×1024, nearest), `.../mask_valid/taskonomy/<visit>/...png` (255 where depth>0), `.../point_info/taskonomy/<visit>/point_<F>_view_0_domain_point_info.json` (`{"field_of_view_rads": 2·atan(H/(2·fy))}`); image names `taskonomy_<visit>_point_<F>_view_0_domain_rgb.png` so their loader's depth branch fires. The pod gets symlinks `/home/ubuntu/monoarti_data -> /workspace/data/3doi` and `DEFAULT_DEPTH_ROOT -> /workspace/data/3doi/omnidata_filtered/depth_zbuffer/taskonomy` (no code change).
- Per element instance: `keypoint` = element mask centroid (normalised x,y), `movable: "one_hand"`, `rigid: "yes"`, `kinematic: "rotation"|"translation"`, `pull_or_push: "n/a"`, `affordance` = same centroid, `bbox` = mask bbox normalised `[x1,y1,x2,y2]`, `mask` = ONE polygon: the convex hull of the (rolled, resized) splat mask pixels, normalised `[[x,y],…]` (documented approximation), `axis` = `[x1,y1,x2,y2]` normalised: the GT 3D axis line projected (origin and origin+n·1 m through K after roll), intersected with the image rectangle, endpoints clipped to `[0.002, 0.998]`, ordered so that `x1 > 0.0` (their validity test is `axis[:,0] > 0`); if the projected line misses the image, `[-1,-1,-1,-1]`.
- Padding to 15 queries is done by their loader; max 11 elements per frame in our data.
- CLI: `--out /workspace/datasets/baselines/stage_us/3doi --workers 32 [--limit-frames N]`; splits train/val(bvalid)/test.
- [ ] Step 1: failing tests: roll+resize+keypoint consistency, polygon rasterises to IoU ≥ 0.7 with the splat mask on a synthetic blob, axis line clipping keeps `x1 > 0` and points on the projected line, depth png round-trips to metres within 2 mm, `data_*.pt` loads with `torch.load` and has the exact keys.
- [ ] Step 2: implement; tests on dev pod; 50-frame smoke; render 4 debug PNGs (image + polygon + axis line + keypoint) into `viz/20260913_3doi_data_check/`.
- [ ] Step 3: full conversion (39k frames: ~10 GB RGB + ~40 GB depth) on the dev pod in the background (~40 min; announce to the peer); commit.

### Task 8: `threedoi_export.py` + tests

- Loads their config + checkpoint (`checkpoint.pth`), builds `InteractionDataset`-compatible batches from `data_test.pt` (query = GT element keypoint), runs the SAM model forward (`model(**batch)` in eval mode returns `pred_masks`, `pred_boxes`, `kinematic` logits, `pred_axis` (sin,cos,r), `pred_depth`), decodes the axis line with `axis_ops.line_angle_to_xyxy(pred_axis, center=pred_box_center)`; `type` = argmax kinematic (1 rotation → 1, 2 translation → 0, 0 freeform → unmatched); mask = `sigmoid(pred_masks) > 0.5` un-rolled to the native frame → `mask_rle`; **3D lift**: back-project the predicted-mask pixels with the INPUT depth (their `export_video` "aligned to GT depth" variant), RANSAC plane, intersect the two line endpoints' rays with the plane → `axis_cam = normalize(P1−P0)` (rotation) or the plane normal (translation, as in their code), `origin_cam` = the lifted-line point nearest the element's 3D centroid; un-roll vectors for portrait frames; `score` = mask IoU-prediction head if present else 1.0.
- [ ] Step 1: failing tests: line decode/lift on a synthetic plane returns the planted axis within 1°; roll round trip; a frame with `freeform` argmax becomes unmatched.
- [ ] Step 2: implement (needs their env; run tests on the smoke pod); commit.

### Task 9: 3DOI env, smoke gate, recipe freeze

- `setup_env.sh` (cu121 image): torch 2.1.1 already; `pip install accelerate hydra-core pycocotools packaging plotly imageio matplotlib h5py opencv-python-headless tqdm wandb`; clone `https://github.com/JasonQSY/3DOI` pinned; SAM weights `sam_vit_b_01ec64.pth` into `monoarti/checkpoints/`; `WANDB_MODE=offline`; symlinks for `DEFAULT_DATA_ROOT`/`DEFAULT_DEPTH_ROOT`; `config sam_sf3d.yaml` = their `sam.yaml` with `data.*_dataset_names: ['3doi_sf3d']`, `hydra/launcher: basic`, `checkpoint_epoch_interval: 1`, `validation_epoch_interval: 5`, everything else verbatim (batch 2 per GPU, 200 epochs, AdamW 1e-4 / 1e-5, fp16, clip 0.1).
- `chain.sh`: `accelerate launch --num_processes ${NPROC:-4} --mixed_precision fp16 train.py --config-name sam_sf3d hydra.run.dir=$RUNS/3doi` (resume: `resume: True checkpoint_path=$RUNS/3doi/checkpoints/checkpoint.pth`), then `threedoi_export.py`, CHAIN_DONE.
- [ ] Step 1: on `bl-us-smoke` (2×A100): env; rsync `stage_us/3doi`; `NPROC=2` smoke with `optimizer.max_epochs=1 +train.limit_iters=30` (add a tiny patch to break the epoch loop after N iterations, env-guarded), checkpoint written, resume for 10 more iterations, export on 30 test frames → scorer accepts. Record it/s per GPU at batch 2.
- [ ] Step 2: freeze the recipe: with the measured it/s compute hours for 200 epochs × 32,171 frames / (4 GPUs × batch 2); write `config/baselines/3doi_sf3d.yaml` + notes; show the user the cost and, as the alternative, the iteration-matched budget (their 200 epochs × 10k images / 8 ≈ 250k iterations). STOP until confirmed.

### Task 10: 3DOI full run (4×A100 SXM)

- [ ] Step 1: create `bl-us-3doi`; `chain.sh` detached; watchers as in Task 6.
- [ ] Step 2: on CHAIN_DONE: score, notes, INDEX row, STATE; commit; delete pods; reconcile `runpodctl pod list`; keep the US volume until the user decides.

## Self-review

- Coverage: both models (Tasks 2-6, 7-10), faithful recipes (their scripts verbatim, deviations listed: A3VLM effective batch matched via accum on 8 GPUs, our text-query task added alongside their box-query task, per-image depth normalisation as in their generator; 3DOI epoch count decided by the user in Task 9), single-shot guards (smoke gates, resume, frozen recipes), scoring with the existing protocol.
- Placeholders: none; numeric contracts are explicit. Names consistent: `stage_us/{a3vlm,3doi}`, `results/{a3vlm,3doi}`, run ids `20260913_a3vlm`, `20260913_3doi`.

---

## Execution log (2026-09-12/13) — what the de-risking actually found

The plan above was written before any code ran. These are the deviations, all forced by measurement:

1. **A3VLM needs >= 4 GPUs; the 2-GPU smoke was impossible.** `--data_parallel sdp` shards the
   optimizer over DATA-parallel ranks only. With `--model_parallel_size 2`, two GPUs give DP=1,
   nothing is sharded, and each GPU needs ~75 GB for its half of the 13B AdamW state: measured OOM
   at 74.87/79.19 GiB on 2 x H100. `chain.sh` now refuses DP<2. Consequence for the single-shot
   protocol: the A3VLM smoke is only meaningful at the real width, so it runs on the full-size pod
   as the first ~15 minutes of that pod's life, not on a cheap separate one.
2. **A3VLM's repo is a partial overlay.** `changhaonan/A3VLM@436e715` ships `model/accessory`
   without `llama.py`, `configs/global_configs.py` and others; it must be copied over an upstream
   `Alpha-VLLM/LLaMA2-Accessory` checkout. Unpinned `open_clip_torch`/`timm`/`bitsandbytes` drag
   torch to 2.14+cu130 on the torch-2.0.1 image, so they are pinned to their late-2023 releases.
3. **Their evaluator uses one model-parallel group no matter how many GPUs exist.** A single call
   would leave 6 of 8 GPUs idle for ~2 h of generation, so `evaluate()` shards the question JSON
   into NPROC/MP pieces and runs one generation group per GPU pair, then merges. Same model,
   prompts and sampling; only the work split differs.
4. **Their evaluator cannot identify which element an answer belongs to.** It records image and
   prompt only, and our chained questions for two elements of one frame can be byte-identical.
   `runpod/baselines/a3vlm/patch_eval.py` threads our `key` through, and also disables the
   resumption block that drops every question sharing an already-answered image.
5. **The axis-segment length materially changes A3VLM's achievable score.** Their answer format
   quantises (u, v, d) to 2 decimals, and on SF3D one depth step is ~2.4 cm. With the short
   element-sized segments first generated (mean 0.135 m) the GT round trip already costs 5.09 deg
   mean axis error and only 89.5% of elements fall inside the 10 deg MA threshold -- i.e. the
   encoding alone would have capped A3VLM near MA 89. Emitting the longest segment that still
   projects inside the padded square (`AXIS_LENGTHS`, mean 0.5 m) moves the round trip to
   **1.21 deg mean / 99.9% within 10 deg**. The data was regenerated before any training.
6. **3DOI validates at epoch 0 over the whole val split** (`epoch % interval == 0`), ~1 h per pass,
   and nothing selects a checkpoint from it. Capped with `SF3D_LIMIT_VAL_ITERS` (default 200
   batches); training is untouched.
7. **3DOI needs a 40 GB+ card.** Batch 2 at 1024x768 peaks near 20 GB, and the resume leg (which
   also loads the optimizer state) OOMed on the 24 GB dev GPU after the training leg had succeeded.
8. **Scorer change:** IoU is now taken from `mask_rle` rather than from `matched`, so a
   point-prompted method that segments but declines to predict a joint is not charged IoU 0 for a
   good mask. Every existing baseline emits a mask only on matched rows, so all five published
   numbers are provably unchanged.
9. **Where each model runs.** The Euler per-user cap of 2 GPUs is real and enforced by a
   client-side `cli_filter` (`--gpus=4` and `--gpus=8` are rejected before a job id exists), so
   A3VLM cannot run there at all and stays on RunPod. 3DOI fits inside the cap and runs on Euler
   (2 x RTX PRO 6000, 96 GB each, 5-day wall clock, self-requeueing), which removes what would have
   been the single most expensive RunPod item.

**Validated end to end before any full run:** both converters (unit tests + GT round trip), 3DOI
train -> checkpoint -> resume -> export -> score (45 predictions, 43 above IoU 0.5, 15.2 deg matched
axis error after 30 iterations), and the A3VLM answer encode/parse/export path.
