"""Minimal patches to 3DOI/monoarti for SF3D. Idempotent; every replacement is asserted unique.

1. depth validity: their loader marks holes as -1 and the loss checks only pixel (0, 0) to decide
   whether a frame has depth at all (fine for hole-free Taskonomy renders). SF3D sensor depth has
   holes, so the check becomes "any valid pixel" (the per-pixel mask ``tgt_depths > 1e-8`` already
   excludes holes from the loss).
2. train.py / stats.py import visdom and submitit at module level although neither is used with
   the basic launcher; guard the imports so the env needs neither.
3. optional iteration cap for smokes: ``SF3D_LIMIT_ITERS`` (env) breaks the epoch loop early.

  python patch_train.py <path to 3DOI/monoarti>
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])


def patch(path, edits, marker):
    p = root / path
    s = p.read_text()
    if marker in s:
        print("already patched", p)
        return
    for old, new in edits:
        assert s.count(old) == 1, (path, old)
        s = s.replace(old, new)
    p.write_text(s)
    print("patched", p)


for f in ("monoarti/sam_transformer.py", "monoarti/transformer.py"):
    patch(f, [("valid_depth = depth[:, 0, 0] > 0", "valid_depth = (depth > 0).flatten(1).any(1)  # SF3D_PATCHED")], "SF3D_PATCHED")

patch("train.py", [
    ("from visdom import Visdom\n", "try:  # SF3D_PATCHED\n    from visdom import Visdom\nexcept ImportError:\n    Visdom = None\n"),
    # Their validation walks the whole val split, and `epoch % interval == 0` fires at epoch 0, so a
    # 3,495-frame pass costs ~1 h per validation. Nothing selects a checkpoint from it (train.py just
    # overwrites checkpoint.pth every interval and we export from the last one), so cap the number of
    # val batches. Training is untouched.
    ("    for iteration, batch in enumerate(val_dataloader):\n        loss = 0.0\n",
     "    for iteration, batch in enumerate(val_dataloader):\n"
     "        if os.environ.get('SF3D_LIMIT_VAL_ITERS') and iteration >= int(os.environ['SF3D_LIMIT_VAL_ITERS']):\n"
     "            break\n"
     "        loss = 0.0\n"),
    ("        for iteration, batch in enumerate(train_dataloader):\n            optimizer.zero_grad()\n",
     "        for iteration, batch in enumerate(train_dataloader):\n"
     "            if os.environ.get('SF3D_LIMIT_ITERS') and iteration >= int(os.environ['SF3D_LIMIT_ITERS']):\n"
     "                break\n"
     "            optimizer.zero_grad()\n"),
], "SF3D_PATCHED")

# --- early stopping on validation loss (user request 2026-09-13) -------------------------------
# Their val loop composes a per-batch loss and throws it away (`losses = []` is never appended to),
# validates only every `validation_epoch_interval` epochs and always exports the LAST checkpoint.
# We (a) accumulate the val loss and return it, (b) keep the best-val checkpoint separately, and
# (c) stop when it has not improved for SF3D_EARLY_STOP_PATIENCE validations. Training maths,
# losses and hyper-parameters are untouched; this only decides when to stop and which checkpoint
# to export.
patch("train.py", [
    ("        loss += cfg.optimizer.lbd_mask * metrics['loss_mask']\n        loss += cfg.optimizer.lbd_dice * metrics['loss_dice']\n\n        # valid\n",
     "        loss += cfg.optimizer.lbd_mask * metrics['loss_mask']\n        loss += cfg.optimizer.lbd_dice * metrics['loss_dice']\n"
     "        losses.append(float(loss))  # SF3D_EARLYSTOP: their `losses` list is never filled\n\n        # valid\n"),
    ("    if accelerator.is_local_main_process:\n        logger.info(\"-------------------------------------------------------\")\n        logger.info(\"validation results:\")\n",
     "    _vt = torch.tensor([float(sum(losses)), float(len(losses))], device=accelerator.device)  # SF3D_EARLYSTOP\n"
     "    _vt = accelerator.reduce(_vt, reduction='sum')  # SF3D_EARLYSTOP: one GLOBAL val loss, identical on every rank\n"
     "    val_results['loss'] = float(_vt[0].item() / max(_vt[1].item(), 1.0))\n"
     "    if accelerator.is_local_main_process:\n        logger.info(\"-------------------------------------------------------\")\n        logger.info(\"validation results:\")\n"),
    ("        if epoch % cfg.validation_epoch_interval == 0:\n            evaluate(cfg, model, val_dataloader, accelerator, stats)\n",
     "        if epoch % cfg.validation_epoch_interval == 0:\n"
     "            _val = evaluate(cfg, model, val_dataloader, accelerator, stats)\n"
     "            _vl = float(_val.get('loss', float('nan')))\n"
     "            _pat = int(os.environ.get('SF3D_EARLY_STOP_PATIENCE', '0'))\n"
     "            if _pat > 0 and _vl == _vl:\n"
     "                _bvp = os.path.join(output_dir, 'checkpoints', 'best_val.json')\n"
     "                if _vl < _best_val[0] - 1e-4:\n"
     "                    _best_val[0], _best_val[1] = _vl, 0\n"
     "                    if accelerator.is_main_process:\n"
     "                        _bp = os.path.join(output_dir, 'checkpoints', 'checkpoint_best.pth')\n"
     "                        torch.save({'model': accelerator.unwrap_model(model).state_dict(),\n"
     "                                    'optimizer': optimizer.state_dict(), 'stats': pickle.dumps(stats),\n"
     "                                    'val_loss': _vl, 'epoch': epoch}, _bp)\n"
     "                        logger.info(f'SF3D_EARLYSTOP new best val loss {_vl:.4f} at epoch {epoch} -> {_bp}')\n"
     "                else:\n"
     "                    _best_val[1] += 1\n"
     "                    if accelerator.is_main_process:\n"
     "                        logger.info(f'SF3D_EARLYSTOP val loss {_vl:.4f} did not beat {_best_val[0]:.4f} '\n"
     "                                    f'({_best_val[1]}/{_pat} validations without improvement)')\n"
     "                if accelerator.is_main_process:\n"
     "                    __import__('json').dump({'best': _best_val[0], 'since': _best_val[1], 'epoch': epoch}, open(_bvp, 'w'))\n"
     "                if _best_val[1] >= _pat:\n"
     "                    if accelerator.is_main_process:\n"
     "                        logger.info('SF3D_EARLYSTOP stopping: validation loss has stopped falling')\n"
     "                    break\n"),
    ("    # Run the main training loop.\n    for epoch in range(start_epoch, cfg.optimizer.max_epochs):\n",
     "    # Run the main training loop.\n    _best_val = [float('inf'), 0]  # SF3D_EARLYSTOP: [best val loss, validations since]\n"
     "    _bvp0 = os.path.join(output_dir, 'checkpoints', 'best_val.json')\n"
     "    if os.path.isfile(_bvp0):  # SF3D_EARLYSTOP: a resumed/requeued run keeps its early-stop state\n"
     "        try:\n"
     "            _bj = __import__('json').load(open(_bvp0)); _best_val = [float(_bj['best']), int(_bj.get('since', 0))]\n"
     "            logger.info(f'SF3D_EARLYSTOP restored best val loss {_best_val[0]:.4f} ({_best_val[1]} validations since)')\n"
     "        except Exception as _e:\n"
     "            logger.info(f'SF3D_EARLYSTOP could not read {_bvp0}: {_e}')\n"
     "    for epoch in range(start_epoch, cfg.optimizer.max_epochs):\n"),
    # (d) their resume restores the model and the epoch counter but leaves the optimizer state
    # commented out (test.py loads it). Restore it, so a requeue does not reset AdamW's moments.
    ("        # optimizer_state_dict = loaded_data[\"optimizer\"]\n",
     "        optimizer_state_dict = loaded_data[\"optimizer\"]  # SF3D_EARLYSTOP: restore AdamW state on resume\n"),
], "SF3D_EARLYSTOP")

patch("monoarti/stats.py", [
    ("from visdom import Visdom\n", "try:\n    from visdom import Visdom  # SF3D_PATCHED: optional\nexcept ImportError:\n    Visdom = None\n"),
], "SF3D_PATCHED")

