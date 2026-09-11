from tools.baselines_sf3d.mopd_compose_ckpt import compose, remap_esam, unwrap_opd


def test_remap_and_compose_prefixes():
    opd = {"backbone.stem.conv1.weight": 1, "sem_seg_head.predictor.query_feat.weight": 2}
    esam = {"image_encoder.patch_embed.proj.weight": 3, "image_encoder.blocks.0.norm1.weight": 4,
            "prompt_encoder.point_embeddings.0.weight": 5, "mask_decoder.x": 6}
    normal = {"original_model.conv_stem.weight": 7}
    sd = compose(unwrap_opd({"model": opd, "iteration": 60000}), esam, normal)
    assert sd["backbone.stem.conv1.weight"] == 1
    assert sd["image_encoder.module.patch_embed.proj.weight"] == 3
    assert sd["image_encoder.module.blocks.0.norm1.weight"] == 4
    assert "prompt_encoder.point_embeddings.0.weight" not in sd and "mask_decoder.x" not in sd
    assert sd["normal_encoder.original_model.conv_stem.weight"] == 7
    assert remap_esam({"other": 0}) == {}
