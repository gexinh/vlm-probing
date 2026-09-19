"""Offline, tiny image+text models for all seven supported VLM architectures.

These use real Transformers modules with random weights, not pretrained models.
Run: python examples/hf_model_matrix.py
"""
import torch

from hf_smoke import make_qwen
from vlm_probing import Prober

FAMILIES = ("qwen3_5", "qwen3_vl", "qwen2_5_vl", "qwen2_vl", "smolvlm", "internvl", "llava")


def make_model(family):
    import transformers as hf

    if family == "qwen2_5_vl":
        return make_qwen()
    text = dict(vocab_size=40, hidden_size=16, intermediate_size=24,
                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                pad_token_id=0)
    vision = dict(hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                  num_attention_heads=2, image_size=4, patch_size=2)
    if family.startswith("qwen"):
        vision = dict(depth=2, hidden_size=16, intermediate_size=24, num_heads=2,
                      patch_size=2, temporal_patch_size=1, spatial_merge_size=2,
                      out_hidden_size=16)
        text["rope_parameters"] = dict(rope_type="default", mrope_section=[1, 1, 2])
        if family == "qwen2_vl":
            vision.update(embed_dim=16, mlp_ratio=2)
            cls, cfg = hf.Qwen2VLForConditionalGeneration, hf.Qwen2VLConfig
        elif family == "qwen3_vl":
            text.update(head_dim=8)
            vision.update(num_position_embeddings=16, deepstack_visual_indexes=[0])
            cls, cfg = hf.Qwen3VLForConditionalGeneration, hf.Qwen3VLConfig
        else:
            text.update(num_hidden_layers=4, head_dim=16,
                        layer_types=["linear_attention"] * 3 + ["full_attention"],
                        linear_conv_kernel_dim=2, linear_key_head_dim=4, linear_value_head_dim=4,
                        linear_num_key_heads=2, linear_num_value_heads=2)
            text["rope_parameters"]["partial_rotary_factor"] = 0.5
            vision.update(num_position_embeddings=16)
            cls, cfg = hf.Qwen3_5ForConditionalGeneration, hf.Qwen3_5Config
        config = cfg(text_config=text, vision_config=vision, image_token_id=28, video_token_id=30,
                     vision_start_token_id=27, vision_end_token_id=29)
        inputs = {"input_ids": torch.tensor([[1, 27, 28, 29, 3, 0], [1, 27, 28, 28, 29, 3]]),
                  "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]]),
                  "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 4]]),
                  "pixel_values": torch.randn(12, 12)}
    else:
        text["model_type"] = "qwen2" if family == "internvl" else "llama"
        if family == "smolvlm":
            # SmolVLM-256M/500M/2B checkpoints use the native Idefics3 class.
            config = hf.Idefics3Config(text_config=text, vision_config=vision,
                                     image_token_id=28, pad_token_id=0, scale_factor=2)
            cls, count = hf.Idefics3ForConditionalGeneration, 1
        elif family == "internvl":
            config = hf.InternVLConfig(text_config=text, vision_config=vision,
                                      image_token_id=28, image_seq_length=1, downsample_ratio=0.5)
            cls, count = hf.InternVLForConditionalGeneration, 1
        elif family == "llava":
            config = hf.LlavaConfig(text_config=text, vision_config=vision,
                                   image_token_index=28, image_seq_length=4, vision_feature_layer=-1)
            cls, count = hf.LlavaForConditionalGeneration, 4
        else:
            raise ValueError(f"unknown family: {family}")
        ids = torch.tensor([[1] + [28] * count + [3, 0], [0, 1] + [28] * count + [3]])
        inputs = {"input_ids": ids, "attention_mask": (ids != 0).long(),
                  "pixel_values": torch.randn(2, 3, 4, 4)}
        if family == "smolvlm":
            inputs["pixel_values"] = inputs["pixel_values"][:, None]
    config._attn_implementation = "eager"
    return cls(config).eval(), inputs


if __name__ == "__main__":
    torch.manual_seed(3)
    torch.set_num_threads(1)
    for family in FAMILIES:
        model, inputs = make_model(family)
        probe = Prober(model)
        result = probe.lens.logit().run(inputs)
        print(family, tuple(result.tensors["logits"].shape),
              probe.lens.embed().run(inputs, top_k=2).tensors["positions"].tolist())
