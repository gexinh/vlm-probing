"""Small randomly initialized HF models; CPU only, no pretrained downloads.

Requires the optional transformers extra. Qwen runs its actual vision encoder,
projector, multimodal position handling, and language decoder on synthetic pixels.
"""
import torch

from vlm_probing import Prober, TokenMargin


def make_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(vocab_size=40, hidden_size=16, intermediate_size=24,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    inputs = {"input_ids": torch.tensor([[1, 2, 3, 0], [0, 1, 4, 5]]),
              "attention_mask": torch.tensor([[1, 1, 1, 0], [0, 1, 1, 1]])}
    return model, inputs


def make_qwen():
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    config = Qwen2_5_VLConfig(
        text_config={"vocab_size": 40, "hidden_size": 16, "intermediate_size": 24,
                     "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
                     "rope_scaling": {"type": "default", "mrope_section": [1, 1, 2]}},
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 24,
                       "num_heads": 2, "patch_size": 2, "temporal_patch_size": 1,
                       "spatial_merge_size": 2, "out_hidden_size": 16,
                       "window_size": 8, "fullatt_block_indexes": [0]},
        image_token_id=28, video_token_id=30, vision_start_token_id=27, vision_end_token_id=29,
    )
    config._attn_implementation = "eager"
    model = Qwen2_5_VLForConditionalGeneration(config).eval()
    # One image per example, with one vs. two merged visual tokens.
    inputs = {"input_ids": torch.tensor([[1, 27, 28, 29, 3, 0], [1, 27, 28, 28, 29, 3]]),
              "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]]),
              "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 4]]),
              "pixel_values": torch.randn(12, 12)}
    return model, inputs


def main():
    torch.manual_seed(3)
    torch.set_num_threads(1)
    for factory in (make_llama, make_qwen):
        model, inputs = factory()
        probe = Prober(model)
        print(type(model).__name__, probe.lens.logit().run(inputs).tensors["logits"].shape)
        probe.lens.embed(tokens="all").run(inputs, top_k=3)
        probe.lens.tuned(layers=[0]).fit(inputs, steps=2).run(inputs)
        probe.lens.attention(layers=[0]).fit(inputs, steps=2).run(inputs)
        probe.lens.jacobian(layers=[0]).fit(inputs, skip_first=0, exclude_last=False).run(inputs)
        probe.lens.patchscope(layers=[0], source_position=2, target_layer=0, target_position=2).run(
            inputs, target_inputs=inputs)
        metric = TokenMargin(4, 5)
        probe.causal.patch(layers=[0], tokens="all").run(inputs, source=inputs, metric=metric)
        probe.attention.profile().run(inputs)
        probe.attention.rollout().run(inputs)
        probe.attention.relevance().run(inputs, metric=metric)
        probe.attention.head_logits().run(inputs)
    print("HF smoke paths completed; no pretrained weights downloaded.")


if __name__ == "__main__":
    main()
