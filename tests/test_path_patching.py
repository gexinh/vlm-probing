"""IOI path controls checked against an independent native-hook experiment."""

from contextlib import contextmanager
import importlib.util
import math
import unittest

import torch
from torch import nn

from vlm_probing import (
    CapabilityError, HeadSite, HookPoint, ModelReadout, ModelSpec, Prober,
    TokenLayout, TokenMargin, TorchModelAdapter,
)


class CircuitBlock(nn.Module):
    """Two physical query heads, optional shared KV head, and real LN/MLP."""

    def __init__(self, kv_heads):
        super().__init__()
        self.kv_heads = kv_heads
        self.norm1, self.norm2 = nn.LayerNorm(4), nn.LayerNorm(4)
        self.q = nn.Linear(4, 4, bias=False)
        self.k, self.v = nn.Linear(4, kv_heads * 2, bias=False), nn.Linear(4, kv_heads * 2, bias=False)
        self.head_out = nn.Identity()
        self.o = nn.Linear(4, 4, bias=False)
        self.mlp = nn.Sequential(nn.Linear(4, 7), nn.Tanh(), nn.Linear(7, 4))
        self.residual = nn.Identity()

    def forward(self, hidden, record, layer):
        normalized = self.norm1(hidden)
        q = self.q(normalized).reshape(*hidden.shape[:2], 2, 2)
        k = self.k(normalized).reshape(*hidden.shape[:2], self.kv_heads, 2)
        v = self.v(normalized).reshape(*hidden.shape[:2], self.kv_heads, 2)
        record["norm"][layer] = normalized.detach().clone()
        for kind, value in (("q", q), ("k", k), ("v", v)):
            record[kind][layer] = value.detach().clone()
        key, value = (tensor.repeat_interleave(2 // self.kv_heads, dim=2) for tensor in (k, v))
        scores = torch.einsum("bthd,bshd->bhts", q, key) / math.sqrt(2)
        future = torch.ones(hidden.shape[1], hidden.shape[1], dtype=torch.bool).triu(1)
        attention = scores.masked_fill(future, -torch.inf).softmax(-1)
        heads = self.head_out(torch.einsum("bhts,bshd->bthd", attention, value))
        record["heads"][layer] = heads.detach().clone()
        hidden = hidden + self.o(heads.flatten(-2))
        mlp = self.mlp(self.norm2(hidden))
        record["mlp"][layer] = mlp.detach().clone()
        hidden = self.residual(hidden + mlp)
        record["residual"][layer] = hidden.detach().clone()
        return hidden


class CircuitModel(nn.Module):
    """A deterministic decoder with logs of the values actually used each run."""

    def __init__(self, *, kv_heads=2, flattened_heads=False):
        super().__init__()
        self.embedding = nn.Embedding(13, 4)
        self.blocks = nn.ModuleList(CircuitBlock(kv_heads) for _ in range(3))
        self.final_residual, self.norm = nn.Identity(), nn.LayerNorm(4)
        self.readout = nn.Linear(4, 13, bias=False)
        self.records = []
        self.fail_run = None
        self.kv_heads, self.flattened_heads = kv_heads, flattened_heads
        self.double()

    def forward(self, input_ids, attention_mask=None):
        record = {name: {} for name in ("q", "k", "v", "heads", "norm", "mlp", "residual")}
        self.records.append(record)
        if self.fail_run == len(self.records):
            raise RuntimeError("intentional controlled-run failure")
        hidden = self.embedding(input_ids)
        for layer, block in enumerate(self.blocks):
            hidden = block(hidden, record, layer)
        hidden = self.final_residual(hidden)
        record["final"] = hidden.detach().clone()
        logits = self.readout(self.norm(hidden))
        record["logits"] = logits.detach().clone()
        return {"logits": logits}

    def probing_adapter(self):
        sites, residuals, head_sites, qkv = {}, {}, {}, {}
        for layer in range(3):
            residuals[layer] = f"residual.{layer}"
            sites[residuals[layer]] = HookPoint(f"blocks.{layer}.residual")
            output = f"head_output.{layer}"
            # The output projection sees flattened messages. The identity sees
            # the same messages with an explicit physical head dimension.
            sites[output] = (HookPoint(f"blocks.{layer}.o", kind="input", selector=0)
                             if self.flattened_heads else HookPoint(f"blocks.{layer}.head_out"))
            head_sites[layer] = HeadSite(output, heads=2, head_dim=2)
            qkv[layer] = {}
            for kind in ("q", "k", "v"):
                name = f"{kind}.{layer}"
                sites[name] = HookPoint(f"blocks.{layer}.{kind}")
                qkv[layer][kind] = HeadSite(name, heads=2 if kind == "q" else self.kv_heads, head_dim=2)
        sites["final"] = HookPoint("final_residual")

        def layout(inputs):
            ids = inputs["input_ids"]
            valid = inputs.get("attention_mask", torch.ones_like(ids)).bool()
            return TokenLayout(valid=valid, token_ids=ids)

        spec = ModelSpec(residuals, layout=layout, path_heads=head_sites,
                         path_qkv=qkv, path_final="final")
        return TorchModelAdapter(self, sites, spec=spec, model_id="independent-path-circuit",
                                 readout=ModelReadout(self.readout, norm=self.norm))


@contextmanager
def native_hooks(edits):
    """Direct PyTorch hooks, independent of adapter and path implementation."""
    handles = []
    try:
        for module, edit in edits:
            handles.append(module.register_forward_hook(lambda _m, _a, value, edit=edit: edit(value)))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def native_pre_hooks(edits):
    handles = []
    try:
        for module, edit in edits:
            handles.append(module.register_forward_pre_hook(
                lambda _m, args, edit=edit: (edit(args[0]), *args[1:])))
        yield
    finally:
        for handle in handles:
            handle.remove()


def native_gpt2_oracle(model, base, donor, senders, receivers, selected):
    """Operate on native packed projections without HookPoint or HeadSite."""
    def forward(inputs, *, output_edits=(), input_edits=()):
        captured = {"heads": {}, "qkv": {}}
        def keep(value, *, layer, kind):
            captured[kind][layer] = value.detach().clone()
            return value
        projections = [(block.attn.c_attn, lambda value, layer=i: keep(value, layer=layer, kind="qkv"))
                       for i, block in enumerate(model.transformer.h)]
        heads = [(block.attn.c_proj, lambda value, layer=i: keep(value, layer=layer, kind="heads"))
                 for i, block in enumerate(model.transformer.h)]
        with native_hooks(output_edits), native_pre_hooks(input_edits), native_hooks(projections), native_pre_hooks(heads):
            logits = model(**inputs, use_cache=False, return_dict=True).logits
        return logits, captured

    with torch.no_grad():
        base_logits, clean = forward(base)
        donor_logits, changed = forward(donor)
        freezes = []
        for layer, block in enumerate(model.transformer.h):
            def freeze(value, layer=layer):
                result = clean["heads"][layer].reshape(*value.shape[:2], model.config.n_head, -1).clone()
                source = changed["heads"][layer].reshape_as(result)
                for sender_layer, head in senders:
                    if sender_layer == layer:
                        result[:, :, head][selected] = source[:, :, head][selected]
                return result.reshape_as(value)
            freezes.append((block.attn.c_proj, freeze))
        _, controlled = forward(base, input_edits=freezes)
        targets = {}
        for layer, head, kind in receivers:
            targets.setdefault(layer, []).append((head, kind))
        injections = []
        for layer, targets_in_layer in targets.items():
            def inject(value, layer=layer, targets_in_layer=targets_in_layer):
                result = value.clone()
                for head, kind in targets_in_layer:
                    start = {"q": 0, "k": 1, "v": 2}[kind] * model.config.n_embd
                    stop = start + model.config.n_embd
                    target = result[..., start:stop].reshape(*value.shape[:2], model.config.n_head, -1)
                    source = controlled["qkv"][layer][..., start:stop].reshape_as(target)
                    target[:, :, head][selected] = source[:, :, head][selected]
                return result
            injections.append((model.transformer.h[layer].attn.c_attn, inject))
        intervention_logits, _ = forward(base, output_edits=injections)
    return base_logits, donor_logits, intervention_logits


def native_path_oracle(model, base, donor, senders, receivers, sender_mask, donor_mask, receiver_mask):
    """Four real forward passes implementing the requested output-freeze rule."""
    model.eval()
    with torch.no_grad():
        base_logits = model(**base)["logits"]
        clean = model.records[-1]
        donor_logits = model(**donor)["logits"]
        changed = model.records[-1]
        edits = []
        for layer, block in enumerate(model.blocks):
            def freeze(value, layer=layer):
                result = clean["heads"][layer].clone()
                for sender_layer, head in senders:
                    if sender_layer != layer:
                        continue
                    for batch in range(result.shape[0]):
                        result[batch, sender_mask[batch], head] = changed["heads"][layer][batch, donor_mask[batch], head]
                return result
            edits.append((block.head_out, freeze))
        with native_hooks(edits):
            model(**base)
        controlled = model.records[-1]
        edits = []
        if receivers == "residual":
            def inject_final(value):
                result = value.clone()
                result[receiver_mask] = controlled["final"][receiver_mask]
                return result
            edits.append((model.final_residual, inject_final))
        else:
            groups = {}
            for layer, head, kind in receivers:
                groups.setdefault((layer, kind), []).append(head)
            for (layer, kind), heads in groups.items():
                def inject(value, layer=layer, kind=kind, heads=heads):
                    result = value.reshape(*value.shape[:2], model.blocks[layer].q.out_features // 2 if kind == "q" else model.kv_heads, 2).clone()
                    for head in heads:
                        result[:, :, head][receiver_mask] = controlled[kind][layer][:, :, head][receiver_mask]
                    return result.reshape_as(value)
                edits.append((getattr(model.blocks[layer], kind), inject))
        with native_hooks(edits):
            intervention_logits = model(**base)["logits"]
    return base_logits, donor_logits, intervention_logits


class PathPatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(194)
            self.model = CircuitModel()
        self.probe = Prober(self.model)
        self.base = {"input_ids": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])}
        self.donor = {"input_ids": torch.tensor([[5, 6, 7, 8], [8, 7, 6, 5]])}
        self.metric = TokenMargin(positive=2, negative=8)
        self.end = torch.tensor([[False, False, False, True]]).expand(2, -1)

    def score(self, logits):
        return logits[:, -1, 2] - logits[:, -1, 8]

    def assert_clean(self):
        self.assertFalse(self.probe.adapter._active)
        for module in self.model.modules():
            self.assertEqual(len(module._forward_hooks), 0)
            self.assertEqual(len(module._forward_pre_hooks), 0)

    def assert_oracle(self, *, senders=((0, 0),), receivers=((2, 1, "q"),),
                      sender_mask=None, donor_mask=None, receiver_mask=None):
        sender_mask = self.end if sender_mask is None else sender_mask
        donor_mask = sender_mask if donor_mask is None else donor_mask
        receiver_mask = self.end if receiver_mask is None else receiver_mask
        result = self.probe.causal.path(senders=list(senders), receivers=receivers,
                                       sender_tokens=sender_mask, donor_tokens=donor_mask,
                                       receiver_tokens=receiver_mask).run(
            self.base, donor=self.donor, metric=self.metric, alignment="position")
        self.assertEqual(len(self.model.records), 4)
        self.assert_clean()
        actual_records = self.model.records[:]
        expected = native_path_oracle(self.model, self.base, self.donor, senders,
                                      receivers, sender_mask, donor_mask, receiver_mask)
        for name, logits in zip(("baseline_score", "donor_score", "intervention_score"), expected):
            torch.testing.assert_close(result.tensors[name], self.score(logits))
        torch.testing.assert_close(result.tensors["effect"], self.score(expected[2]) - self.score(expected[0]))
        self.assertEqual(result.tensors["effect"].shape, (2,))
        self.assertEqual(result.metadata["stages"], 4)
        self.assert_clean()
        return result, actual_records

    def test_four_real_runs_match_independent_hooks_and_have_nonzero_effect(self):
        result, records = self.assert_oracle()
        self.assertGreater(result.tensors["effect"].abs().max().item(), 1e-8)
        torch.testing.assert_close(records[2]["heads"][2], records[0]["heads"][2])
        self.assertGreater((records[2]["q"][2] - records[0]["q"][2]).abs().max().item(), 1e-8)

    def test_other_heads_frozen_while_layernorm_and_mlp_recompute(self):
        _, records = self.assert_oracle()
        base, donor, controlled, final = records
        torch.testing.assert_close(controlled["heads"][0][:, -1, 0], donor["heads"][0][:, -1, 0])
        torch.testing.assert_close(controlled["heads"][0][:, :, 1], base["heads"][0][:, :, 1])
        torch.testing.assert_close(controlled["heads"][0][:, :-1, 0], base["heads"][0][:, :-1, 0])
        for layer in (1, 2):
            torch.testing.assert_close(controlled["heads"][layer], base["heads"][layer])
        self.assertGreater((controlled["norm"][1] - base["norm"][1]).abs().max().item(), 1e-8)
        self.assertGreater((controlled["mlp"][0] - base["mlp"][0]).abs().max().item(), 1e-8)
        # The receiver-only pass starts with the original sender output; the
        # receiver's output is free to change in response to its injected Q.
        torch.testing.assert_close(final["heads"][0], base["heads"][0])
        self.assertGreater((final["heads"][2] - base["heads"][2]).abs().max().item(), 1e-8)

    def test_path_is_not_an_ordinary_residual_node_patch(self):
        result, _ = self.assert_oracle()
        patch = self.probe.causal.patch(layers=[0], tokens="last_prompt").run(
            self.base, source=self.donor, metric=self.metric, alignment="position")
        self.assertGreater((result.tensors["effect"] - patch.tensors["effect"][0]).abs().max().item(), 1e-6)

    def test_q_k_v_and_multiple_senders_and_receivers(self):
        selected = torch.tensor([[False, True, False, False]]).expand(2, -1)
        for kind in ("q", "k", "v"):
            with self.subTest(kind=kind):
                self.model.records.clear()
                self.assert_oracle(senders=((0, 0), (0, 1)),
                                   receivers=((2, 0, kind), (2, 1, kind)),
                                   sender_mask=selected, receiver_mask=selected)

    def test_donor_position_mapping_does_not_move_receiver_capture(self):
        sender = torch.tensor([[False, True, False, True], [True, False, True, False]])
        donor = torch.tensor([[True, False, True, False], [False, True, False, True]])
        receiver = torch.tensor([[False, False, False, True], [False, False, True, False]])
        _, records = self.assert_oracle(sender_mask=sender, donor_mask=donor, receiver_mask=receiver)
        base, donor_run, controlled, final = records
        for batch in range(2):
            torch.testing.assert_close(controlled["heads"][0][batch, sender[batch], 0],
                                       donor_run["heads"][0][batch, donor[batch], 0])
        torch.testing.assert_close(final["q"][2][:, :, 1][receiver], controlled["q"][2][:, :, 1][receiver])
        torch.testing.assert_close(final["q"][2][:, :, 1][~receiver], base["q"][2][:, :, 1][~receiver])

    def test_final_residual_receiver_matches_oracle(self):
        result, records = self.assert_oracle(receivers="residual")
        torch.testing.assert_close(records[3]["final"][self.end], records[2]["final"][self.end])
        self.assertGreater(result.tensors["effect"].abs().max().item(), 1e-8)

    def test_identity_donor_and_noncausal_sender_have_zero_effect(self):
        method = self.probe.causal.path(senders=[(0, 0)], receivers=[(2, 1, "q")])
        result = method.run(self.base, donor=self.base, metric=self.metric)
        torch.testing.assert_close(result.tensors["effect"], torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0)
        for receiver_layer in (0, 2):
            with self.subTest(receiver_layer=receiver_layer):
                self.model.records.clear()
                result = self.probe.causal.path(senders=[(2, 0)], receivers=[(receiver_layer, 1, "q")]).run(
                    self.base, donor=self.donor, metric=self.metric, alignment="position")
                torch.testing.assert_close(result.tensors["effect"], torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0)

    def test_head_freezing_blocks_cross_position_q_propagation(self):
        result = self.probe.causal.path(senders=[(0, 0)], receivers=[(2, 1, "q")],
                                       sender_tokens=[1], receiver_tokens=[3]).run(
            self.base, donor=self.donor, metric=self.metric, alignment="position")
        torch.testing.assert_close(result.tensors["effect"], torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0)

    def test_flattened_head_messages_match_explicit_head_axis(self):
        self.model.flattened_heads = True
        self.probe = Prober(self.model)
        self.assert_oracle()

    def test_grouped_query_kv_receivers_use_physical_head_indices(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(194)
            self.model = CircuitModel(kv_heads=1)
        self.probe = Prober(self.model)
        for kind in ("k", "v"):
            with self.subTest(kind=kind):
                self.model.records.clear()
                self.assert_oracle(receivers=((2, 0, kind),))
                with self.assertRaises(ValueError):
                    self.probe.causal.path(senders=[(0, 0)], receivers=[(2, 1, kind)])

    def test_partial_head_coverage_is_rejected_before_forward(self):
        self.probe.spec.path_heads.pop(1)
        with self.assertRaises(CapabilityError):
            self.probe.causal.path(senders=[(0, 0)])
        self.assertEqual(len(self.model.records), 0)
        self.assertFalse(self.probe.describe()["methods"]["causal.path"]["available"])

    def test_padding_and_per_example_selection_counts_are_validated(self):
        base = {**self.base, "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]])}
        donor = {**self.donor, "attention_mask": base["attention_mask"]}
        with self.assertRaises(ValueError):
            self.probe.causal.path(senders=[(0, 0)], sender_tokens=[3]).run(
                base, donor=donor, metric=self.metric, alignment="position")
        for sender, donor_mask in (
            (torch.tensor([[False, False, False, True], [False, False, False, False]]), self.end),
            (self.end, torch.tensor([[False, True, False, True], [False, False, False, True]])),
        ):
            with self.subTest(sender=sender.tolist(), donor=donor_mask.tolist()):
                with self.assertRaises(ValueError):
                    self.probe.causal.path(senders=[(0, 0)], sender_tokens=sender, donor_tokens=donor_mask).run(
                        self.base, donor=self.donor, metric=self.metric, alignment="position")
        self.assert_clean()

    def test_strict_alignment_preserves_guard_on_changed_token_ids(self):
        with self.assertRaises(ValueError):
            self.probe.causal.path(senders=[(0, 0)]).run(self.base, donor=self.donor, metric=self.metric)
        self.assert_clean()

    def test_failure_removes_control_hooks_and_restores_training_flags(self):
        self.model.train()
        self.model.blocks[1].norm1.eval()
        before = [module.training for module in self.model.modules()]
        for failed_stage in (3, 4):
            with self.subTest(failed_stage=failed_stage):
                self.model.records.clear()
                self.model.fail_run = failed_stage
                with self.assertRaisesRegex(RuntimeError, "intentional"):
                    self.probe.causal.path(senders=[(0, 0)]).run(
                        self.base, donor=self.donor, metric=self.metric, alignment="position")
                self.assert_clean()
                self.assertEqual([module.training for module in self.model.modules()], before)
        self.model.fail_run = None
        self.model.records.clear()
        retry = self.probe.causal.path(senders=[(0, 0)]).run(
            self.base, donor=self.base, metric=self.metric)
        torch.testing.assert_close(retry.tensors["effect"], torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0)

    def test_scalar_metric_and_result_serialization(self):
        import json
        from pathlib import Path
        import tempfile

        result = self.probe.causal.path(senders=[(0, 0)]).run(
            self.base, donor=self.donor, metric=lambda logits: self.score(logits).mean(), alignment="position")
        self.assertEqual(result.tensors["effect"].shape, ())
        with tempfile.TemporaryDirectory() as directory:
            result.save(directory)
            metadata = json.loads((Path(directory) / "metadata.json").read_text())
            self.assertEqual(metadata["stages"], 4)

    def test_per_example_metric_length_must_match_batch(self):
        with self.assertRaises(ValueError):
            self.probe.causal.path(senders=[(0, 0)]).run(
                self.base, donor=self.donor, metric=lambda logits: logits.new_zeros(3), alignment="position")
        self.assert_clean()


@unittest.skipUnless(importlib.util.find_spec("transformers"), "optional transformers extra not installed")
class NativePathPatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_gpt2_packed_qkv_matches_native_hook_oracle(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(73)
            config = GPT2Config(vocab_size=13, n_embd=8, n_layer=2, n_head=2,
                                n_positions=8, use_cache=False, attn_pdrop=0,
                                resid_pdrop=0, embd_pdrop=0)
            config._attn_implementation = "eager"
            model = GPT2LMHeadModel(config).eval()
        probe = Prober(model)
        base = {"input_ids": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])}
        donor = {"input_ids": torch.tensor([[5, 6, 7, 8], [8, 7, 6, 5]])}
        selected = torch.tensor([[False, True, False, True]]).expand(2, -1)
        senders = [(0, 0)]
        receivers = [(1, 0, "q"), (1, 1, "k"), (1, 0, "v")]
        method = probe.causal.path(senders=senders, receivers=receivers,
                                   sender_tokens=selected, receiver_tokens=selected)
        metric = TokenMargin(2, 8)
        result = method.run(base, donor=donor, metric=metric, alignment="position")
        expected = native_gpt2_oracle(model, base, donor, senders, receivers, selected)
        for name, logits in zip(("baseline_score", "donor_score", "intervention_score"), expected):
            torch.testing.assert_close(result.tensors[name], logits[:, -1, 2] - logits[:, -1, 8])
        self.assertGreater(result.tensors["effect"].abs().max().item(), 1e-7)
        identity = method.run(base, donor=base, metric=metric)
        torch.testing.assert_close(identity.tensors["effect"], torch.zeros(2), atol=1e-7, rtol=0)
        for module in model.modules():
            self.assertEqual(len(module._forward_hooks), 0)
            self.assertEqual(len(module._forward_pre_hooks), 0)

    def test_llama_sdpa_path_and_physical_grouped_query_kv(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(73)
            config = LlamaConfig(vocab_size=13, hidden_size=8, intermediate_size=16,
                                 num_hidden_layers=2, num_attention_heads=2,
                                 num_key_value_heads=1, use_cache=False)
            config._attn_implementation = "sdpa"
            model = LlamaForCausalLM(config).eval()
        probe = Prober(model)
        self.assertTrue(probe.describe()["methods"]["causal.path"]["available"])
        self.assertFalse(probe.describe()["methods"]["attention.profile"]["available"])
        base = {"input_ids": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]),
                "attention_mask": torch.ones(2, 4, dtype=torch.long)}
        donor = {**base, "input_ids": torch.tensor([[5, 6, 7, 8], [8, 7, 6, 5]])}
        for kind in ("k", "v"):
            with self.subTest(kind=kind):
                method = probe.causal.path(senders=[(0, 1)], receivers=[(1, 0, kind)],
                                           sender_tokens=[1], receiver_tokens=[1])
                identity = method.run(base, donor=base, metric=TokenMargin(2, 8))
                torch.testing.assert_close(identity.tensors["effect"], torch.zeros(2), atol=1e-7, rtol=0)
                actual = method.run(base, donor=donor, metric=TokenMargin(2, 8), alignment="position")
                self.assertGreater(actual.tensors["effect"].abs().max().item(), 1e-8)
                with self.assertRaises(ValueError):
                    probe.causal.path(senders=[(0, 1)], receivers=[(1, 1, kind)])
        for module in model.modules():
            self.assertEqual(len(module._forward_hooks), 0)
            self.assertEqual(len(module._forward_pre_hooks), 0)


if __name__ == "__main__":
    unittest.main()
