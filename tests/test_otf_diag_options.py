"""Opt-in feedback alignment, alternating blocks, and training seed.

Unset configs must match the historical forward and the equal-block selector.
"""

from __future__ import annotations

import random
from io import StringIO

import numpy as np
import pytest
import torch

from dvae.learning_algo import (
    autonomous_selector_kwargs,
    block_lengths_for_sampling,
    resolve_training_seed,
    seed_training_rng,
)
from dvae.model.mt_rnn import build_MT_RNN
from dvae.model.mt_vrnn import build_MT_VRNN
from dvae.model.rnn import build_RNN
from dvae.model.vrnn import build_VRNN
from dvae.utils.model_mode_selector import (
    create_autonomous_mode_selector,
    create_autonomous_mode_selector_1d,
)
from dvae.utils.read_config import myconf


BUILDERS = {"RNN": build_RNN, "MT_RNN": build_MT_RNN}


def _cfg(name="RNN", feedback=None, **training):
    feedback_line = (
        "" if feedback is None else "feedback_alignment = {}\n".format(feedback)
    )
    training_lines = "".join(
        "{} = {}\n".format(key, value) for key, value in training.items()
    )
    alphas = "0.2, 0.8" if name == "MT_RNN" else ""
    text = """
[Network]
name = {name}
tag = {name}
activation = tanh
x_dim = 2
dense_x =
dense_h_x =
dim_rnn = 5
num_rnn = 1
type_rnn = RNN
dropout_p = 0
alphas = {alphas}
{feedback_line}
[Training]
beta = 1
{training_lines}
""".format(
        name=name,
        alphas=alphas,
        feedback_line=feedback_line,
        training_lines=training_lines,
    )
    cfg = myconf()
    cfg.read_file(StringIO(text))
    return cfg


def _build(name, feedback=None, seed=0):
    torch.manual_seed(seed)
    model = BUILDERS[name](_cfg(name, feedback=feedback), device="cpu")
    model.train()
    return model


def _copy_weights(src, name, feedback):
    copied = _build(name, feedback=feedback, seed=seed_offset(name))
    copied.load_state_dict(src.state_dict())
    copied.train()
    return copied


def seed_offset(name):
    return 11 if name == "MT_RNN" else 3


def _mode(seq_len, mode):
    return create_autonomous_mode_selector_1d(seq_len, mode=mode)


def _inputs(seq_len, batch, x_dim, with_nans, seed=123):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(seq_len, batch, x_dim, generator=generator)
    if with_nans:
        x = x.clone()
        x[0, 0, 0] = float("nan")
        x[3, 1, 1] = float("nan")
        x[5, :, 0] = float("nan")
    return x


def delayed_reference_forward(model, x, mode_selector):
    """Historical BaseRNN order: step t feeds back y[t-1], then writes y[t]."""
    seq_len, batch_size, _ = x.shape
    y = torch.zeros((seq_len, batch_size, model.y_dim), device=model.device)
    h_t = torch.zeros(
        model.num_rnn, batch_size, model.dim_rnn, device=model.device
    )
    c_t = None
    if model.type_rnn == "LSTM":
        c_t = torch.zeros_like(h_t)
    mode_selector = model.prepare_mode_selector(
        mode_selector, seq_len, batch_size, model.x_dim, model.device
    )
    noise_selector = model.prepare_mode_selector(
        None, seq_len, batch_size, model.x_dim, model.device
    )
    running_mean_t = torch.zeros(batch_size, model.x_dim, device=model.device)
    running_var_t = torch.ones(batch_size, model.x_dim, device=model.device)
    local_std = torch.sqrt(running_var_t + 1e-6)
    base_noise = torch.randn(seq_len, batch_size, model.x_dim, device=model.device)
    for t in range(seq_len):
        input_t = x[t, :, :]
        mode_selector_t = mode_selector[t, :, :]
        noise_selector_t = noise_selector[t, :, :]
        noise_t = base_noise[t] * local_std * model.noise_std_factor
        if model.impute_with_which == "auto":
            fill_value = y_t if t > 0 else torch.zeros_like(input_t)
        elif model.impute_with_which == "noise":
            fill_value = noise_t
        imputed_input_t = model.impute_inputs_nans_with_output(
            input_t, fill_value, t
        )
        running_mean_t, running_var_t, local_std = model._update_running_stats(
            imputed_input_t, running_mean_t, running_var_t
        )
        mixed_input_t = model.mix_inputs_with_outputs(
            imputed_input_t,
            y_t if t > 0 else torch.zeros_like(input_t),
            mode_selector_t,
            t,
        )
        final_input_t = model.mix_inputs_with_outputs(
            mixed_input_t, noise_t, noise_selector_t, t
        )
        feature_xt = model.feature_extractor_x(final_input_t)
        h_t_last = h_t[-1]
        y_t = model.generation_x(h_t_last)
        y[t, :, :] = y_t
        if model.type_rnn == "LSTM":
            h_t, c_t = model.recurrence(feature_xt.unsqueeze(0), h_t, c_t)
        else:
            h_t, _ = model.recurrence(feature_xt.unsqueeze(0), h_t)
    return y


def _capture_forward(model, x, mode_selector):
    seen = []

    def _hook(_module, inputs, _output):
        seen.append(inputs[0].detach().clone())

    handle = model.feature_extractor_x.register_forward_hook(_hook)
    try:
        y = model(x, mode_selector=mode_selector)
    finally:
        handle.remove()
    fed = torch.stack(seen, dim=0)
    return y, fed


@pytest.fixture
def _restore_rng():
    torch_state = torch.get_rng_state()
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    cuda_state = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    yield
    torch.set_rng_state(torch_state)
    np.random.set_state(numpy_state)
    random.setstate(python_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


@pytest.mark.parametrize("builder_name", ["RNN", "MT_RNN"])
@pytest.mark.parametrize("mode_name", ["all_0", "all_1", "half_half"])
@pytest.mark.parametrize("with_nans", [False, True])
def test_delayed_matches_unset_and_historical_loop(
    builder_name, mode_name, with_nans, _restore_rng
):
    unset = _build(builder_name, feedback=None, seed=0)
    delayed = _copy_weights(unset, builder_name, feedback="delayed")
    assert unset.feedback_alignment == "delayed"
    assert delayed.feedback_alignment == "delayed"
    assert "feedback_alignment: delayed" in "\n".join(unset.get_info())

    seq_len, batch, x_dim = 8, 2, 2
    x = _inputs(seq_len, batch, x_dim, with_nans)
    mode = _mode(seq_len, mode_name)

    with torch.no_grad():
        torch.manual_seed(100)
        reference = delayed_reference_forward(unset, x, mode)
        torch.manual_seed(100)
        y_unset, fed_unset = _capture_forward(unset, x, mode)
        torch.manual_seed(100)
        y_delayed, fed_delayed = _capture_forward(delayed, x, mode)

    assert torch.equal(y_unset, reference)
    assert torch.equal(y_delayed, y_unset)
    assert torch.equal(fed_delayed, fed_unset)


@pytest.mark.parametrize("builder_name", ["RNN", "MT_RNN"])
def test_aligned_autonomous_feeds_current_output(builder_name, _restore_rng):
    delayed = _build(builder_name, feedback="delayed", seed=1)
    aligned = _copy_weights(delayed, builder_name, feedback="aligned")
    assert aligned.feedback_alignment == "aligned"
    assert "feedback_alignment: aligned" in "\n".join(aligned.get_info())

    seq_len, batch, x_dim = 8, 2, 2
    x = _inputs(seq_len, batch, x_dim, with_nans=False, seed=9)
    mode = _mode(seq_len, "all_1")

    torch.manual_seed(100)
    y_aligned, fed_aligned = _capture_forward(aligned, x, mode)
    torch.manual_seed(100)
    y_delayed, fed_delayed = _capture_forward(delayed, x, mode)

    # Pure autonomous: the tensor consumed at t>0 is y[t], not y[t-1].
    assert torch.equal(fed_aligned[1:], y_aligned[1:])
    assert not torch.equal(y_aligned[1:], y_aligned[:-1])
    assert torch.equal(fed_delayed[1:], y_delayed[:-1])
    assert not torch.equal(fed_delayed[1:], y_delayed[1:])

    y_aligned.sum().backward()
    assert aligned.gen_out.weight.grad is not None


@pytest.mark.parametrize("builder_name", ["RNN", "MT_RNN"])
def test_aligned_teacher_forcing_without_nans_matches_delayed(
    builder_name, _restore_rng
):
    delayed = _build(builder_name, feedback=None, seed=2)
    aligned = _copy_weights(delayed, builder_name, feedback="aligned")
    x = _inputs(6, 2, 2, with_nans=False, seed=4)
    mode = _mode(6, "all_0")

    with torch.no_grad():
        torch.manual_seed(50)
        y_delayed, fed_delayed = _capture_forward(delayed, x, mode)
        torch.manual_seed(50)
        y_aligned, fed_aligned = _capture_forward(aligned, x, mode)

    assert torch.equal(fed_delayed, x)
    assert torch.equal(fed_aligned, x)
    assert torch.equal(y_aligned, y_delayed)


@pytest.mark.parametrize("builder_name", ["RNN", "MT_RNN"])
def test_aligned_imputes_nans_with_current_output(builder_name, _restore_rng):
    """Teacher forcing with a hole: the fill at t>0 is y[t], not y[t-1]."""
    delayed = _build(builder_name, feedback="delayed", seed=5)
    aligned = _copy_weights(delayed, builder_name, feedback="aligned")
    x = _inputs(8, 2, 2, with_nans=False, seed=6)
    x = x.clone()
    x[4, 0, 1] = float("nan")
    mode = _mode(8, "all_0")

    with torch.no_grad():
        torch.manual_seed(7)
        y_delayed, fed_delayed = _capture_forward(delayed, x, mode)
        torch.manual_seed(7)
        y_aligned, fed_aligned = _capture_forward(aligned, x, mode)

    assert torch.equal(fed_aligned[4, 0, 1], y_aligned[4, 0, 1])
    assert torch.equal(fed_delayed[4, 0, 1], y_delayed[3, 0, 1])
    assert not torch.equal(fed_aligned[4, 0, 1], fed_delayed[4, 0, 1])


def test_invalid_feedback_alignment_raises():
    with pytest.raises(ValueError, match="feedback_alignment"):
        _build("RNN", feedback="sideways")
    with pytest.raises(ValueError, match="feedback_alignment"):
        _build("MT_RNN", feedback="Aligned")


def test_vrnn_rejects_aligned_feedback():
    cfg = myconf()
    cfg.read_file(StringIO("[Network]\nfeedback_alignment = aligned\n"))
    with pytest.raises(NotImplementedError, match="not implemented"):
        build_VRNN(cfg)
    with pytest.raises(NotImplementedError, match="not implemented"):
        build_MT_VRNN(cfg)

    blank = myconf()
    blank.read_file(StringIO("[Network]\nfeedback_alignment =\n"))
    # Refusal runs before the VRNN constructor. Blank must not raise here.
    from dvae.model.base_rnn import reject_aligned_feedback

    reject_aligned_feedback(blank)
    reject_aligned_feedback(_cfg("RNN", feedback=None))


def _legacy_equal_blocks(seq_len, block_len):
    return np.array(
        [1.0 if (i // block_len) % 2 == 1 else 0.0 for i in range(seq_len)],
        dtype=np.float32,
    )


def test_alternating_blocks_pattern_and_default():
    expected = np.array(
        [0, 0, 1, 1, 1, 0, 0, 1, 1, 1], dtype=np.float32
    )
    got = create_autonomous_mode_selector_1d(
        10, mode="alternating_blocks", block_len=2, auto_block_len=3
    )
    assert np.array_equal(got, expected)

    wrapped = create_autonomous_mode_selector(
        10,
        mode="alternating_blocks",
        block_len=2,
        auto_block_len=3,
        batch_size=2,
        x_dim=1,
    )
    assert torch.equal(wrapped[:, 0, 0], torch.from_numpy(expected))
    assert torch.equal(wrapped[:, 1, 0], torch.from_numpy(expected))

    # A longer window extends the same pattern; blocks are not rescaled.
    short = create_autonomous_mode_selector_1d(
        5, mode="alternating_blocks", block_len=2, auto_block_len=3
    )
    long = create_autonomous_mode_selector_1d(
        20, mode="alternating_blocks", block_len=2, auto_block_len=3
    )
    assert np.array_equal(long[:5], short)
    assert np.array_equal(short, expected[:5])

    for length in (1, 7, 20):
        legacy = _legacy_equal_blocks(length, 5)
        assert np.array_equal(
            create_autonomous_mode_selector_1d(
                length, mode="alternating_blocks", block_len=5
            ),
            legacy,
        )
        assert np.array_equal(
            create_autonomous_mode_selector_1d(
                length,
                mode="alternating_blocks",
                block_len=5,
                auto_block_len=None,
            ),
            legacy,
        )
        assert np.array_equal(
            create_autonomous_mode_selector_1d(
                length, mode="alternating_blocks", block_len=5, auto_block_len=5
            ),
            legacy,
        )

    half = create_autonomous_mode_selector_1d(9, mode="half_half")
    assert np.array_equal(
        half, np.array([0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.float32)
    )


def test_block_lengths_read_only_for_alternating_blocks():
    class _Guard(myconf):
        def get(self, section, option, *args, **kwargs):
            if option in ("tf_block_len", "auto_block_len"):
                raise AssertionError("read {}".format(option))
            return super().get(section, option, *args, **kwargs)

    guarded = _Guard()
    guarded.read_file(StringIO("[Training]\nsampling_method = ss\n"))
    assert block_lengths_for_sampling(guarded, "ss") is None
    assert block_lengths_for_sampling(guarded, "ptf") is None

    missing = _cfg("RNN")
    assert block_lengths_for_sampling(missing, "alternating_blocks") == (100, 400)

    explicit = _cfg("RNN", tf_block_len=2, auto_block_len=3)
    assert block_lengths_for_sampling(explicit, "alternating_blocks") == (2, 3)

    blank_auto = _cfg("RNN", tf_block_len=7, auto_block_len="")
    assert block_lengths_for_sampling(blank_auto, "alternating_blocks") == (7, 400)

    assert autonomous_selector_kwargs({"mode": "bernoulli_sampling"}) == {}
    assert autonomous_selector_kwargs(
        {"block_len": 2, "auto_block_len": 3}
    ) == {"block_len": 2, "auto_block_len": 3}


def test_same_seed_builds_identical_parameters(_restore_rng):
    class _Boom:
        def get(self, *args, **kwargs):
            raise AssertionError("reload must not read Training.seed")

    assert resolve_training_seed(_Boom(), reload=True) is None
    assert resolve_training_seed(_cfg("RNN"), reload=False) is None
    assert resolve_training_seed(_cfg("RNN", seed=""), reload=False) is None
    assert resolve_training_seed(_cfg("RNN", seed="7"), reload=False) == 7
    assert resolve_training_seed(_cfg("RNN", seed="7"), reload=True) is None
    with pytest.raises(ValueError, match="Training.seed"):
        resolve_training_seed(_cfg("RNN", seed="nope"), reload=False)

    seed_training_rng(7)
    first_draw = (random.random(), float(np.random.rand()), torch.rand(1).item())
    seed_training_rng(7)
    second_draw = (random.random(), float(np.random.rand()), torch.rand(1).item())
    assert first_draw == second_draw

    seed_training_rng(7)
    first = build_RNN(_cfg("RNN"), device="cpu")
    seed_training_rng(7)
    second = build_RNN(_cfg("RNN"), device="cpu")
    for left, right in zip(first.parameters(), second.parameters()):
        assert torch.equal(left, right)

    seed_training_rng(8)
    other = build_RNN(_cfg("RNN"), device="cpu")
    assert any(
        not torch.equal(left, right)
        for left, right in zip(first.parameters(), other.parameters())
    )

    if torch.cuda.is_available():
        seed_training_rng(7)
        cuda_first = torch.rand(4, device="cuda")
        seed_training_rng(7)
        cuda_second = torch.rand(4, device="cuda")
        assert torch.equal(cuda_first, cuda_second)
