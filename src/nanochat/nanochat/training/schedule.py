"""Small scheduling helpers shared by nanochat training scripts."""


def should_eval_base(step: int, eval_every: int, eval_at_start: bool, last_step: bool) -> bool:
    """Return whether base pretraining should run validation at this step."""
    return eval_every > 0 and (
        last_step or (step > 0 and step % eval_every == 0) or (step == 0 and eval_at_start)
    )


def should_eval_sft(step: int, eval_every: int, eval_at_start: bool, last_step: bool) -> bool:
    """Return whether SFT should run validation at this step."""
    return last_step or (
        eval_every > 0 and ((step > 0 and step % eval_every == 0) or (step == 0 and eval_at_start))
    )


def eval_step_count(
    eval_tokens: int,
    device_batch_size: int,
    max_seq_len: int,
    ddp_world_size: int,
) -> int:
    """Return validation loader steps, always at least one."""
    tokens_per_eval_step = device_batch_size * max_seq_len * ddp_world_size
    return max(1, eval_tokens // tokens_per_eval_step)
