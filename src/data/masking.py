from __future__ import annotations

import torch

from config import MaskingConfig


def build_mlm_inputs(
    event_value_ids: torch.Tensor,
    event_key_ids: torch.Tensor,
    event_token_mask: torch.Tensor,
    event_mask: torch.Tensor,
    event_text_mask: torch.Tensor | None,
    *,
    mask_token_id: int,
    unk_token_id: int,
    config: MaskingConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    device = event_value_ids.device
    if generator is None:
        generator = torch.Generator(device=device)

    valid_tokens = event_token_mask.bool()
    selected = torch.zeros_like(valid_tokens)

    token_draw = torch.rand(event_value_ids.shape, device=device, generator=generator)
    selected |= valid_tokens & (token_draw < config.token_mask_probability)

    event_draw = torch.rand(event_mask.shape, device=device, generator=generator)
    selected_events = event_draw < config.event_mask_probability
    selected |= valid_tokens & selected_events.unsqueeze(-1)

    batch_size = event_value_ids.shape[0]
    for batch_index in range(batch_size):
        present_keys = torch.unique(event_key_ids[batch_index][valid_tokens[batch_index]])
        for key_id in present_keys.tolist():
            key_draw = torch.rand((), device=device, generator=generator)
            if key_draw.item() < config.key_mask_probability:
                selected[batch_index] |= valid_tokens[batch_index] & (
                    event_key_ids[batch_index] == key_id
                )

    masked_value_ids = event_value_ids.clone()
    labels = torch.full_like(event_value_ids, fill_value=config.ignore_index)
    if not selected.any():
        text_target_mask = None
        if event_text_mask is not None:
            text_target_mask = torch.zeros_like(event_text_mask, dtype=torch.bool)
        return masked_value_ids, labels, text_target_mask

    unk_draw = torch.rand(event_value_ids.shape, device=device, generator=generator)
    use_unk = selected & (unk_draw < config.unk_probability)
    use_mask = selected & ~use_unk

    masked_value_ids[use_mask] = mask_token_id
    masked_value_ids[use_unk] = unk_token_id
    text_target_mask = None
    if event_text_mask is None:
        labels[use_mask] = event_value_ids[use_mask]
    else:
        non_text_mask = use_mask & ~event_text_mask
        labels[non_text_mask] = event_value_ids[non_text_mask]
        text_target_mask = use_mask & event_text_mask
    return masked_value_ids, labels, text_target_mask
