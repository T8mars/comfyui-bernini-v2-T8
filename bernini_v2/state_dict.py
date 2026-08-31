"""Authoritative split of the combined Bernini v2 checkpoint.

The Hugging Face repository stores every trainable component under
``bernini/model-*.safetensors``. Repacking strips exactly one component prefix
while preserving the upstream parameter name. Renderer keys are therefore
Diffusers Wan names at this stage; the native Wan name conversion is kept as a
separate, testable transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Component(str, Enum):
    WAN_HIGH = "wan_high"
    WAN_LOW = "wan_low"
    MLLM = "mllm"
    T5 = "t5_text_encoder"
    CONNECTOR = "connector"
    MASK_TOKENS = "mask_tokens"
    VIT_DECODER = "vit_decoder"


@dataclass(frozen=True)
class ComponentRule:
    component: Component
    prefix: str


# Longest/specific prefixes first. These names are verified against
# bernini/model.safetensors.index.json from the official v2 repository.
RULES: tuple[ComponentRule, ...] = (
    ComponentRule(Component.WAN_HIGH, "diff_dec.transformer."),
    ComponentRule(Component.WAN_LOW, "diff_dec_low.transformer_2."),
    ComponentRule(Component.MLLM, "mllm."),
    ComponentRule(Component.T5, "t5_text_encoder."),
    ComponentRule(Component.CONNECTOR, "connector."),
    ComponentRule(Component.VIT_DECODER, "vit_decoder."),
    ComponentRule(Component.MASK_TOKENS, "mask_tokens"),
)

COMPONENTS: tuple[Component, ...] = tuple(rule.component for rule in RULES)


def classify_key(key: str) -> Component:
    """Return the unique runtime component owning an upstream checkpoint key."""

    matches = [rule.component for rule in RULES if key == rule.prefix or key.startswith(rule.prefix)]
    if len(matches) != 1:
        raise ValueError(f"expected one component for {key!r}, found {matches}")
    return matches[0]


def component_key(key: str) -> tuple[Component, str]:
    """Return ``(component, key_without_upstream_component_prefix)``."""

    component = classify_key(key)
    rule = next(candidate for candidate in RULES if candidate.component == component)
    if key == rule.prefix:
        target = key
    else:
        target = key[len(rule.prefix) :]
    if not target:
        raise ValueError(f"empty target key for {key!r}")
    return component, target
