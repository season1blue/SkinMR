import math
import weakref
from types import MethodType
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.logits_process import LogitsProcessorList
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.mistral.modeling_mistral import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.mm_utils import get_anyres_image_grid_shape
from llava.model.llava_arch import unpad_image
import ipdb

def _fit_tensor_to_shape(source: Optional[torch.Tensor], target_shape) -> Optional[torch.Tensor]:
    if source is None:
        return None
    if not isinstance(source, torch.Tensor):
        source = torch.as_tensor(source)

    target_numel = 1
    for dim in target_shape:
        target_numel *= dim

    flat_source = source.reshape(-1)
    if flat_source.numel() == 0:
        return torch.zeros(target_shape, dtype=source.dtype, device=source.device)

    repeat_count = math.ceil(target_numel / flat_source.numel())
    resized = flat_source.repeat(repeat_count)[:target_numel]
    return resized.reshape(target_shape).to(device=source.device, dtype=source.dtype)


def _get_backbone(model):
    backbone = getattr(model, "model", None)
    if backbone is not None and hasattr(backbone, "layers"):
        return backbone
    if hasattr(model, "layers"):
        return model
    return None


def _reset_runtime(mlp):
    mlp.adpt_sign = 0
    mlp.adpt_w1 = None
    mlp.adpt_w2 = None


def _compute_topk_entropy(logits: Optional[torch.Tensor], k: int = 10) -> Optional[float]:
    if logits is None or not isinstance(logits, torch.Tensor):
        return None

    if logits.dim() == 3:
        token_logits = logits[:, -1, :]
    elif logits.dim() == 2:
        token_logits = logits
    else:
        return None

    topk = min(int(k), int(token_logits.shape[-1]))
    if topk <= 1:
        return None

    topk_scores, _ = torch.topk(token_logits.float(), topk, dim=-1)
    probabilities = torch.softmax(topk_scores, dim=-1).clamp_min(1e-12)
    entropy = -(probabilities * torch.log(probabilities)).sum(dim=-1) / math.log(float(topk))
    return float(entropy.mean().item())


def _tensor_summary(tensor: Optional[torch.Tensor]):
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def _get_image_token_mask(image_token_mask: Optional[torch.Tensor], hidden_states: Optional[torch.Tensor]):
    if image_token_mask is None or not isinstance(hidden_states, torch.Tensor) or hidden_states.dim() != 3:
        return None

    img_mask = image_token_mask
    if img_mask.dim() > 2:
        img_mask = img_mask.squeeze(-1)
    if img_mask.dim() != 2 or img_mask.shape[1] != hidden_states.shape[1]:
        return None
    return img_mask.to(device=hidden_states.device, dtype=torch.bool)


def _extract_image_token_states(hidden_states: Optional[torch.Tensor], image_token_mask: Optional[torch.Tensor]):
    img_mask = _get_image_token_mask(image_token_mask, hidden_states)
    if img_mask is None or not img_mask.any():
        return None
    return hidden_states[img_mask]


def _overwrite_image_token_states(
    hidden_states: Optional[torch.Tensor],
    image_token_mask: Optional[torch.Tensor],
    replacement_tokens: Optional[torch.Tensor],
    ratio: float,
) -> Tuple[Optional[torch.Tensor], bool]:
    img_mask = _get_image_token_mask(image_token_mask, hidden_states)
    if img_mask is None or not img_mask.any() or replacement_tokens is None:
        return hidden_states, False

    current_tokens = hidden_states[img_mask]
    replacement_tokens = _fit_tensor_to_shape(replacement_tokens, current_tokens.shape)
    if replacement_tokens is None:
        return hidden_states, False

    replacement_tokens = replacement_tokens.to(device=current_tokens.device, dtype=current_tokens.dtype)
    norm = torch.mean(torch.abs(current_tokens)) / (torch.mean(torch.abs(replacement_tokens)) + 1e-6)
    mixed_tokens = current_tokens * (1.0 - ratio) + replacement_tokens * norm * ratio

    new_hidden_states = hidden_states.clone()
    new_hidden_states[img_mask] = mixed_tokens
    return new_hidden_states, True


def _ensure_mlp_forward_patch(mlp_cls):
    if getattr(mlp_cls, "_memvr_forward_patched", False):
        return

    original_forward = mlp_cls.forward

    def _patched_forward(self, x, *args, **kwargs):
        base_out = original_forward(self, x, *args, **kwargs)
        if getattr(self, "adpt_sign", 0) != 1:
            return base_out

        adpt_w1 = getattr(self, "adpt_w1", None)
        adpt_w2 = getattr(self, "adpt_w2", None)
        if adpt_w1 is None or adpt_w2 is None:
            return base_out

        ratio = float(getattr(self, "retracing_ratio", 0.0))
        if ratio <= 0.0:
            return base_out

        adapter_source = x[0] if isinstance(x, torch.Tensor) and x.dim() == 3 else x
        if not isinstance(adapter_source, torch.Tensor):
            return base_out
        if adapter_source.dim() == 1:
            adapter_source = adapter_source.unsqueeze(0)

        if isinstance(adpt_w1, torch.Tensor) and adpt_w1.dim() > 2:
            adpt_w1 = adpt_w1[0]
        if isinstance(adpt_w2, torch.Tensor) and adpt_w2.dim() > 2:
            adpt_w2 = adpt_w2[0]

        adpt_w1 = adpt_w1.to(device=adapter_source.device, dtype=adapter_source.dtype)
        adpt_w2 = adpt_w2.to(device=adapter_source.device, dtype=adapter_source.dtype)

        try:
            adapter_mid = torch.matmul(adapter_source, adpt_w1.T)
            act_fn = getattr(self, "act_fn", None)
            if callable(act_fn):
                adapter_mid = act_fn(adapter_mid)
            else:
                adapter_mid = F.silu(adapter_mid)
            adapter_out = torch.matmul(adapter_mid, adpt_w2.T)
        except Exception:
            return base_out

        if isinstance(base_out, torch.Tensor) and base_out.dim() == 3 and adapter_out.dim() == 2:
            adapter_out = adapter_out.unsqueeze(0)

        if not isinstance(base_out, torch.Tensor) or base_out.shape != adapter_out.shape:
            return base_out

        norm = torch.mean(torch.abs(base_out)) / (torch.mean(torch.abs(adapter_out)) + 1e-6)
        return base_out * (1.0 - ratio) + adapter_out * norm * ratio

    mlp_cls.forward = _patched_forward
    mlp_cls._memvr_forward_patched = True


def _inject_adapter_from_visual_seed(backbone, target_layer_idx: int, adapter_seed: Optional[torch.Tensor], ratio: float) -> bool:
    if adapter_seed is None:
        return False
    if target_layer_idx < 0 or target_layer_idx >= len(backbone.layers):
        return False

    if isinstance(adapter_seed, torch.Tensor) and adapter_seed.dim() > 2:
        adapter_seed = adapter_seed[0]
    if isinstance(adapter_seed, torch.Tensor) and adapter_seed.dim() == 1:
        adapter_seed = adapter_seed.unsqueeze(0)

    next_mlp = backbone.layers[target_layer_idx].mlp
    in_proj = getattr(next_mlp, "up_proj", None)
    out_proj = getattr(next_mlp, "down_proj", None)
    if in_proj is None or out_proj is None:
        return False

    next_w1_seed = _fit_tensor_to_shape(adapter_seed, in_proj.weight.shape)
    next_w2_seed = _fit_tensor_to_shape(adapter_seed.T, out_proj.weight.shape)
    if next_w1_seed is None or next_w2_seed is None:
        return False

    next_mlp.adpt_w1 = nn.Parameter(torch.zeros_like(next_w1_seed))
    next_mlp.adpt_w2 = nn.Parameter(torch.zeros_like(next_w2_seed))
    next_mlp.adpt_w1 += (
        torch.mean(torch.abs(in_proj.weight)) / (torch.mean(torch.abs(next_w1_seed)) + 1e-6)
    ) * next_w1_seed
    next_mlp.adpt_w2 += (
        torch.mean(torch.abs(out_proj.weight)) / (torch.mean(torch.abs(next_w2_seed)) + 1e-6)
    ) * next_w2_seed
    next_mlp.adpt_sign = 1
    next_mlp.retracing_ratio = float(ratio)
    return True


def _memvr_backbone_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    use_cache = self.config.use_cache if use_cache is None else use_cache
    output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
    output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
    return_dict = self.config.use_return_dict if return_dict is None else return_dict

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    if position_ids is None:
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        position_ids = position_ids.unsqueeze(0)

    mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
    causal_mask = mask_function(
        config=self.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    layer = 0
    entropy_list = []
    mlp0 = self.layers[0].mlp
    apply_memvr = bool(getattr(mlp0, "apply_memvr", False))
    visual_token = getattr(mlp0, "visual_token", None)
    dynamic_visual_token = None
    retracing_ratio = float(getattr(mlp0, "retracing_ratio", 0.0))
    entropy_threshold = float(getattr(mlp0, "entropy_threshold", 1.0))
    starting_layer = int(getattr(mlp0, "starting_layer", 0))
    ending_layer = int(getattr(mlp0, "ending_layer", len(self.layers) - 1))
    method = str(getattr(mlp0, "memvr_method", "memvr") or "memvr").lower()
    retrace_delay_layers = max(1, int(getattr(mlp0, "retrace_delay_layers", 1)))
    state_drift_threshold = float(getattr(mlp0, "state_drift_threshold", 0.5))
    state_drift_pooling = str(getattr(mlp0, "state_drift_pooling", "mean") or "mean").lower()
    image_token_mask = getattr(mlp0, "image_token_mask", None) if past_seen_tokens == 0 else None
    visual_retracing_event = False
    pending_token_target_layer = -1
    pending_token_seed = None
    clear_layer_idx = -1
    prev_prev_img_state = None
    prev_img_state = None
    state_drift_score = 0.0
    state_drift_ready = False
    current_image_token_mask = None

    owner_ref = getattr(self, "_memvr_owner_model_ref", None)
    owner = owner_ref() if callable(owner_ref) else None

    if owner is not None:
        existing_debug = getattr(owner, "_memvr_debug_info", {})
        if not isinstance(existing_debug, dict):
            existing_debug = {}
        should_reset_debug = (past_seen_tokens == 0) or (not existing_debug.get("forward_initialized", False))
        if should_reset_debug:
            prepare_mask_count = int(existing_debug.get("prepare_image_token_mask_count", 0) or 0)
            prepare_cached_visual_token = existing_debug.get("cached_visual_token")
            forward_mask_count = int(image_token_mask.sum().item()) if isinstance(image_token_mask, torch.Tensor) else 0
            owner._memvr_last_entropy = None
            owner._memvr_last_target_layer = -1
            owner._memvr_debug_info = {
                "enabled": bool(apply_memvr),
                "method": method,
                "entropy_threshold": float(entropy_threshold),
                "retracing_ratio": float(retracing_ratio),
                "retrace_delay_layers": int(retrace_delay_layers),
                "starting_layer": int(starting_layer),
                "ending_layer": int(ending_layer),
                "prepare_image_token_mask_count": prepare_mask_count,
                "forward_image_token_mask_count": forward_mask_count,
                "image_token_mask_count": forward_mask_count,
                "initial_visual_token": _tensor_summary(visual_token),
                "prepare_cached_visual_token": prepare_cached_visual_token,
                "prefill_past_seen_tokens": int(past_seen_tokens),
                "triggered": False,
                "trigger_layer": None,
                "target_layer": None,
                "trigger_metric": None,
                "trigger_value": None,
                "trigger_source": None,
                "injection_success": False,
                "dynamic_visual_token": None,
                "used_dynamic_visual_token": False,
                "entropy_trace": [],
                "forward_initialized": True,
            }

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if method == "memvr" and layer == pending_token_target_layer:
            hidden_states, token_injection_success = _overwrite_image_token_states(
                hidden_states,
                image_token_mask,
                pending_token_seed,
                retracing_ratio,
            )
            pending_token_target_layer = -1
            pending_token_seed = None
            if owner is not None:
                owner._memvr_debug_info["injection_success"] = bool(token_injection_success)
                owner._memvr_debug_info["injection_mode"] = "token"
                if token_injection_success:
                    owner._memvr_last_target_layer = int(layer)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        if not apply_memvr:
            layer += 1
            continue

        current_image_token_mask = _get_image_token_mask(image_token_mask, hidden_states)

        if method in {"memvr", "evo"}:
            dynamic_visual_token = _extract_image_token_states(hidden_states, current_image_token_mask)

        if method == "evo" and image_token_mask is not None and hidden_states.dim() == 3:
            img_mask = current_image_token_mask
            if img_mask is not None:
                valid_img_samples = img_mask.any(dim=1)
                if valid_img_samples.any():
                    img_mask_f = img_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
                    pooled_img_state = (hidden_states * img_mask_f).sum(dim=1) / img_mask_f.sum(dim=1).clamp_min(1.0)
                    state_drift_ready = False
                    if prev_img_state is not None and prev_prev_img_state is not None:
                        v_prev = prev_img_state - prev_prev_img_state
                        v_curr = pooled_img_state - prev_img_state
                        cos_sim = F.cosine_similarity(v_curr, v_prev, dim=-1, eps=1e-6)
                        drift_per_sample = 1.0 - cos_sim
                        drift_per_sample = torch.where(
                            valid_img_samples,
                            drift_per_sample,
                            torch.zeros_like(drift_per_sample),
                        )
                        if state_drift_pooling == "max":
                            state_drift_score = float(drift_per_sample.max().item())
                        else:
                            state_drift_score = float(drift_per_sample[valid_img_samples].mean().item())
                        state_drift_ready = True
                    prev_prev_img_state = prev_img_state
                    prev_img_state = pooled_img_state

        norm_hidden_states = self.norm(hidden_states)
        lm_head = getattr(self, "lm_head", None)
        if lm_head is None:
            lm_head = getattr(owner, "lm_head", None) if owner is not None else None
        logits = lm_head(norm_hidden_states) if lm_head is not None else None
        if isinstance(logits, torch.Tensor):
            logits = logits[:, -1, :].float()
            if logits_processor is not None and input_ids is not None:
                logits = logits_processor(input_ids, logits)

        entropy = _compute_topk_entropy(logits, k=10)
        if entropy is not None:
            entropy_list.append(f"{entropy:.3f}")
            if owner is not None:
                owner._memvr_last_entropy = float(entropy)
                owner._memvr_debug_info["entropy_trace"].append({
                    "layer": int(layer),
                    "entropy": float(entropy),
                })

        if clear_layer_idx == layer:
            _reset_runtime(self.layers[layer].mlp)
            clear_layer_idx = -1

        if entropy is None:
            layer += 1
            continue

        if method == "evo":
            trigger_hit = state_drift_ready and (state_drift_score > state_drift_threshold)
        else:
            trigger_hit = (current_image_token_mask is not None) and bool(current_image_token_mask.any()) and (entropy > entropy_threshold)

        target_layer = layer + retrace_delay_layers
        if (
            trigger_hit
            and not visual_retracing_event
            and layer > starting_layer
            and layer < ending_layer
            and target_layer < len(self.layers)
        ):
            use_dynamic_visual = dynamic_visual_token is not None
            adapter_seed = dynamic_visual_token if use_dynamic_visual else visual_token
            trigger_source = "dynamic_visual_token" if use_dynamic_visual else "visual_token"
            if method == "memvr":
                injection_success = adapter_seed is not None
                if injection_success:
                    pending_token_target_layer = int(target_layer)
                    pending_token_seed = adapter_seed
            else:
                injection_success = _inject_adapter_from_visual_seed(self, target_layer, adapter_seed, retracing_ratio)
            if owner is not None:
                owner._memvr_debug_info.update({
                    "triggered": bool(trigger_hit),
                    "trigger_layer": int(layer),
                    "target_layer": int(target_layer),
                    "trigger_metric": "state_drift" if method == "evo" else "entropy",
                    "trigger_value": float(state_drift_score) if method == "evo" else float(entropy),
                    "trigger_source": trigger_source,
                    "injection_success": bool(injection_success),
                    "injection_mode": "token" if method == "memvr" else "ffn_adapter",
                    "dynamic_visual_token": _tensor_summary(dynamic_visual_token),
                    "used_dynamic_visual_token": bool(use_dynamic_visual),
                    "adapter_seed": _tensor_summary(adapter_seed),
                })
            if injection_success:
                visual_retracing_event = True
                if method != "memvr":
                    clear_layer_idx = target_layer
                    if owner is not None:
                        owner._memvr_last_target_layer = int(target_layer)

        layer += 1

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if owner is not None:
        owner._memvr_entropy_trace = entropy_list
        owner._memvr_debug_info["final_visual_retracing_event"] = bool(visual_retracing_event)
        owner._memvr_debug_info["final_clear_layer_idx"] = int(clear_layer_idx)

    result = BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
    if not return_dict:
        return tuple(v for v in [result.last_hidden_state, result.past_key_values, result.hidden_states, result.attentions] if v is not None)
    return result


def _memvr_prepare_inputs_labels_for_multimodal(
    self,
    input_ids,
    position_ids,
    attention_mask,
    past_key_values,
    labels,
    images,
    image_sizes=None,
):
    vision_tower = self.get_vision_tower()
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        try:
            if past_key_values is not None:
                restore_cached_visual_token(self)
        except Exception:
            pass
        return input_ids, position_ids, attention_mask, past_key_values, None, labels

    if type(images) is list or images.ndim == 5:
        if type(images) is list:
            images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
        concat_images = torch.cat([image for image in images], dim=0)
        image_features = self.encode_images(concat_images)
        split_sizes = [image.shape[0] for image in images]
        image_features = torch.split(image_features, split_sizes, dim=0)
        mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
        image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
        if mm_patch_merge_type == "flat":
            image_features = [x.flatten(0, 1) for x in image_features]
        elif mm_patch_merge_type.startswith("spatial"):
            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                if image_feature.shape[0] > 1:
                    base_image_feature = image_feature[0]
                    image_feature = image_feature[1:]
                    height = width = self.get_vision_tower().num_patches_per_side
                    assert height * width == base_image_feature.shape[0]
                    if image_aspect_ratio == "anyres":
                        num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                            image_sizes[image_idx],
                            self.config.image_grid_pinpoints,
                            self.get_vision_tower().config.image_size,
                        )
                        image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                    else:
                        raise NotImplementedError
                    if "unpad" in mm_patch_merge_type:
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(image_feature, image_sizes[image_idx])
                        image_feature = torch.cat(
                            (
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device),
                            ),
                            dim=-1,
                        )
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    else:
                        image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                        image_feature = image_feature.flatten(0, 3)
                    image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                else:
                    image_feature = image_feature[0]
                    if "unpad" in mm_patch_merge_type:
                        image_feature = torch.cat((image_feature, self.model.image_newline[None].to(image_feature.device)), dim=0)
                new_image_features.append(image_feature)
            image_features = new_image_features
        else:
            raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
    else:
        image_features = self.encode_images(images)

    if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
        raise NotImplementedError

    raw_labels = labels
    raw_position_ids = position_ids
    raw_attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    new_input_embeds = []
    new_labels = []
    new_image_masks = []
    cur_image_idx = 0
    cached_visual_token = None

    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        if num_images == 0:
            cur_image_features = image_features[cur_image_idx]
            cached_visual_token = cur_image_features
            cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels[batch_idx])
            new_image_masks.append(torch.zeros(cur_input_embeds.shape[0], dtype=torch.bool, device=cur_input_embeds.device))
            cur_image_idx += 1
            continue

        image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

        cur_new_input_embeds = []
        cur_new_labels = []
        cur_new_image_mask = []

        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            cur_new_image_mask.append(
                torch.zeros(cur_input_embeds_no_im[i].shape[0], dtype=torch.bool, device=cur_input_embeds_no_im[i].device)
            )
            if i < num_images:
                cur_image_features = image_features[cur_image_idx]
                cached_visual_token = cur_image_features
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(
                    torch.full(
                        (cur_image_features.shape[0],),
                        IGNORE_INDEX,
                        device=cur_labels.device,
                        dtype=cur_labels.dtype,
                    )
                )
                cur_new_image_mask.append(
                    torch.ones(cur_image_features.shape[0], dtype=torch.bool, device=cur_image_features.device)
                )

        cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)
        cur_new_image_mask = torch.cat(cur_new_image_mask)

        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)
        new_image_masks.append(cur_new_image_mask)

    tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
    if tokenizer_model_max_length is not None:
        new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
        new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
        new_image_masks = [x[:tokenizer_model_max_length] for x in new_image_masks]

    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)

    new_input_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    image_token_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=new_labels[0].device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

    for i, (cur_new_embed, cur_new_labels, cur_new_image_mask) in enumerate(zip(new_input_embeds, new_labels, new_image_masks)):
        cur_len = cur_new_embed.shape[0]
        if getattr(self.config, "tokenizer_padding_side", "right") == "left":
            pad_embed = torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
            new_input_embeds_padded.append(torch.cat((pad_embed, cur_new_embed), dim=0))
            if cur_len > 0:
                new_labels_padded[i, -cur_len:] = cur_new_labels
                image_token_mask[i, -cur_len:] = cur_new_image_mask
                attention_mask[i, -cur_len:] = True
                position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
        else:
            pad_embed = torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
            new_input_embeds_padded.append(torch.cat((cur_new_embed, pad_embed), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                image_token_mask[i, :cur_len] = cur_new_image_mask
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

    if raw_labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if raw_attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=raw_attention_mask.dtype)

    if raw_position_ids is None:
        position_ids = None

    if cached_visual_token is not None:
        set_cached_visual_token(self, cached_visual_token)
        self.model.layers[0].mlp.image_token_mask = image_token_mask
        self._memvr_debug_info = getattr(self, "_memvr_debug_info", {})
        self._memvr_debug_info.update({
            "cached_visual_token": _tensor_summary(cached_visual_token),
            "prepare_image_token_mask_count": int(image_token_mask.sum().item()),
            "image_token_mask_count": int(image_token_mask.sum().item()),
        })

    return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


def apply_memvr_llava(
    model,
    starting_layer,
    ending_layer,
    entropy_threshold,
    retracing_ratio,
    retrace_delay_layers=1,
    retrace_target_layers="",
    method="memvr",
    state_drift_threshold=0.5,
    state_drift_pooling="mean",
):
    backbone = _get_backbone(model)
    if backbone is None or not hasattr(backbone, "layers"):
        raise RuntimeError("Unable to locate llava language backbone layers for memvr.")

    _ensure_mlp_forward_patch(backbone.layers[0].mlp.__class__)

    backbone._memvr_owner_model_ref = weakref.ref(model)
    backbone.forward = MethodType(_memvr_backbone_forward, backbone)
    model.prepare_inputs_labels_for_multimodal = MethodType(_memvr_prepare_inputs_labels_for_multimodal, model)

    model._llava_memvr_enabled = True
    model._memvr_last_entropy = None
    model._memvr_last_target_layer = -1
    model._memvr_entropy_trace = []

    for layer in backbone.layers:
        mlp = layer.mlp
        mlp.apply_memvr = True
        mlp.visual_token = None
        mlp.retracing_ratio = float(retracing_ratio)
        mlp.entropy_threshold = float(entropy_threshold)
        mlp.starting_layer = int(starting_layer)
        mlp.ending_layer = int(ending_layer)
        mlp.retrace_delay_layers = max(1, int(retrace_delay_layers))
        mlp.retrace_target_layers = str(retrace_target_layers or "")
        mlp.memvr_method = str(method or "memvr").lower()
        mlp.state_drift_threshold = float(state_drift_threshold)
        mlp.state_drift_pooling = str(state_drift_pooling or "mean")
        mlp.image_token_mask = None
        _reset_runtime(mlp)


def set_cached_visual_token(model, visual_token: Optional[torch.Tensor]):
    if visual_token is None:
        return

    backbone = _get_backbone(model)
    if backbone is None or not hasattr(backbone, "layers") or len(backbone.layers) == 0:
        return

    token = visual_token
    if isinstance(token, torch.Tensor) and token.dim() > 2:
        token = token[0]
    if isinstance(token, torch.Tensor) and token.dim() == 1:
        token = token.unsqueeze(0)

    model._memvr_cached_visual_token = token
    backbone.layers[0].mlp.visual_token = token


def restore_cached_visual_token(model):
    cached = getattr(model, "_memvr_cached_visual_token", None)
    if cached is None:
        return
    backbone = _get_backbone(model)
    if backbone is None or not hasattr(backbone, "layers") or len(backbone.layers) == 0:
        return
    backbone.layers[0].mlp.visual_token = cached


def prepare_memvr_step(model):
    return


def update_memvr_entropy_from_logits(model, logits: Optional[torch.Tensor]):
    if not bool(getattr(model, "_llava_memvr_enabled", False)):
        return
    entropy = _compute_topk_entropy(logits, k=10)
    if entropy is not None:
        model._memvr_last_entropy = float(entropy)
