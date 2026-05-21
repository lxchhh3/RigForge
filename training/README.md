# training/

LoRA dataset construction and training scripts. Deferred — LLM module is interface-only in v1 (see PLAN.md "Decisions Locked" → DeepSeek V4 Flash + Lambda Labs cloud LoRA, blocked on weights availability).

Planned scripts (not yet implemented):
- `dataset_build.py` — emit labeled examples from curated avatars + modder-collaborator pairs + augmentations
- `train.py` — Lambda Labs LoRA recipe
- `eval.py` — holdout per-bone role accuracy + per-rig EditPlan equivalence

Present today:
- `synth_clothing.py` — synthetic clothing fixture generator (perturbs Maya.fbx for end-to-end testing without real booth assets). See task #13.
