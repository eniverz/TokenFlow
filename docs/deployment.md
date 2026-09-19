# ResVLA deployment

Deployment uses the resolved training task configuration for data transforms
and normalization, while the model and processor are loaded from a
`save_pretrained` checkpoint. No separate inference YAML is required.

## Checkpoint contents

The checkpoint directory must be produced by the current `Trainer` and contain
at least:

```text
checkpoint/step_<N>/
├── config.json
├── model.safetensors              # or sharded safetensors plus an index
├── preprocessor_config.json
└── tokenizer files and chat template
```

`config.json` is used by `AutoModel.from_pretrained`. The processor files are
used by `AutoProcessor.from_pretrained`. The optimizer directory is not needed
for inference.

The deployment process also needs the norm-stat JSON files referenced by the
task transform configuration. Those statistics are external to the model
checkpoint.

## Starting the server

The server composes the normal Hydra task configuration, resolves all
interpolations, and passes the resulting configuration to `CheckpointPolicy`.
Registering the custom ResVLA auto classes happens before model loading.

The model's `action_dim` and `action_horizon` are read from the checkpoint and
checked against the same top-level fields in the resolved task cfg. They are
not inferred from the transform list. The transform configuration is used for
norm-stat paths, repacking, input preprocessing, and action inversion.

Run from the repository root:

```bash
python -m resVLA.deployment.server \
  task=resvla_libero \
  +checkpoint_dir=results/resvla_libero_1001/checkpoint/step_10000 \
  +host=0.0.0.0 \
  +port=10093 \
  +device=cuda \
  +dtype=bf16
```

`checkpoint_dir`, `host`, `port`, `device`, `dtype`, and `idle_timeout` are
runtime Hydra overrides. They do not form a second inference configuration.

The parallel LIBERO wrapper uses the same interface:

```bash
examples/LIBERO/run_parallel_eval.sh \
  --task resvla_libero \
  --checkpoint-dir results/resvla_libero_1001/checkpoint/step_10000 \
  --gpus 0,1
```

## Input request

The WebSocket request payload accepts the raw observation structure expected by
the training dataset's `RepackTransform`:

```python
{
    "data": {
        "observation": {
            "state": state_array,
            "images": {
                "image": camera_image,
                "wrist_image": wrist_image,
            },
        },
        "task": "open the drawer",
    },
    "embodiment_name": "franka",
    "execution_horizon": 8,
}
```

The exact raw keys come from the task's `RepackTransform`. For the default
LIBERO task, the mapping is:

```text
observation.state              -> states
observation.images.image       -> images.cam_high
observation.images.wrist_image -> images.cam_wrist
task                            -> prompt
```

For compatibility with the existing LIBERO client, the server also accepts a
flat request containing `images`, `state`, and `prompt`; it repacks those values
using the configured mapping before running the normal transform chain.

The image dictionary order must match training order because the processor
encodes images in `images.values()` order.

## Input construction

For the selected embodiment, the policy instantiates the transform list from
the task cfg and applies it in the training order. This includes:

1. repacking raw environment fields;
2. prompt and CoT prompt transforms;
3. embodiment injection;
4. state normalization;
5. state padding;
6. image conversion and resize.

Transforms that require actions skip their action branch when the raw
observation has no `actions` field. The resulting images and prompt are passed
to `ResVLAProcessor.encode_input`, and the model receives:

```python
{
    "vlm_inputs": processor_output,
    "states": normalized_padded_state,  # [1, 1, action_dim]
    "embodiment": ["franka"],
}
```

## Output construction

The model returns normalized, padded actions. The policy applies the inverse of
the configured action transforms in strict reverse order:

```text
model output
  -> remove action padding using the configured action mask
  -> unnormalize actions
  -> undo delta/absolute action conversion when configured
  -> truncate to execution_horizon
```

For a forward chain such as:

```text
ToDeltaAction -> Normalizer -> PadStateAction
```

the inverse is:

```text
unpad -> Unnormalizer -> delta_to_absolute
```

The current implementation keeps the transformed state so that unnormalizing
the state can provide the correct coordinate system for the final delta/absolute
inverse. The unpadding operation is a function, not a separate transform class.

## Configuration assumptions

The current deployment resolver assumes:

- one dataset/norm-stat definition per embodiment;
- every deployment dataset transform contains an embodiment injection;
- every deployment dataset transform contains a normalizer;
- the top-level cfg `action_dim` and `action_horizon` agree with the checkpoint;
- `PadStateAction` is present so the original action width can be recovered from
  the action norm mask;
- the model checkpoint's `action_dim`, `action_horizon`, and embodiment list
  agree with the training cfg;
- the norm-stat files are readable from the paths resolved by Hydra.

If multiple datasets later use the same embodiment with different normstats,
the resolver should be extended with an explicit dataset selector instead of
silently choosing one.
