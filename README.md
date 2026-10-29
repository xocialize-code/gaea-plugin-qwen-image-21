# gaea-plugin-qwen-image-21

**Qwen-Image-2.1** (a 7B single-stream block-causal DiT with a Qwen3-VL-8B conditioner and a 64-channel RGBA VAE) as
a [Gaea](https://github.com/xocialize-code/gaea) **research plugin**: the 8-step **Turbo** checkpoint and the 40-step
**base**. It is built on [GaeaPluginKit](https://github.com/xocialize-code/gaea-plugin-kit) over
[qwen-image21-swift](https://github.com/xocialize/qwen-image21-swift), runs in its own process, and speaks the Gaea
Plugin Protocol v1. Add it to Gaea by its repo URL:

```bash
gaea plugins add https://github.com/xocialize-code/gaea-plugin-qwen-image-21
```

## Licence: research and evaluation only

> **The Qwen-Image-2.1 weights are under the Qwen RESEARCH LICENSE AGREEMENT: non-commercial use only, which the
> licence defines as research or evaluation (§1(i), §2).** Commercial use needs a separate licence from Tongyi
> (model-business@notice.qwencloud.com). Don't use this plugin, or its images, in a product or for a client.

- **In Gaea it is opt-in and named-only.** The weights' licence (`LicenseRef-Qwen-Research`) is not permissive, so
  Gaea installs both models switched off: an operator turns them on, and even then the normalizer never picks them.
  A request runs one only by naming it (`"model": "qwen-image-2.1"`). Every offer also says `named_only: true`, and
  shows the licence with `commercial_use: false` and its notice.
- **The plugin code is Apache-2.0** (Gaea G-21), and qwen-image21-swift is MIT. Neither ships weights.
- **Attribution.** The weight mirrors carry the agreement and its NOTICE, as §3 asks: "Qwen is licensed under the
  Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights
  Reserved." Built with Qwen.

## What it offers

| Model | MES operation | Quality tiers | Notes |
|---|---|---|---|
| `qwen-image-2.1` | `image.generate`, `image.edit` | standard | The Turbo checkpoint: the 2.1 DiT distilled to 8 fixed steps, its own sampling grid, without guidance. About 5× the base's speed. A negative prompt does nothing (`negative_prompt_ignored`). |
| `qwen-image-2.1-base` | `image.generate`, `image.edit` | best | The original checkpoint, 40 steps. True CFG (two passes a step) with `x-qwen21.guidance` above 1 and a negative prompt. For layout-critical typography, and the parity reference. |

- **Sizes.** Multiples of 32, 256–2752 px a side. Text to image goes up to 2400×1792 in area, the largest size on the
  model card (decoded in tiles above 1024²); the default is 1024². An edit goes up to 1024² in area. Anything else is
  fitted and reported `size_snapped`. `aspect` with `megapixels` works too.
- **Edits.** 1–10 references, named `<image1>`, `<image2>`… in the instruction. The output is the first image's
  size, fitted to 1024²; the package resizes every reference to 1024² in area. Drawn from noise, so `strength` does
  nothing (`param_ignored`).
- **RGBA.** Output is an RGBA PNG with native transparency. Ask for it in the prompt ("This is an RGBA image with
  transparency. … The image has alpha channel and the background is transparent."). MES's Image profile has no
  background parameter, and the package's surfaces don't declare `background: transparent`, so there is none here.
- **`count`** on `image.generate`: 1–8 images, each with `seed + i`. A request without a seed gets one, reported in
  `effective`. `image.edit` has one output.
- **Options**, as MES extensions, on the base only: `x-qwen21.steps` (1–100, default 40) and `x-qwen21.guidance` (1–20,
  default 1, no guidance). Turbo's offers have none, and the worker reports either as `param_ignored` should one reach it.
- **Not yet offered:** int8 and int4 lanes (the package has only bf16), keeping the encoder resident between requests,
  and the PE prompt-rewriter models.

## Lanes and memory

One lane per model, `bf16` (the DiT in bf16, the VAE in fp32), with the package's footprints from
`dist/manifest.json`:

| Model | Lane | Resident | Peak while running | Gaea schedules |
|---|---|---|---|---|
| `qwen-image-2.1` | `bf16` | 15.6 GB | 43.2 GB | 58.8 GB |
| `qwen-image-2.1-base` | `bf16` | 15.6 GB | 43.2 GB | 58.8 GB |

The peak is the worst case inside the envelope, an edit from 10 references at 1024². It includes the Qwen3-VL-8B
encoder (about 17.5 GB), which loads for each request and leaves before the denoise. A 1024² generate peaks near 33 GB
in all. With Gaea's default budget (three quarters of the Mac's memory) either model needs a Mac with 96 GB or more.

## Weights

Gaea sends its models root in `configure` (MLXEngine's store by default; `gaea models root` changes it). The worker
looks there for three snapshots, and downloads a missing one on `prepare` or `materialize`:
- **`xocialize/Qwen-Image-2.1-Turbo`**: Turbo's DiT and pipeline config. An unmodified, hash-verified mirror of
  `Qwen/Qwen-Image-2.1-Turbo`, redistributed with the agreement and NOTICE under the Qwen Research licence §3.
- **`xocialize/Qwen-Image-2.1`**: the base DiT, and the fp32 VAE both models use. A mirror of `Qwen/Qwen-Image-2.1`,
  under the same licence and terms.
- **`Qwen/Qwen3-VL-8B-Instruct`** (Apache-2.0): the conditioner. The 2.1 text encoder is byte-identical to it, so the
  mirrors leave it out.

## Build and check locally

```bash
scripts/build-plugin.sh
```

```bash
CHECK_MODELS_ROOT=<a models root holding the three snapshots> scripts/check-plugin.sh
```

`check-plugin.sh` generates one 512² image with Qwen-Image-2.1 (Turbo) bf16, with every `.build` folder hidden so MLX
can find its metallib only inside the bundle. A control then shows that the bare worker, outside the bundle, can't.
`scripts/release.sh` builds, checks, notarizes with the fleet's `mlx-notary` profile, and publishes the GitHub release.
See the [demo plugin](https://github.com/xocialize-code/gaea-plugin-demo) for signing, notarization and CI secrets.

`qwen21-inproc` (`swift run -c release qwen21-inproc`; its header lists the options) is not part of the plugin. It
runs the same lane in-process through MLXServeEngine, building its request with the worker's own functions, the
reference that Gaea's runs are compared against byte for byte.
