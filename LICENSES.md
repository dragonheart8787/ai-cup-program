# Third-party resources and licenses

Date checked: 2026-10-13. Nothing here is legal advice. Entries marked UNKNOWN or
"not verified" must be re-checked on the official page before anything is submitted.
Whether the competition rules restrict licenses, external models or pre-trained
weights is **UNKNOWN / NEED OFFICIAL FILE** (the rules document has not been uploaded).

| resource | URL | version | license | usage | date checked |
|---|---|---|---|---|---|
| Ultralytics (YOLO code) | https://github.com/ultralytics/ultralytics | `__version__ = "8.4.176"` on `main` (read from `ultralytics/__init__.py`); PyPI release not checked | **AGPL-3.0** (repo `LICENSE` file is the GNU AGPL v3 text); README also offers a separate paid *Ultralytics Enterprise License* | Candidate detector / instance-segmentation baseline. NOT in `requirements.txt`. See the AGPL section below | 2026-10-13 |
| SegFormer code (upstream) | https://github.com/NVlabs/SegFormer | not checked | **NVIDIA Source Code License for SegFormer** (repo `LICENSE` file). Per a search-result summary of the README, use is limited to non-commercial research or evaluation; the full text was not read | Reference only | 2026-10-13 |
| SegFormer HF checkpoint (e.g. `nvidia/segformer-b0-finetuned-ade-512-512`) | https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512 | UNKNOWN (HF page unreachable from this environment: proxy returned 403) | UNKNOWN from the official page. Third-party mirrors report HF tag `license: other` and derivation from the NVIDIA Source Code License. Not verified | Candidate segmentation baseline. Re-check the HF model card and the LICENSE file in that repo from the local machine | 2026-10-13 |
| Qwen3-VL-4B | https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct | UNKNOWN (HF page unreachable, 403) | Official code repo https://github.com/QwenLM/Qwen3-VL has an **Apache-2.0** `LICENSE` (verified). The HF model card itself was not read; third-party listings say Apache-2.0. Treat the checkpoint as Apache-2.0 *unverified* | Candidate root-cause text generator (also check the `-Instruct` vs base vs `-Thinking` variants) | 2026-10-13 |
| SAM 2 / SAM 2.1 | https://github.com/facebookresearch/sam2 | Python package `VERSION = "1.0"` (`setup.py` on `main`); checkpoint release names not checked | **Apache-2.0** for model checkpoints, demo code and training code (README "License" section). Optional `cc_torch` post-processing has its own `LICENSE_cctorch` (type not read); demo fonts are SIL OFL 1.1 | Candidate mask-refinement / annotation aid | 2026-10-13 |

## Ultralytics and submitting source code

[HYPOTHESIS] based on the AGPL-3.0 text and the README, not on the competition rules:

- AGPL-3.0 is a strong copyleft license. Code that imports or is derived from Ultralytics
  becomes a combined work that must be offered under AGPL-3.0 when it is conveyed to others,
  with complete corresponding source. Handing the organisers a source-code submission counts
  as conveying a copy.
- Consequences if the submission includes or depends on Ultralytics:
  1. The submitted code (not only the Ultralytics part) would have to be released under
     AGPL-3.0. That can conflict with competition rules that require you to keep exclusive
     rights, grant the organisers a different license, or forbid copyleft dependencies.
     Competition terms on this are UNKNOWN.
  2. Section 13 (network use) matters if the model is ever run as a hosted service.
  3. Whether trained weights produced with Ultralytics code are also AGPL-covered is
     not verified here. Treat them as possibly covered.
- The only documented way around AGPL is the paid Enterprise License from Ultralytics.
- Safer default for this repo: keep Ultralytics out of the submitted source tree and
  `requirements.txt` until the rules are read; prefer non-copyleft options (e.g. SegFormer
  via `transformers`, torchvision, SAM 2). Using Ultralytics only for private experiments
  and not submitting its code or weights would need confirmation from the organisers.
- SegFormer is not a clean alternative: its checkpoint license is unverified and may be
  non-commercial (see table). Resolve both before choosing a final stack.
