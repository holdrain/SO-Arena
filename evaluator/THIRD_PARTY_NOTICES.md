# Third-party provenance and distribution status

This candidate bundles byte-identical ToolSandbox and BFCL simulator/checker sources, reviewed JEV adapters and selected fixtures. runtime/original_sha256.json records exact original file hashes. Path relocation is in-memory and does not rewrite or disable source integrity checks. Original source copyright/license notices remain in the bundled tree.

ToolSandbox upstream: https://github.com/apple/ToolSandbox ; pinned source revision c8571d7854316d2e1c5f288e59fe1e34e53f6dd1. See runtime/original/curation/continuation_20261025/tools_label_audit/source_cache/ToolSandbox/LICENSE.

BFCL source/data provenance is retained in the BFCL runtime manifests. This package does not grant an independent license to upstream datasets or collaborator code.

Web sources include Mind2Web and BreakingWeb. The source guide states that Mind2Web unzipped test data should not be redistributed online and that BreakingWeb ships no license. Therefore public distribution of this complete package is pending source-specific redistribution clearance or an approved fetch/rebuild distribution mechanism.

Existing static source attributions remain in the input/answer metadata . Redistribution review of every source and an agreed license/citation for JEV-owned code are still required before a public release. Do not apply a blanket license to this mixed-source package.

Wheel package license metadata stays inside the original wheels under runtime/wheelhouse. The offline environment is Linux x86_64 CPU-only; model hardware/runtime packages are supplied by the model service owner.
