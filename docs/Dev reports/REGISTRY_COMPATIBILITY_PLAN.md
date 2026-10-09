# Registry compatibility work

Scope implemented locally on 2026-10-09: file containment, restricted data loading,
configuration parsing, installation-time Shared Runtime setup, user preferences,
dedicated-runtime migration, and Registry archive cleanup. Local verification is recorded below.
These verification records do not constitute Registry approval.

## 1. File access

- Restrict workflow and HTTP-supplied file paths to the configured ComfyUI
  input/output/temp or explicitly registered model/voice folders, as appropriate
  for the operation. Resolve links before checking containment.
- Keep the audio analyzer's existing widget and browser upload/drag-and-drop.
  Uploaded audio is already placed in input by ComfyUI's upload endpoint.
- Reject paths outside the permitted folders. Copying an arbitrary server-side
  path into input would still read that path and expose its contents.
- Audit the analyzer preview route and node execution together, plus the suite's
  other file-backed widgets and routes. Protect generated cache/output names
  from raw node IDs and other request values.
- Preserve explicitly configured external model and voice directories; do not
  turn their normal use into an unrestricted path exception.
- Restrict model download destinations before network access, writes, or replacement;
  reject traversal and links escaping the permitted data roots. Custom-node code
  folders are excluded from data roots even though ComfyUI registers them.

## 2. Checkpoint and dictionary loading

- Prefer restricted PyTorch loading for state dictionaries; keep existing model
  formats where safe loading supports them. Do not require conversion or copies
  of upstream weights as the default solution.
- The local format audit covered 52 PyTorch files and one plain-pickle language
  dictionary: 50 checkpoints opened directly with weights_only=True, two Dots
  latent-statistics files needed a scoped NumPy type allowlist, and the dictionary
  loaded with a data-only unpickler that refuses class/global lookups.
- The audit used the configured test interpreter with Torch 2.10.0+cu130. It did
  not establish full engine inference, every remote variant, TorchScript safety,
  or compatibility with old whole-model HuBERT files and all training resumes.
- Review RVC inference/separation/training loaders, DramaBox training, Dots
  integration, optional Russian stress dictionaries, and remaining unsafe
  helpers. No automatic fallback to unrestricted loading for an untrusted file.
- Keep dependency patches scoped to suite integration rather than changing
  unrelated installed packages or globally weakening deserialization.

## 3. Configuration expressions

- Replace the Qwen tokenizer's constant eval with its literal value.
- Replace bundled FunASR configuration eval calls with restricted data parsing.
  Preserve required list/tuple expressions, including existing list addition and
  repetition; literal_eval alone does not cover those expressions.
- Keep neural-network model.eval() calls. Add the required adjacent TTS Audio
  Suite patch comments for bundled-code changes.

## 4. Optional isolated environments

The maintainer rejected a manual setup command as the normal user experience.
Required behavior: users get automatic setup or a clear install action inside
ComfyUI, without editing files, running terminal commands, or learning about
Python environments. Keep existing workflow widgets and their order. Generation now checks prepared-runtime metadata and never installs packages.
The installer prepares or reuses Shared Runtime independently of its main-dependency fast path.

Implemented: prepare the single shared legacy environment automatically from the existing install.py hook during normal
Manager installation/update. Default Qwen3-TTS, Qwen ASR, Step Audio EditX, and
Higgs Audio 2 already select this environment. Users keep the existing experience
without terminal setup or additional support packages. Model weights still
download only when the corresponding engine is used.

The local shared environment measured 139.1 MB (132.7 MiB), including 6,429
files. It inherits the main environment's heavy packages rather than installing
another PyTorch copy. This measures the current machine, not download size or
a universal upper bound; missing/incompatible host dependencies can change it.

Keep an advanced opt-out for automatic shared-environment preparation. A local
installer configuration can apply before first installation; a ComfyUI setting
can control subsequent installation/update/repair runs. Skipping preparation
must neither delete an existing environment nor silently switch engines to an
incompatible main environment. A missing required environment should report
that support must be enabled and installation repaired through Manager.

### Implemented advanced controls

- Add a default-on "Install shared runtime automatically" setting under
  ComfyUI Settings > TTS Audio Suite > Runtime installation.
- Store the preference in one server-side JSON file read by the installer.
  The UI edits that same file and shows its actual location; a hand-edited
  preference must not be overwritten with stale browser defaults on page load.
  Prefer the suite's existing ComfyUI System User storage, outside package files
  replaced during updates. Resolve the configured user directory consistently
  in the running suite and the separate installer process.
  Default location: ComfyUI/user/__tts_audio_suite/runtime_settings.json.
  Document the equivalent location under a custom user directory so pre-install
  opt-out does not depend on opening a UI that is not installed yet.
- File schema:

  ```json
  {
    "install_shared_runtime": false
  }
  ```

- Only accept the defined boolean. Do not
  accept executable paths, package names, URLs, or shell commands from this file,
  a workflow, or a management request. Protect settings-management writes.
- Turning shared preparation off affects later installer runs. It does not
  remove the installed environment, stop its use by existing workflows, or
  select Main Environment instead. The file option supports pre-install opt-out;
  the UI becomes available after the suite is installed.
- Do not add automatic deletion on restart. Show whether the environment is
  installed, its folder location, and instructions to close ComfyUI before
  manually deleting runtimes/shared_legacy_t4. Explain that engines still set
  to Shared Runtime then need that environment restored or another prepared,
  compatible runtime selected. Model weights and voice files are separate.

### Runtime choices implemented locally

- Remove Dedicated Runtime from Qwen3-TTS and Step Audio EditX. Runtime widgets
  stay in their existing positions with Main Environment and Shared Runtime.
- Old Dedicated Runtime widget/API values and old dedicated profile names map
  to Shared Runtime. Frontend migration also updates saved workflow widgets.
- Remove the separate Qwen, Step, and VibeVoice legacy profile definitions; keep
  the shared profile unchanged so existing installations can be reused.
- Leave existing dedicated environment folders on disk. Do not delete user files
  as part of this change. Users can remove unused folders after closing ComfyUI.
- No dedicated installation controls or file settings are needed. Reuse valid
  shared runtime metadata on later updates, checking profile changes separately
  from the main dependency installer's fast path.

Manager can defer its installation jobs until restart, including on Windows.
That is still Manager-requested package installation. A workflow node recording
flags and our prestartup/import code calling pip on every boot is a different
boundary; timing alone does not make it acceptable. No new installer node is
needed for the recommended default. The implementation still requires Registry review; it does not assert approval under the latest policy.

Reuse existing compatible environments. An update must not silently delete a
working environment during generation. Model downloads remain separate from
installing executable Python dependencies. An HTTP route or workflow node that
invokes pip after a click would still be runtime installation; renaming pip or
hiding it in a helper is not a policy fix.

Evidence checked:

- [Official standards](https://docs.comfy.org/registry/standards) prohibit runtime
  package installation through subprocess calls.
- [Manager's install queue](https://github.com/Comfy-Org/ComfyUI-Manager/blob/main/glob/manager_server.py)
  accepts registered node IDs/versions and delegates to its package installer.
  Its [special-purpose files documentation](https://github.com/Comfy-Org/ComfyUI-Manager#custom-node-support)
  documents automatic install.py execution during installation.
- [Manager's deferred installer](https://github.com/Comfy-Org/ComfyUI-Manager/blob/main/glob/manager_core.py#L2070)
  schedules installation jobs for restart when appropriate, rather than requiring
  each node pack to create an independent startup installer.
- [SAM3 Registry versions](https://api.comfy.org/nodes/comfyui-sam3/versions?include_status_reason=true):
  0.1.21 is Active, with a manual SAFE decision by drltdata@comfy.org under
  policy-v0.1. Its [published archive](https://cdn.comfy.org/pznodes/comfyui-sam3/0.1.21/node.zip)
  contains install.py calling comfy_env.install(), plus isolated-environment
  dependency configuration. This is an accepted installation-time precedent,
  not a blanket exception for our current first-use bootstrap or newer policies.
- [comfy-env installation documentation](https://docs.comfy-forge.org/comfy-env/install/)
  describes installation-time creation rather than environment creation during
  node execution. Adoption of that library is not required for this plan.
- Registry review requests [226](https://github.com/Comfy-Org/registry-backend/issues/226),
  [261](https://github.com/Comfy-Org/registry-backend/issues/261), and
  [210](https://github.com/Comfy-Org/registry-backend/issues/210) are unanswered;
  they do not prove approval of optional runtime installers.

### User documentation

README.md now describes automatic installation, the UI/file opt-out, custom user
directories, repair through Manager, and manual cleanup. The details command shows
the actual preference file and runtime folder; no UI route or node runs pip.

## 5. Registry archive contents

- Added .comfyignore to exclude root tests/scripts, GitHub automation, the project
  index, and development reports from Registry archives. Repository contents
  remain available for development.
- Keep the two metadata YAML files in docs/Dev reports. The auxiliary model
  registry reads its YAML at runtime. Keep user guides, example workflows,
  frontend assets, and all engine/training code.
- The publisher's current Git filename handling skips quoted emoji paths.
  Configure core.quotepath=false in publishing so those guides/examples ship.

## Verification for the implementation

- Compare an archive with an independently collected, NUL-delimited Git source
  manifest (git ls-files -z), not the packer's own parsed filename list. Require
  every missing file to have an intentional exclusion and verify retained bytes.
  This catches silent omissions from quoting, decoding, and path parsing.
- Exercise uploads, allowed typed paths, rejected traversal/link escapes, and
  malformed node IDs through both node execution and HTTP preview routes.
- Check restricted loaders against the actual local formats, then exercise the
  affected engine inference and training paths using the configured test setup.
- Check existing prepared environments and missing/stale profiles. Confirm that
  generation cannot install packages or replace an environment.
- Recheck the published Registry status separately from successful upload;
  packaging cleanup by itself does not fix the runtime findings.

## Local checks for dedicated runtime removal

- The existing runtime unit test module passed all 27 cases, including legacy
  mode/profile migration and rejection of unsupported values.
- JavaScript migration checks covered 20 hook/value combinations and confirmed
  other widget positions/values and unrelated nodes remain unchanged.
- The restarted ComfyUI server exposes only Main Environment and Shared Runtime
  for Qwen3-TTS and Step Audio EditX, with the original input order. Its prompt
  validation accepts old dedicated values and rejects unknown modes. Requests
  deliberately omitted required downstream inputs, so no generation was queued.
- ComfyUI serves the new frontend extension. Full-fix checks are recorded below.

- Source rewrites must match AST call identity. Compare preserved attribute-call
  identities against the original source, independently of edit selection; builtin
  eval removal must not rename model.eval() or other unrelated methods.

## Completed local verification

- 59 targeted existing unit cases passed, covering runtime/profile migration,
  installer repairs, audio processing, voice compatibility, and dependency guidance.
- An empty temporary Shared Runtime was created by the real installer using the
  configured test Python. It installed Transformers 4.57.3, inherited host PyTorch,
  passed import/readiness/reuse checks, and left the host packages unchanged. The
  temporary environment was removed. Windows console logging uses ASCII output.
- Installer hook checks covered enabled preparation and file-based opt-out.
  Missing/stale runtime checks cannot run subprocess installers during generation;
  simulated failed installation restores the previous runtime.
- Live API checks passed for uploaded audio analysis, permitted paths, blocked
  traversal/external paths/node IDs, strict settings values, same-origin writes,
  preference persistence, and preservation of installed runtime files.
- The real browser showed the default-on setting and the details menu with the
  correct file/folder paths and manual cleanup instructions. Backend file edits
  override stale browser preferences. Frontend checks also covered rapid toggles
  and restoring the checkbox after a failed save.
- Live Step TTS followed by a giggle edit succeeded through the Shared Runtime,
  including migration from the old dedicated API value. The final run produced
  non-silent 24 kHz audio lasting 3.12 seconds.
- Restricted Dots latent statistics, the 3,060,182-entry Russian dictionary, NumPy
  data/DAC formats, and Demucs architecture metadata passed. A small real Demucs
  model restored its state through audio-separator's existing load_model entry
  point. Malicious pickle/NumPy/Torch payloads and model-constructor REDUCE calls
  were rejected. Dependency patches preserve global torch/importlib references.
- Python source compilation and independent AST comparison preserved every
  existing neural model.eval() call. Required config list arithmetic remains usable.
- The candidate archive includes intended uncommitted source without staging it:
  1,936 packaged files and 104 intentional exclusions. Runtime files, Unicode
  paths, user guides, and both metadata YAML files retain exact bytes. The auxiliary
  model registry reads its YAML successfully from the packaged layout.

Limits: Manager is absent from the test installation, so its real queue/restart
flow was not exercised; the install.py hook and fresh runtime installation were
tested directly. This does not establish every engine's inference/training resume,
support arbitrary whole-object checkpoints, or constitute Registry approval.
