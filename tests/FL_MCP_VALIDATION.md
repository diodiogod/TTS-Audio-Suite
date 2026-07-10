# FL-MCP Live ComfyUI Validation

This runbook defines an optional live acceptance test for new engine implementations using [ComfyUI_FL-MCP](https://github.com/filliptm/ComfyUI_FL-MCP).

FL-MCP gives an MCP-capable LLM access to ComfyUI workflow state, queue execution, history, errors, output files, and browser screenshots. It does not replace pytest, canonical Windows validation, or human evaluation of audio quality.

## Efficient Tool Use

Use terminal tools for code inspection, edits, pytest, process management, startup logs, and routine log monitoring. Use FL-MCP at live-validation checkpoints: capability audit, relevant widget configuration, workflow queueing, prompt-specific execution verification, and output inspection.

Do not use FL-MCP as a replacement for efficient terminal work. Avoid fetching full workflow JSON, repeated node definitions or workflow overviews, screenshots, broad execution history, and polling calls unless they are needed to construct the workflow or diagnose a failure. Prefer one targeted call with the known node or prompt ID. A representative FL-MCP acceptance pass after automated tests is normally sufficient; repeat it only after changes that can affect live behavior.

## Preconditions

- Use the canonical Windows ComfyUI installation documented in the repository instructions.
- ComfyUI starts with TTS Audio Suite enabled and no suite import errors.
- FL-MCP is installed according to its upstream documentation.
- The FL-MCP server is connected to the MCP client.
- For browser-only operations, ComfyUI is open and the FL-MCP browser bridge reports connected.
- The MCP client is bound to the same browser session. Check the bridge session endpoint or capability audit and require both `has_frontend: true` and `has_mcp: true` for one session. A frontend-only session provides no canvas access to the LLM.
- Required model files and reference audio are available.
- The user has authorized any installation or configuration changes. Do not weaken FL-MCP safety gates just to complete a test.

If any precondition is missing, record it as a blocked or skipped check. Do not claim FL-MCP validation.

## Implementation Restart Loop

ComfyUI imports custom-node Python code at startup. Editing an engine does not update an already-running ComfyUI process. During implementation, use this loop:

1. Edit the engine implementation.
2. Run the relevant automated tests.
3. Identify the Windows process listening on port `8188` and inspect its command line. Stop it only if it is the canonical ComfyUI `main.py` process.
4. Resolve the ComfyUI root and Python executable from the user's local configuration, then launch ComfyUI. For example:

   ```powershell
   $comfyRoot = '<COMFYUI_ROOT>'
   $pythonExe = '<COMFYUI_PYTHON>'
   Set-Location $comfyRoot
   & $pythonExe 'main.py'
   ```

   On Windows, prefer a visible Windows Terminal tab with live output (PowerShell console only as fallback) so clicking the log cannot pause ComfyUI. When the agent needs searchable logs, also use `Start-Transcript` to a temporary file under `ComfyUI/temp` and inspect only relevant tails. Preserve required canonical launch arguments and never track machine-specific paths.
5. Poll `http://127.0.0.1:8188/system_stats` until it responds or startup fails. Inspect the startup log for custom-node import errors.
6. Refresh the already-open ComfyUI browser tab. A hard refresh may be required for changed JavaScript.
7. Confirm the FL-MCP backend is healthy and the browser bridge is connected again. Canvas tools are not available until the browser reconnects.
8. Run the live workflow checks below.
9. After a fix, repeat from step 2. Never validate new Python code against a process that started before the edit.

Do not stop every `python.exe` process. Other engines and applications may use Python. Resolve the owner of port `8188`, verify its executable/command line, and stop only the intended ComfyUI process.

If FL-MCP's embedded launcher starts and then shuts down ComfyUI on Windows, run `backend/server.py` in its own visible Windows Terminal window, launch ComfyUI in a separate window, then refresh the browser and reconnect the FL-MCP panel before auditing capabilities.

For manual Windows mode, keep two visible service terminals: one running ComfyUI and one running FL-MCP `backend/server.py`; the MCP client (for example Codex) is a separate process that must be restarted or rebound with the browser session ID so one session reports both `has_frontend: true` and `has_mcp: true`.

FL-MCP process-control permission is not required for this loop. An authorized coding agent can manage the canonical process through PowerShell. Keep `FL_MCP_ENABLE_COMFY_PROCESS_CONTROL=false` unless the user explicitly chooses FL-MCP-managed process control.

### MCP client session binding

REST access does not imply browser/canvas access. Before starting the implementation session:

1. Open ComfyUI and obtain the persistent FL-MCP browser session ID from its panel or the backend session endpoint.
2. Configure the MCP client process with:

   ```text
   FL_MCP_MODE=subprocess
   FL_MCP_SESSION_ID=<browser-session-id>
   FL_MCP_WS_URL=ws://127.0.0.1:8000/ws
   COMFYUI_SERVER_URL=http://127.0.0.1:8188
   ```

3. Restart the MCP client session so it launches the server with the new environment.
4. Run the capability audit and verify the same session reports both frontend and MCP connections.

The browser stores its session ID in local storage, so normal ComfyUI restarts and browser refreshes should retain it. If the browser session is reset or a different browser/profile is used, update the MCP client configuration and restart that client again.

## Validation Layers

Run all applicable layers:

1. Existing unit and integration tests through `tests/run_tests.py`.
2. Live ComfyUI registration and workflow validation through FL-MCP, or the manual fallback below.
3. Human listening evaluation.

FL-MCP is strongest at layer 2. A screenshot alone is not a passing test.

## Standard Smoke Sequence

Tool names can change between FL-MCP versions. Discover the connected server's available tools and use the current equivalents rather than assuming an exact name.

### 1. Establish a baseline

- Record the ComfyUI URL, TTS Audio Suite branch/commit, FL-MCP version, target engine, model variant, and runtime mode.
- Record the ComfyUI process start time and confirm it started after the implementation changes being tested.
- Check ComfyUI health and queue state. Read recent execution history only when needed to separate an existing failure from the new run.
- Preserve pre-existing errors separately so they are not attributed to the new engine.

### 2. Verify registration and UI state

- Inspect the target node definition and confirm the new engine node is registered. Do not repeatedly fetch definitions that have already been verified after the current restart.
- Confirm every relevant unified node is registered: TTS Text and SRT for TTS engines, plus VC, ASR, or special nodes when scoped.
- Open or construct a minimal workflow.
- Verify important widget values and connections with targeted node/workflow calls. Fetch full workflow JSON only when construction, migration, or debugging requires it.
- Capture a canvas screenshot only when UI layout, dynamic widgets, or visual state is part of the change or useful failure evidence.

### 3. Run the smallest real generation

- Use short deterministic text and the smallest suitable model/configuration.
- Queue the workflow.
- Wait for terminal execution state; do not treat prompt submission as success.
- Inspect execution history/details for the prompt.
- On failure, capture the node ID/type, exception, traceback or message, and relevant configuration.
- On success, verify that an expected audio output is listed and exists in ComfyUI output or temp storage.

### 4. Exercise required paths

For every TTS engine:

- Run TTS Text.
- Run SRT with at least two subtitle entries.
- Run one supported character, narrator, or reference-audio path as applicable.
- Run pause tags.
- Run one real engine-specific parameter switch when supported.
- Repeat an identical request to exercise generated-audio caching when applicable.
- Clear VRAM/unload, then generate again.
- Start a sufficiently long generation and test interrupt/cancel.

Also test VC, ASR, and special nodes when they are part of the agreed engine scope.

### 5. Inspect outputs

- Confirm output audio metadata is plausible, including sample rate, channel/batch shape where exposed, and non-zero duration.
- Confirm SRT output timing and artifact count match the workflow contract.
- Verify each run through its prompt-specific execution result rather than repeatedly fetching broad history or relying on the canvas alone.
- Ask the user to listen to representative outputs. The LLM must not claim audio quality, pronunciation, similarity, or naturalness based only on metadata.

## Required Evidence

Return a compact validation report containing:

- Environment and exact branch/commit.
- Saved workflow path, or the relevant node configuration and connections. Export full workflow JSON only when needed to reproduce the result.
- Screenshot when visual UI behavior was tested; otherwise mark it `not required`.
- Prompt/execution ID and terminal status for each run.
- Relevant execution error details for failed runs.
- Output artifact paths for successful runs.
- A pass/fail/blocked table for each required path.
- Automated test results.
- Explicitly untested items and reasons.
- Human audio-quality verdict, or `not evaluated`.

Do not commit generated audio, screenshots, logs, downloaded models, or local FL-MCP configuration unless the user explicitly requests tracked fixtures.

## Manual Fallback

If FL-MCP is unavailable:

1. Start the canonical Windows ComfyUI instance and inspect startup logs for node import errors.
2. Build or load the same minimal workflows in the browser.
3. Queue each required path and wait for completion.
4. Copy full errors from ComfyUI/server logs.
5. Export workflow JSON and record output artifact paths.
6. Capture screenshots manually.
7. Produce the same validation report and mark the driver as `manual`.

Manual execution is valid evidence. Do not install FL-MCP during an engine task unless installation was explicitly authorized.

## Stop Conditions

Stop the live test and report the blocker when:

- Installing dependencies would modify the user's environment without authorization.
- A required model download, license acceptance, credential, or reference file is unavailable.
- Repeated execution risks corrupting data or exhausting disk space.
- The connected MCP tools cannot identify terminal execution state or retrieve actionable errors.

Fix implementation defects that are within the engine task. Do not hide failures by changing safety settings, deleting unrelated queues/files, or weakening assertions.
