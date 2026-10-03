# Engineering Verification and Honesty (Krystal-Stack)

## Behavioral Rule
Apply to every change, document and piece of public copy in this repository.

1. **Measured claims only.** Docs, READMEs and community or marketing copy may state only numbers that were measured on a named machine, with the date and the number of runs. Label everything else "target" or "unverified". This host (i5-1135G7, Iris Xe) has no NPU or TPU. OpenVINO, iGPU inference and Vulkan GPU dispatch are unverified unless a measurement exists in the repo.
2. **Installs.** Install only into an isolated venv (`.venv-keras`), and only after the user agrees. Never run a global `pip install`. Never try to bypass Windows Application Control, SmartScreen or antivirus blocks: report the block and use the fallback.
3. **Stdlib default.** Core packages (`krystal_kernel`, `krystal_bot`) must run on bare Python. Optional heavy dependencies (Keras, OpenVINO, numpy) are lazy, guarded, and run out-of-process.
4. **Learned components need a gate.** No synthetic training data. Accept a model only if it beats a stated baseline on a held-out split with a minimum number of positives, and report when it is rejected.
5. **Localhost services.** Verify with a Python script that sets timeouts, never with curl loops. Before testing, check that exactly one process owns the port (`Get-NetTCPConnection`) and kill stale servers, including the WindowsApps python stub and its child. New API surfaces must not add wildcard CORS and must cap request bodies before reading them.
6. **Secrets and licences.** API keys come from environment variable names, never from files, logs or errors. When adapting open-source designs (for example OpenClaw, MIT, or Mesa), study and re-implement. Do not paste code. Record the licence and the idea borrowed.
