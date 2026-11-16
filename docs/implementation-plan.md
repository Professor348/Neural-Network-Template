# Neural library implementation plan

Goal: replace the head tracking application with a readable C++17 library.
The user has preserved the original on GitHub and authorized proceeding without backup verification.

- Implement scalar double precision dense networks, linear/tanh/sigmoid/ReLU,
  mean squared error, and minibatch SGD, with explicit initialization/shuffle seeds.
- Keep architecture, device preferences and training options in public value types;
  numerical code, CPU execution and device policy have separate internal files.
- Test analytic gradients against central differences, an independently calculated
  SGD update, XOR learning, dimension/finiteness validation and reproducibility.
- Test deterministic device policy using injected inventories, never simulated execution.
- Add a separately linked XOR example, installable CMake target and consumer check.
- Run Debug/Release and sanitizer checks; record actual accelerator limitations.

Review focus: invalid enum values, AUTO with index, nonfinite inputs and updates,
empty/ragged data, callback exceptions and persistent device overrides.

Initial accelerator scope (superseded for GPU by `gpu-plan.md`): no CUDA toolkit
or OpenVINO development runtime is installed.
Only CPU execution is implemented in this template; GPU/NPU requests fail explicitly.
Do not retain the old unverified accelerator code or advertise detection as execution.
