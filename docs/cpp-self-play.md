# C++ self-play backend on Windows

The batched self-play engine has an optional C++17 backend. It moves the game
state, win detection, PUCT selection, expansion, backup, and action sampling to
C++. Neural-network inference remains in Python/PyTorch and is called in
batches.

GCC is not required on Windows. Install **Visual Studio Build Tools 2022** with
the **Desktop development with C++** workload, then open an "x64 Native Tools
Command Prompt for VS 2022" (or initialize the same MSVC environment).

Build and install the project with:

```powershell
$env:SCORE4_REQUIRE_CPP = "1"
python -m pip install -e ".[train]"
```

`SCORE4_REQUIRE_CPP=1` makes installation fail if the native extension cannot
be compiled. Without it, installation succeeds and uses the original Python
implementation when no compiler is available.

At training startup, the selected implementation is printed as either:

```text
self_play_backend=cpp
```

or:

```text
self_play_backend=python
```

Only batched self-play (`--self-play-batch-size` greater than 1) uses the C++
backend. Single-game self-play remains the reference Python implementation.
