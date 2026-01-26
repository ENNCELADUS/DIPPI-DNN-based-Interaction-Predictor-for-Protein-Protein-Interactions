# Dead Code Analysis

- Generated: `2026-01-26T14:16:28+00:00`
- Repo root: `/Users/richardwang/Documents/XPI`

## Targets Analyzed

- `/Users/richardwang/Documents/XPI/src`

## Tool Runs

- `ruff` (src): exit=1; cmd=`/opt/homebrew/Caskroom/miniconda/base/envs/esm/bin/python -m ruff check /Users/richardwang/Documents/XPI/src --select F401,F841`
- `vulture` (src): SKIPPED; cmd=``
- `deptry` (.): SKIPPED; cmd=``
- `pytest` (.): exit=2; cmd=`/opt/homebrew/Caskroom/miniconda/base/bin/conda run -n esm python -m pytest`

## Severity (Heuristic Buckets)

Treat this section as a starting point; validate each item before deletion.

### SAFE

- Remove unused imports in `src/embed/embed.py`:
  - `typing.List` is unused (ruff `F401`).
  - `typing.Tuple` is unused (ruff `F401`).

### CAUTION

- (none detected by heuristics)

### DANGER

- (none detected by heuristics)


## Proposed Safe Deletions (Fill In)

- `src/embed/embed.py`: remove unused `typing.List` and `typing.Tuple` imports (ruff `F401`; no runtime behavior change).
  - Status: **NOT EXECUTED** (baseline `pytest` fails during collection; no changes applied without a passing test gate).

## Raw Outputs

### ruff (src)

- cmd: `/opt/homebrew/Caskroom/miniconda/base/envs/esm/bin/python -m ruff check /Users/richardwang/Documents/XPI/src --select F401,F841`
- returncode: `1`
- skipped_reason: `None`

**stdout**

```text
F401 [*] `typing.List` imported but unused
  --> src/embed/embed.py:23:20
   |
21 | import torch
22 | from pathlib import Path
23 | from typing import List, Tuple, Any
   |                    ^^^^
24 |
25 | try:
   |
help: Remove unused import

F401 [*] `typing.Tuple` imported but unused
  --> src/embed/embed.py:23:26
   |
21 | import torch
22 | from pathlib import Path
23 | from typing import List, Tuple, Any
   |                          ^^^^^
24 |
25 | try:
   |
help: Remove unused import

Found 2 errors.
[*] 2 fixable with the `--fix` option.
```

**stderr**

```text

```


### vulture (src)

- cmd: ``
- returncode: `None`
- skipped_reason: `module-not-installed`

**stdout**

```text

```

**stderr**

```text

```


### deptry (.)

- cmd: ``
- returncode: `None`
- skipped_reason: `module-not-installed`

**stdout**

```text

```

**stderr**

```text

```


### pytest (.)

- cmd: `/opt/homebrew/Caskroom/miniconda/base/bin/conda run -n esm python -m pytest`
- returncode: `2`
- skipped_reason: `None`

**stdout**

```text

==================================== ERRORS ====================================
_______ ERROR collecting tests/integration/test_data_loading_sampling.py _______
ImportError while importing test module '/Users/richardwang/Documents/XPI/tests/integration/test_data_loading_sampling.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/homebrew/Caskroom/miniconda/base/envs/esm/lib/python3.10/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/integration/test_data_loading_sampling.py:12: in <module>
    from src.utils.data_io import build_dataloaders
E   ImportError: cannot import name 'build_dataloaders' from 'src.utils.data_io' (/Users/richardwang/Documents/XPI/src/utils/data_io.py)
______________ ERROR collecting tests/unit/embed/cli/test_main.py ______________
ImportError while importing test module '/Users/richardwang/Documents/XPI/tests/unit/embed/cli/test_main.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/homebrew/Caskroom/miniconda/base/envs/esm/lib/python3.10/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/unit/embed/cli/test_main.py:12: in <module>
    from src.embed.core.types import EmbeddingConfig, EmbeddingResult
E   ModuleNotFoundError: No module named 'src.embed.core'
_____________ ERROR collecting tests/unit/embed/core/test_types.py _____________
ImportError while importing test module '/Users/richardwang/Documents/XPI/tests/unit/embed/core/test_types.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/homebrew/Caskroom/miniconda/base/envs/esm/lib/python3.10/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/unit/embed/core/test_types.py:8: in <module>
    from src.embed.core.types import (
E   ModuleNotFoundError: No module named 'src.embed.core'
__________ ERROR collecting tests/unit/embed/core/test_validation.py ___________
ImportError while importing test module '/Users/richardwang/Documents/XPI/tests/unit/embed/core/test_validation.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/homebrew/Caskroom/miniconda/base/envs/esm/lib/python3.10/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/unit/embed/core/test_validation.py:9: in <module>
    from src.embed.core.validation import (
E   ModuleNotFoundError: No module named 'src.embed.core'
___________ ERROR collecting tests/unit/embed/io/test_filesystem.py ____________
ImportError while importing test module '/Users/richardwang/Documents/XPI/tests/unit/embed/io/test_filesystem.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/homebrew/Caskroom/miniconda/base/envs/esm/lib/python3.10/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/unit/embed/io/test_filesystem.py:11: in <module>
    from src.embed.core.types import EmbeddingResult
E   ModuleNotFoundError: No module named 'src.embed.core'
____________ ERROR collecting tests/unit/embed/io/test_structure.py ____________
ImportError while importing test module '/Users/richardwang/Documents/XPI/tests/unit/embed/io/test_structure.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/homebrew/Caskroom/miniconda/base/envs/esm/lib/python3.10/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/unit/embed/io/test_structure.py:11: in <module>
    from src.embed.io.structure import (
E   ModuleNotFoundError: No module named 'src.embed.io'
________ ERROR collecting tests/unit/embed/pipelines/test_multimodal.py ________
ImportError while importing test module '/Users/richardwang/Documents/XPI/tests/unit/embed/pipelines/test_multimodal.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/homebrew/Caskroom/miniconda/base/envs/esm/lib/python3.10/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/unit/embed/pipelines/test_multimodal.py:10: in <module>
    from src.embed.core.types import (
E   ModuleNotFoundError: No module named 'src.embed.core'
_________ ERROR collecting tests/unit/embed/pipelines/test_sequence.py _________
ImportError while importing test module '/Users/richardwang/Documents/XPI/tests/unit/embed/pipelines/test_sequence.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/homebrew/Caskroom/miniconda/base/envs/esm/lib/python3.10/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/unit/embed/pipelines/test_sequence.py:12: in <module>
    from src.embed.core.types import EmbeddingConfig
E   ModuleNotFoundError: No module named 'src.embed.core'
_______________ ERROR collecting tests/unit/embed/test_config.py _______________
ImportError while importing test module '/Users/richardwang/Documents/XPI/tests/unit/embed/test_config.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/homebrew/Caskroom/miniconda/base/envs/esm/lib/python3.10/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/unit/embed/test_config.py:5: in <module>
    from src.embed.config import PathSettings, default_config
E   ModuleNotFoundError: No module named 'src.embed.config'
___________________ ERROR collecting tests/unit/test_run.py ____________________
ImportError while importing test module '/Users/richardwang/Documents/XPI/tests/unit/test_run.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/homebrew/Caskroom/miniconda/base/envs/esm/lib/python3.10/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/unit/test_run.py:11: in <module>
    from src.run import (
E   ImportError: cannot import name 'bootstrap_runtime' from 'src.run' (/Users/richardwang/Documents/XPI/src/run.py)
_____________ ERROR collecting tests/unit/utils/test_checkpoint.py _____________
ImportError while importing test module '/Users/richardwang/Documents/XPI/tests/unit/utils/test_checkpoint.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/homebrew/Caskroom/miniconda/base/envs/esm/lib/python3.10/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/unit/utils/test_checkpoint.py:10: in <module>
    from src.utils.checkpoint import (
E   ImportError: cannot import name 'infer_resume_global_step' from 'src.utils.checkpoint' (/Users/richardwang/Documents/XPI/src/utils/checkpoint.py)
______________ ERROR collecting tests/unit/utils/test_data_io.py _______________
ImportError while importing test module '/Users/richardwang/Documents/XPI/tests/unit/utils/test_data_io.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/homebrew/Caskroom/miniconda/base/envs/esm/lib/python3.10/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/unit/utils/test_data_io.py:14: in <module>
    from src.utils.data_io import (
E   ImportError: cannot import name '_clean_tokens' from 'src.utils.data_io' (/Users/richardwang/Documents/XPI/src/utils/data_io.py)
=========================== short test summary info ============================
ERROR tests/integration/test_data_loading_sampling.py
ERROR tests/unit/embed/cli/test_main.py
ERROR tests/unit/embed/core/test_types.py
ERROR tests/unit/embed/core/test_validation.py
ERROR tests/unit/embed/io/test_filesystem.py
ERROR tests/unit/embed/io/test_structure.py
ERROR tests/unit/embed/pipelines/test_multimodal.py
ERROR tests/unit/embed/pipelines/test_sequence.py
ERROR tests/unit/embed/test_config.py
ERROR tests/unit/test_run.py
ERROR tests/unit/utils/test_checkpoint.py
ERROR tests/unit/utils/test_data_io.py
!!!!!!!!!!!!!!!!!!! Interrupted: 12 errors during collection !!!!!!!!!!!!!!!!!!!
12 errors in 11.80s

ERROR conda.cli.main_run:execute(127): `conda run python -m pytest` failed. (See above for error)
```

**stderr**

```text

```
