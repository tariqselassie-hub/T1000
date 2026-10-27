# Kaggle Competitions & Notebook Runtime Troubleshooting Manual

This document is mirrored from the sovereign global skill [`kaggle`](file:///C:/Users/zoddj/.gemini/config/skills/kaggle/SKILL.md). It codifies the operational protocols, filesystem architectures, submission modalities, C-extension linking patterns, and diagnostic workflows for Kaggle competitions.

---

## 1. Filesystem Topology & Dynamic Path Resolution

```
/kaggle/
├── input/                 # Read-only mounted datasets and competition files
│   ├── <dataset-slug>/
│   └── competitions/<slug>/
├── working/               # Primary writable workspace (evaluated by notebook submission graders)
└── temp/                  # Ephemeral scratch directory
```

### Dynamic Path Resolver
```python
from pathlib import Path

possible_roots = [
    Path('/kaggle/input/competitions/<competition-slug>'),
    Path('/kaggle/input/<competition-slug>'),
]
comp_dir = next(r for r in possible_roots if r.exists())
```

---

## 2. Submission Modalities & Dual-Format Output Architecture

### The Two Evaluation Mechanisms
1. **Notebook Output Evaluation (Output Picker / Save Version):** The grader evaluates `/kaggle/working/` directly. All expected target files (`invsqrt.kmb`, `dqi.kmb`, `dfthcsos/`) must be placed unzipped directly in `/kaggle/working/`.
2. **Direct Archive Upload (Web UI Upload):** The grader extracts `submission.zip` and expects root-relative paths.

### Standard Dual-Format Pattern
Always output **both unzipped files and `submission.zip`** into `/kaggle/working/`:
```python
import shutil, zipfile
from pathlib import Path

w = Path('/kaggle/working')

# 1. Unzipped targets directly in /kaggle/working/
(w / 'target_file.kmb').write_bytes(data)
shutil.copy2(src / 'component.kmb', w / 'component.kmb')
if (w / 'subfolder').exists():
    shutil.rmtree(w / 'subfolder')
shutil.copytree(src / 'subfolder', w / 'subfolder')

# 2. Concurrently package submission.zip
with zipfile.ZipFile(w / 'submission.zip', 'w', compression=zipfile.ZIP_STORED) as zf:
    zf.write(w / 'target_file.kmb', 'target_file.kmb')
    zf.write(w / 'component.kmb', 'component.kmb')
    for f in (w / 'subfolder').glob('*.kmb'):
        zf.write(f, 'subfolder/' + f.name)
```

---

## 3. C-Extensions, Dynamic Libraries & Namespace Package Traps

### The Python PEP 420 Namespace Bug
If a folder `/kaggle/input/.../mod/` contains `mod.so`:
1. `import mod` imports the directory as a dummy namespace package (`<module 'mod' (namespace)>`).
2. The compiled `.so` is ignored, raising `AttributeError: module 'mod' has no attribute 'X'`.

### Bulletproof Fix
```python
import sys, importlib.util
from pathlib import Path

so_path = comp_dir / 'module_name' / 'module_name.so'
spec = importlib.util.spec_from_file_location("module_name", str(so_path))
mod = importlib.util.module_from_spec(spec)
sys.modules["module_name"] = mod
spec.loader.exec_module(mod)
```

---

## 4. Monaco Editor Clipboard Truncation Defense

1. Keep code lines strictly under 60 characters.
2. Use string concatenation (`'dfthcsos/' + f.name`) instead of nested f-strings in loops.
3. Split complex logic into 15–20 line micro-cells.

---

## 5. Diagnostic Matrix & Grader Playbook

| Error Signature | Root Cause | Immediate Action |
| :--- | :--- | :--- |
| `The submission must contain exactly one X, found 0.` | Grader expected unzipped files in `/kaggle/working/`, or archive was nested. | Execute Dual-Format Output Pattern: expose both unzipped files and `.zip` in `/kaggle/working/`. |
| `AttributeError: module 'X' has no attribute 'Y'` | C-extension imported as PEP 420 namespace package instead of `.so`. | Use `importlib.util.spec_from_file_location` and bind to `sys.modules`. |
| `x=<N>: phase error ... exceeds tolerance ...` | Candidate circuit violated tolerance on secret-salt boundary values. | Increase fractional output bits (e.g. `result_bits` $+1$), and verify locally against the specific $x$. |
| `FileNotFoundError: <file> not found` | Kernel was restarted or timed out, resetting `/kaggle/working/` to empty. | Re-run synthesis cell to regenerate working artifacts. |
