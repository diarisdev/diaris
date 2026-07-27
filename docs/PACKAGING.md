# Packaging Diaris into a Windows `.exe`

This builds a standalone, double-clickable app with **PyInstaller (onedir)**, wrapped
in an optional **Inno Setup** installer. It assumes the Silero-ONNX work is done (no
runtime `torch.hub` download — the packaging blocker is already cleared).

> **Golden rule:** the app is a *thin GUI over native ML kernels*. The models
> (whisper, pyannote, NLLB) and CUDA are the weight, not your code. So models live
> **next to the exe**, never inside the bundle.

---

## 1. The CPU-vs-GPU insight

The CPU and GPU builds differ in **exactly one thing: the torch wheels.** Everything
else — pyannote, PySide6, ctranslate2, faster-whisper, your `src/`, the spec — is
identical. So you build **one spec from two clean virtualenvs**:

| Variant | Build venv has | Result size |
|---|---|---|
| CPU | CPU torch (`pip install torch --index-url .../cpu`) | ~1.5–2.5 GB |
| GPU | CUDA torch (your current dev wheel, but **pin a stable release** for distribution) | ~5–8 GB |

Do **not** fork the spec. Fork the environment torch is installed into.

---

## 2. One-time setup (per build venv)

```powershell
# In the venv for the variant you're building:
pip install pyinstaller
python scripts/download_models.py        # models/ must be populated (incl. silero_vad.onnx)
```

---

## 3. Build

```powershell
# FIRST attempt — always use --debug (console build).
# If the GUI crashes on a missing import, you'll SEE the traceback.
python scripts/build_app.py --debug

# Once it launches cleanly, build the real (windowed) version:
python scripts/build_app.py
```

Output: `dist/Diaris/` — `Diaris.exe` + an `_internal/` folder.

The helper also drops `.env` (or `.env.example` as a template) next to the exe and
creates an empty `models/`. Add `--with-models` to copy `models/` in for a fully
offline build, or `--clean` to wipe `build/`+`dist/` first.

---

## 4. The iteration you *will* hit: hidden imports

pyannote / lightning / speechbrain do **dynamic imports** PyInstaller can't see from
the import graph. The spec already `collect_all`s the usual suspects, but a build may
still fail at runtime with `ModuleNotFoundError: No module named 'X'`.

**The loop:**
1. Run the `--debug` build, reproduce the crash, read the missing module name.
2. Add it to `hiddenimports` in [`Diaris.spec`](../Diaris.spec) (or add
   the package to the `_COLLECT_ALL` list).
3. Rebuild. Repeat until it launches.

This is normal for this dependency tree — budget for a few rounds. Common additions:
`sklearn.utils._*`, `scipy.*` C extensions, `speechbrain.*`, `asteroid_filterbanks.*`.

### The GPU gotcha: `cublas64_12.dll is not found`

If the app launches but produces no output and you see
`Warm-up inference failed: Library cublas64_12.dll is not found`, the CUDA runtime
DLLs aren't on the frozen app's search path. `ctranslate2`/`torch` load them **by name
at runtime**, so PyInstaller's dependency scan misses them.

This is handled: the spec now globs `site-packages/nvidia/*/bin/*.dll` and `torch/lib/*.dll`
into the bundle root, and `configure_cuda_dll_paths()` adds the bundle dir to the DLL
search path when frozen. If you still hit it, confirm the build venv actually has the
`nvidia-cublas-cu12` wheel installed (`pip show nvidia-cublas-cu12`) and check the
`[spec] CUDA/torch DLL toplandı: N` line during the build is non-zero.

---

## 5. Model delivery (recommended hybrid)

Your models split cleanly by licence:

| Model | HF token? | Size | Ship how |
|---|---|---|---|
| pyannote segmentation + embeddings | **yes** | ~110 MB | **bundle** (tiny; removes runtime token need) |
| faster-whisper | no | 0.5–1.5 GB | download on first run |
| NLLB-200 | no | ~600 MB | download on first run |
| silero_vad.onnx | no | ~2 MB | bundle |

Bundling the tiny token-gated pyannote models means **the end user never needs an HF
token**. The large token-free models download on first launch. To wire "download on
first run," have the app call `scripts/download_models.py` logic when `models/` is
missing assets (a natural follow-up; not required for a working build).

---

## 6. Installer (Inno Setup)

Wrap `dist/Diaris/` in an installer for Start-menu shortcut + install location.
Skeleton (`installer.iss`):

```ini
[Setup]
AppName=Diaris
AppVersion=1.0
DefaultDirName={autopf}\Diaris
DefaultGroupName=Diaris
OutputBaseFilename=Diaris-Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64

[Files]
Source: "dist\Diaris\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Diaris"; Filename: "{app}\Diaris.exe"
Name: "{autodesktop}\Diaris"; Filename: "{app}\Diaris.exe"

[Run]
Filename: "{app}\Diaris.exe"; Description: "Launch Diaris"; Flags: nowait postinstall skipifsilent
```

Build it with the Inno Setup Compiler (GUI or `ISCC.exe installer.iss`).

---

## 7. Do NOT

- **onefile** — unpacks the multi-GB bundle to temp on *every* launch. onedir starts instantly.
- **UPX** — antivirus false-positives and it can corrupt torch/ctranslate2 native DLLs.
- **bundle models into the exe** — they belong next to it.
- **ship your `.env`** — it contains your real `HF_TOKEN`. The helper warns about this.
- **ship the dev/nightly torch** — pin a stable release for a distributable GPU build.

---

## 8. What was already done in the codebase for this

- `config._resolve_project_root()` — when frozen, `models/`, `.env`, `output/` resolve
  to the **exe's directory**, not inside the bundle. (Tested both modes.)
- `configure_cuda_dll_paths()` — skips `site.getsitepackages()` when frozen (PyInstaller
  handles CUDA DLLs), avoiding an `AttributeError`.
- Silero loads from `models/silero_vad.onnx` — no network at runtime.
- [`Diaris.spec`](../Diaris.spec) — committed, onedir, console toggled by
  `AP_BUILD_CONSOLE`.
