"""
ComfyUI custom nodes for MegaStyle-FLUX.

Install:
    ln -s /path/to/MegaStyle/comfyui /path/to/ComfyUI/custom_nodes/MegaStyle
Then restart ComfyUI.

On import, this package also:
  - copies `workflow_megastyle.json` into ComfyUI's
    `user/default/workflows/MegaStyle.json` so the graph shows up in the
    Workflows side panel;
  - symlinks `MegaStyle/ref_styles/*.{jpg,jpeg,png}` into `ComfyUI/input/`
    so the default LoadImage node resolves `00.jpg` out of the box.

Environment variables (all optional):
    MEGASTYLE_REPO_ROOT                override the auto-detected MegaStyle
                                       repo root used by nodes.py
    MEGASTYLE_COMFY_ROOT               override the auto-detected ComfyUI root
                                       (useful if auto-walk fails)
    MEGASTYLE_AUTO_INSTALL_WORKFLOW=0  disable workflow auto-install
    MEGASTYLE_AUTO_INSTALL_REFS=0      disable ref_styles auto-symlink
"""
import os
import shutil

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


def _looks_like_comfy_root(path: str) -> bool:
    return bool(
        path
        and os.path.isfile(os.path.join(path, "main.py"))
        and os.path.isdir(os.path.join(path, "custom_nodes"))
    )


def _walk_up_for_comfy(start_dir: str, max_up: int = 8):
    candidate = start_dir
    for _ in range(max_up):
        candidate = os.path.dirname(candidate)
        if not candidate or candidate == "/":
            break
        if _looks_like_comfy_root(candidate):
            return candidate
    return None


def _find_comfy_root():
    # 1. Explicit override.
    env = os.environ.get("MEGASTYLE_COMFY_ROOT")
    if env and _looks_like_comfy_root(env):
        return env
    # 2. Walk up from the loading path (keeps symlinks so we stay inside
    #    ComfyUI/custom_nodes/...).
    root = _walk_up_for_comfy(os.path.abspath(__file__))
    if root is not None:
        return root
    # 3. Fallback: walk up from the realpath.
    return _walk_up_for_comfy(os.path.realpath(__file__))


def _install_default_workflow():
    if os.environ.get("MEGASTYLE_AUTO_INSTALL_WORKFLOW", "1") == "0":
        return

    this_dir = os.path.dirname(os.path.realpath(__file__))
    workflow_src = os.path.join(this_dir, "workflow_megastyle.json")
    if not os.path.isfile(workflow_src):
        print(f"[MegaStyle] Workflow source not found: {workflow_src}")
        return

    comfy_root = _find_comfy_root()
    if comfy_root is None:
        print("[MegaStyle] Could not locate ComfyUI root. "
              "Set MEGASTYLE_COMFY_ROOT=/path/to/ComfyUI to enable "
              "auto-install of the workflow.")
        return

    dst_dir = os.path.join(comfy_root, "user", "default", "workflows")
    try:
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, "MegaStyle.json")
        # Only copy if missing or outdated, so the user's edits aren't clobbered
        # once they saved their own version.
        if (not os.path.isfile(dst)) or \
           (os.path.getmtime(workflow_src) > os.path.getmtime(dst)):
            shutil.copy2(workflow_src, dst)
            print(f"[MegaStyle] Installed default workflow -> {dst}")
        else:
            print(f"[MegaStyle] Workflow already up-to-date: {dst}")
    except Exception as e:  # noqa: BLE001
        print(f"[MegaStyle] Could not install default workflow: {e}")


def _install_reference_styles():
    """Symlink MegaStyle/ref_styles/*.jpg into ComfyUI/input/ so the default
    workflow's LoadImage can find '00.jpg' on first launch.

    We create *symlinks* rather than copies to avoid duplicating ~50 images.
    Existing files with the same name are left untouched (user's own uploads
    win). Disable with env: MEGASTYLE_AUTO_INSTALL_REFS=0.
    """
    if os.environ.get("MEGASTYLE_AUTO_INSTALL_REFS", "1") == "0":
        return

    this_dir = os.path.dirname(os.path.realpath(__file__))
    # this_dir = .../MegaStyle/comfyui ; parent is repo root.
    ref_dir = os.path.join(os.path.dirname(this_dir), "ref_styles")
    if not os.path.isdir(ref_dir):
        return

    comfy_root = _find_comfy_root()
    if comfy_root is None:
        return

    input_dir = os.path.join(comfy_root, "input")
    try:
        os.makedirs(input_dir, exist_ok=True)
    except Exception as e:  # noqa: BLE001
        print(f"[MegaStyle] Could not create {input_dir}: {e}")
        return

    installed = 0
    for name in os.listdir(ref_dir):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        src = os.path.join(ref_dir, name)
        dst = os.path.join(input_dir, name)
        if os.path.exists(dst) or os.path.islink(dst):
            continue
        try:
            os.symlink(src, dst)
            installed += 1
        except OSError:
            # Fallback to copy if the FS doesn't support symlinks.
            try:
                shutil.copy2(src, dst)
                installed += 1
            except Exception as e:  # noqa: BLE001
                print(f"[MegaStyle] Could not install {name} -> {dst}: {e}")
    if installed:
        print(f"[MegaStyle] Installed {installed} reference image(s) -> {input_dir}")


_install_default_workflow()
_install_reference_styles()
