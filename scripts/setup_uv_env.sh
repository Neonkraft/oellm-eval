#!/usr/bin/env bash
# Build the evaluation runtime as uv environments, for clusters that do not use
# a container. This is the uv equivalent of containers/*.def.
#
# Usage:
#     scripts/setup_uv_env.sh <env_root>
#
# Result:
#     <env_root>/venv          lm-eval runtime. Give this to --venv_path.
#     <env_root>/judgearena    JudgeArena + vLLM runtime (own torch build).
#     <env_root>/uv-tools      lighteval as an isolated uv tool.
#     <env_root>/evalchemy*    only with WITH_EVALCHEMY=1 (own --venv_path).
#
# `judgearena` and `lighteval` are linked into <env_root>/venv/bin, because the
# generated sbatch script activates one venv and calls both by name.
#
# Environment variables:
#     JUDGEARENA_GIT_REF       JudgeArena commit (default: the ref in the .def files)
#     JUDGEARENA_TORCH_BACKEND cu130 (default, needs NVIDIA driver >= 580) or e.g. cu128
#     WITH_EVALCHEMY           1 to also build the evalchemy venv (default: 0)
#     PYTHON_VERSION           default 3.12
set -euo pipefail

ENV_ROOT="${1:?usage: $0 <env_root>}"
mkdir -p "$ENV_ROOT"
ENV_ROOT="$(cd "$ENV_ROOT" && pwd)"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

JUDGEARENA_GIT_REF="${JUDGEARENA_GIT_REF:-45e3b80705f54463f7b67bdeecd8245413eb7457}"
JUDGEARENA_TORCH_BACKEND="${JUDGEARENA_TORCH_BACKEND:-cu130}"
WITH_EVALCHEMY="${WITH_EVALCHEMY:-0}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VLLM_VERSION="0.19.1"
EVALCHEMY_GIT_REF="54ac97648230c4c3a22c3a2b93068b5a4e573f8d"

VENV="$ENV_ROOT/venv"
JA_VENV="$ENV_ROOT/judgearena"

# --torch-backend needs uv >= 0.6.9.
command -v uv >/dev/null || {
    echo "uv not found. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
}
uv pip install --help | grep -q -- "--torch-backend" || {
    echo "uv is too old ($(uv --version)). Run: uv self update" >&2
    exit 1
}

# Wheels are large; a copy is safer than hardlinks across file systems.
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

echo "==> [1/4] lm-eval venv: $VENV"
uv venv --allow-existing --python "$PYTHON_VERSION" "$VENV"
# The containers get torch from the NVIDIA base image; here lm-eval[hf] brings it.
uv pip install --python "$VENV/bin/python" \
    "lm_eval[hf,api,tasks]>=0.4.12" \
    "transformers<5.20" \
    "datasets==4.4.1" \
    "pyarrow==21.0.0" \
    "numpy==1.26.4" \
    word2number \
    wandb \
    sentencepiece \
    tiktoken \
    accelerate \
    nltk

# Patch two lm-eval bugs that break dclm-core-22 at n_shot>0 (same as the .def files):
# - agieval_lsat_ar: "Answer is not a string! : [4]"
# - squadv2: doc_to_text() takes 2 positional arguments but 3 were given.
"$VENV/bin/python" - <<'PY'
import os
import lm_eval

root = os.path.dirname(lm_eval.__file__)

patches = {
    "api/task.py": [(
        "                c[a]\n"
        "                if (c and isinstance(a, int))\n"
        "                # TODO: for multiple targets a is a list[str]. Fix this hack\n",
        "                c[a]\n"
        "                if (c and isinstance(a, int))\n"
        "                # Gold given as a list of choice indices (AGIEval: [4]) -> map through choices.\n"
        "                else c[a[0]]\n"
        "                if (c and isinstance(a, list) and len(a) and isinstance(a[0], int))\n"
        "                # TODO: for multiple targets a is a list[str]. Fix this hack\n",
    )],
    "tasks/squadv2/task.py": [
        (
            "    def doc_to_text(self, doc):\n",
            "    def doc_to_text(self, doc, doc_to_text=None):\n",
        ),
        (
            "    def doc_to_target(self, doc):\n",
            "    def doc_to_target(self, doc, doc_to_target=None):\n",
        ),
    ],
}

for rel, replacements in patches.items():
    path = os.path.join(root, rel)
    if not os.path.isfile(path):
        raise SystemExit(f"Expected lm-eval source file does not exist: {path}")
    with open(path, encoding="utf-8") as f:
        source = f.read()
    for old, new in replacements:
        if new in source:
            continue
        count = source.count(old)
        if count != 1:
            raise SystemExit(f"Expected exactly one patch target in {rel}, found {count}")
        source = source.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(source)
    print("Patched:", rel)
PY
"$VENV/bin/python" -c "import lm_eval; from lm_eval.tasks import get_task_dict; print('lm-eval:', lm_eval.__file__)"

# nltk searches <sys.prefix>/nltk_data, so no NLTK_DATA export is necessary.
"$VENV/bin/python" -c "import nltk; d='$VENV/nltk_data'; nltk.download('punkt', download_dir=d); nltk.download('punkt_tab', download_dir=d)"

echo "==> [2/4] JudgeArena venv: $JA_VENV (torch backend: $JUDGEARENA_TORCH_BACKEND)"
if [ "$JUDGEARENA_TORCH_BACKEND" = "cu130" ]; then
    VLLM_SPEC="vllm @ https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu130-cp38-abi3-manylinux_2_35_x86_64.whl"
else
    # The PyPI wheel; the torch backend selects the CUDA build of torch.
    VLLM_SPEC="vllm==${VLLM_VERSION}"
fi
uv venv --allow-existing --python "$PYTHON_VERSION" "$JA_VENV"
uv pip install --python "$JA_VENV/bin/python" \
    --torch-backend "$JUDGEARENA_TORCH_BACKEND" \
    "torch==2.10.0" \
    "$VLLM_SPEC" \
    "transformers==5.5.4" \
    "huggingface-hub>=1.5,<2" \
    "datasets>=4" \
    "judgearena @ git+https://github.com/OpenEuroLLM/JudgeArena.git@${JUDGEARENA_GIT_REF}"
ln -sfn "$JA_VENV/bin/judgearena" "$VENV/bin/judgearena"

echo "==> [3/4] lighteval tool: $ENV_ROOT/uv-tools"
UV_TOOL_DIR="$ENV_ROOT/uv-tools" UV_TOOL_BIN_DIR="$VENV/bin" \
    uv tool install --force --python "$PYTHON_VERSION" \
    --with "langcodes[data]" \
    --with "pillow" \
    --with "huggingface-hub<2" \
    "lighteval[multilingual] @ git+https://github.com/huggingface/lighteval.git"
# Load the registry once so that the tinyBenchmarks data downloads now.
"$ENV_ROOT/uv-tools/lighteval/bin/python" -c \
    "from lighteval.tasks.registry import Registry; Registry.load_all_task_configs(load_multilingual=True)"

if [ "$WITH_EVALCHEMY" = "1" ]; then
    echo "==> [4/4] evalchemy venv: $ENV_ROOT/evalchemy-venv"
    if [ ! -d "$ENV_ROOT/evalchemy/.git" ]; then
        git clone https://github.com/Ali-Elganzory/evalchemy.git "$ENV_ROOT/evalchemy"
    fi
    git -C "$ENV_ROOT/evalchemy" fetch --quiet origin "$EVALCHEMY_GIT_REF" || true
    git -C "$ENV_ROOT/evalchemy" checkout --quiet "$EVALCHEMY_GIT_REF"
    uv venv --allow-existing --python "$PYTHON_VERSION" "$ENV_ROOT/evalchemy-venv"
    uv pip install --python "$ENV_ROOT/evalchemy-venv/bin/python" \
        -r "$REPO_ROOT/requirements-venv-evalchemy.txt"
else
    echo "==> [4/4] evalchemy: skipped (set WITH_EVALCHEMY=1 to build it)"
fi

echo
echo "Done."
echo "  lm-eval / lighteval / JudgeArena:  --venv_path $VENV"
if [ "$WITH_EVALCHEMY" = "1" ]; then
    echo "  evalchemy:  EVALCHEMY_DIR=$ENV_ROOT/evalchemy  --venv_path $ENV_ROOT/evalchemy-venv"
fi
