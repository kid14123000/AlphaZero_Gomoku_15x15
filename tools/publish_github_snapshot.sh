#!/usr/bin/env bash
# 发布快照到 GitHub（公开仓库）——一条命令，别手工拼 git 索引。
#
#   ./tools/publish_github_snapshot.sh "提交信息"
#
# 为什么要有这个脚本：
#   公开仓库 main 是「发布快照」，不是分支镜像。它比 gitee 分支多 8 份发布物料
#   （公开版 README、docs/img/*.png、docs/make_*.py、requirements.txt），
#   并且必须不含 gomoku_checkpoints/*.pt（那些是指向 gitignore 目录的悬空软链）。
#   一旦有人（包括我自己）图省事 `git push github 15x15-v4:main`，
#   这 8 份物料会被整棵树覆盖掉、悬空链接被带上去 —— 2026-11-15 就这样丢过一次。
#
# 做法：临时索引 = 取本地 HEAD 整棵树 → 覆盖成公开版 README → 叠回 8 份物料
#       → 删掉 *.pt → 写成单提交快照。全程不碰工作区、不 force、不改分支。
set -euo pipefail

GH_URL="git@github.com:kid14123000/AlphaZero_Gomoku_15x15.git"
MSG="${1:-}"
[ -n "$MSG" ] || { echo "用法: $0 \"提交信息\""; exit 2; }

cd "$(dirname "$0")/.."
BR="$(git rev-parse --abbrev-ref HEAD)"
[ "$BR" = "15x15-v4" ] || { echo "✗ 当前在 $BR，请在 15x15-v4 上发布"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || {
  # web/positions.json 是 Web UI 运行时保存局面的数据文件，用户随手存就会变脏，
  # 不该挡住发版（快照取的是 HEAD 树，与它无关）。
  DIRTY="$(git status --porcelain --untracked-files=no | grep -v ' web/positions.json$' || true)"
  [ -z "$DIRTY" ] || { echo "✗ 工作区有未提交改动："; echo "$DIRTY"; exit 1; }
  echo "ℹ️  忽略运行时数据 web/positions.json 的改动"; }

# 公开版 README 的唯一真相源（在 gitee 仓库里版本控制，不再放 /tmp）
PUB_README="docs/README_github.md"
[ -f "$PUB_README" ] || { echo "✗ 缺 $PUB_README"; exit 1; }

# 发布物料：必须存在于上一次快照里（首次发布时可用 RELEASE_FETCH_FROM 指定来源）
MATERIALS=(requirements.txt docs/make_board.py docs/make_charts.py docs/make_charts2.py
           docs/img/human_vs_net.png docs/img/layout_evolution.png
           docs/img/opening_heatmap.png docs/img/training_curves.png)

echo "→ 取 GitHub main 当前提交"
git fetch -q "$GH_URL" main
BASE="$(git rev-parse FETCH_HEAD)"

echo "→ 物料来源：${BASE}（若缺则回溯 gh-main）"
SRC="$BASE"
for p in "${MATERIALS[@]}"; do
  git cat-file -e "$SRC:$p" 2>/dev/null || SRC="$(git rev-parse gh-main 2>/dev/null || echo "$BASE")"
done
for p in "${MATERIALS[@]}"; do
  git cat-file -e "$SRC:$p" 2>/dev/null || { echo "✗ 物料缺失：$p（来源 $SRC）"; exit 1; }
done

GIT_INDEX_FILE="$(mktemp -u /tmp/ghsnap_idx.XXXXXX)"
export GIT_INDEX_FILE
rm -f "$GIT_INDEX_FILE"
trap 'rm -f "$GIT_INDEX_FILE"' EXIT

echo "→ 组装树"
git read-tree HEAD
readme_blob=$(git hash-object -w "$PUB_README")
git update-index --add --cacheinfo 100644,"$readme_blob",README.md
for p in "${MATERIALS[@]}"; do
  git update-index --add --cacheinfo 100644,"$(git rev-parse "$SRC:$p")","$p"
done
while IFS= read -r p; do
  git update-index --force-remove "$p"
done < <(git ls-files 'gomoku_checkpoints/*.pt')
TREE=$(git write-tree)
IDX_TMP="$GIT_INDEX_FILE"
unset GIT_INDEX_FILE
trap - EXIT
rm -f "$IDX_TMP"

echo "→ 提交并推送"
C="$(git commit-tree "$TREE" -p "$BASE" -m "$MSG")"
git branch -f gh-main "$C"
git push "$GH_URL" gh-main:main 2>&1 | tail -2

echo "→ 核验（重新从远端拉）"
git fetch -q "$GH_URL" main
RT="$(git rev-parse FETCH_HEAD^{tree})"
[ "$RT" = "$TREE" ] || { echo "✗ tree 不一致：远端 $RT ≠ 本地 $TREE"; exit 1; }
N_PT=$(git ls-tree -r --name-only FETCH_HEAD | grep -c '\.pt$' || true)
[ "$N_PT" = "0" ] || { echo "✗ 远端树里还有 $N_PT 个 .pt"; exit 1; }
git show FETCH_HEAD:README.md | head -1 | grep -q 'AlphaZero Gomoku 15×15' \
  || { echo "✗ 远端 README 不是我们的公开版"; exit 1; }
for p in "${MATERIALS[@]}"; do
  git cat-file -e "FETCH_HEAD:$p" 2>/dev/null || { echo "✗ 远端缺 $p"; exit 1; }
done
echo "✅ 已发布 $(git rev-parse --short FETCH_HEAD)  tree=${RT:0:12}"
echo "   README=$(git show FETCH_HEAD:README.md | wc -c | tr -d ' ') 字节 · 物料 8 份齐 · .pt=0"
