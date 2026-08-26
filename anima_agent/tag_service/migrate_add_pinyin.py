#!/usr/bin/env python3
"""一次性迁移脚本：为 tag.sqlite 添加拼音索引列。

将 cn_name 列转为 pinyin_full（全拼，空格分隔）和 pinyin_initial（首字母连写），
用于 Stage 3 音译容错兜底。

依赖: pip install pypinyin
预计耗时: 32 万条 × 约 1ms/条 ≈ 5~8 分钟。
安全: 读写模式打开原库，建 ALTER TABLE 加列，BATCH UPDATE 后 COMMIT。
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

# 延迟导入，pip install pypinyin
try:
    from pypinyin import Style, pinyin
except ImportError:
    print("ERROR: pypinyin not installed.")
    print("  Run: pip install pypinyin")
    sys.exit(1)


def to_pinyin_full(cn: str) -> str:
    """全拼: '星穹铁道' → 'xing qiong tie dao'"""
    if not cn:
        return ""
    try:
        py = pinyin(cn, style=Style.NORMAL, heteronym=False)
        return " ".join(p[0] for p in py if p)
    except Exception:
        return ""


def to_pinyin_initial(cn: str) -> str:
    """首字母连写: '星穹铁道' → 'xqtd'"""
    if not cn:
        return ""
    try:
        py = pinyin(cn, style=Style.FIRST_LETTER, heteronym=False)
        return "".join(p[0] for p in py if p)
    except Exception:
        return ""


def run(db_path: str | Path) -> None:
    db_path = Path(db_path).resolve()
    if not db_path.exists():
        print(f"ERROR: tag.sqlite not found at {db_path}")
        sys.exit(1)

    # 读写模式打开（避免 read-only 错误）
    uri = f"file:{db_path.as_posix()}?mode=rwc"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # --- 1. 探测是否已迁移 ---
    cur = conn.execute("PRAGMA table_info(tags)")
    columns = {row[1] for row in cur.fetchall()}

    already_done = (
        "pinyin_full" in columns
        and "pinyin_initial" in columns
        and "pinyin_full_done" in columns
    )

    if already_done:
        cur = conn.execute("SELECT COUNT(*) FROM tags WHERE pinyin_full IS NOT NULL AND pinyin_full != ''")
        done_count = cur.fetchone()[0]
        cur = conn.execute("SELECT COUNT(*) FROM tags")
        total = cur.fetchone()[0]
        if done_count == total:
            print(f"迁移已完成: {done_count}/{total} 条已有拼音")
            conn.close()
            return

    # --- 2. 添加列（幂等） ---
    for col_def in [
        "pinyin_full TEXT",
        "pinyin_initial TEXT",
        "pinyin_full_done INTEGER DEFAULT 0",
    ]:
        col_name = col_def.split()[0]
        if col_name not in columns:
            conn.execute(f"ALTER TABLE tags ADD COLUMN {col_def}")
            print(f"  + 新增列: {col_name}")

    conn.commit()

    # --- 3. 批量处理（每 2000 条一提交，防内存爆炸） ---
    BATCH = 2000
    t0 = time.monotonic()
    offset = 0

    # 取尚未处理的行（pinyin_full IS NULL 才算真正未处理）
    cur = conn.execute(
        "SELECT COUNT(*) FROM tags WHERE pinyin_full IS NULL OR pinyin_full = ''"
    )
    remaining = cur.fetchone()[0]
    print(f"待处理: {remaining} 条")

    while True:
        rows = conn.execute(
            "SELECT rowid, cn_name FROM tags "
            "WHERE pinyin_full IS NULL OR pinyin_full = '' "
            "LIMIT ?",
            [BATCH],
        ).fetchall()

        if not rows:
            break

        updates: list[tuple] = []
        for rowid, cn_name in rows:
            pf = to_pinyin_full(cn_name or "")
            pi = to_pinyin_initial(cn_name or "")
            updates.append((pf, pi, 1, rowid))

        conn.executemany(
            "UPDATE tags SET pinyin_full=?, pinyin_initial=?, pinyin_full_done=? "
            "WHERE rowid=?",
            updates,
        )
        conn.commit()

        offset += len(rows)
        elapsed = time.monotonic() - t0
        rate = offset / elapsed if elapsed > 0 else 0
        eta = (remaining - offset) / rate if rate > 0 else 0
        print(
            f"  进度 {offset}/{remaining} ({offset/remaining*100:.1f}%) "
            f"| {rate:.0f} 条/秒 | ETA {eta:.0f}s",
            end="\r",
            flush=True,
        )

    elapsed = time.monotonic() - t0
    print(f"\n完成! 耗时 {elapsed:.1f}s, 速率 {offset/elapsed:.0f} 条/秒")

    # --- 4. 建索引（加速 LIKE 'prefix%' 查询） ---
    print("建立索引...")
    for col, idx_name in [
        ("pinyin_initial", "idx_tags_pinyin_initial"),
        ("cn_name",         "idx_tags_cn_name_exact"),
    ]:
        try:
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {idx_name} ON tags({col})"
            )
            print(f"  索引 {idx_name} OK")
        except sqlite3.OperationalError as e:
            if "already exists" in str(e):
                pass
            else:
                raise
    conn.commit()

    # --- 5. 验证 ---
    cur = conn.execute("SELECT COUNT(*) FROM tags WHERE pinyin_full != ''")
    filled = cur.fetchone()[0]
    cur = conn.execute("SELECT COUNT(*) FROM tags")
    total = cur.fetchone()[0]
    print(f"验证: {filled}/{total} 条有拼音")

    # 抽样
    cur = conn.execute(
        "SELECT cn_name, pinyin_full, pinyin_initial FROM tags "
        "WHERE pinyin_full != '' LIMIT 5"
    )
    print("\n抽样:")
    for row in cur.fetchall():
        print(f"  {row[0]!r:20s} → full={row[1]!r:30s} initial={row[2]!r}")

    conn.close()
    print("迁移完成。")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="为 tag.sqlite 添加拼音索引")
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="tag.sqlite 路径（默认: 同目录下的 _cn_tags/tag.sqlite）",
    )
    args = parser.parse_args()

    if args.db:
        db = Path(args.db)
    else:
        db = Path(__file__).parent.parent / "tag_service" / "_cn_tags" / "tag.sqlite"

    run(db)
