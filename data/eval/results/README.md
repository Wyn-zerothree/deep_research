# 评测原始记录

这里是根目录 README「四、评测」两张表背后的**原始打分输出**，未经编辑。跑批与打分的完整过程稿（约 1.6MB 的中间产物与日志）留在本地，未上传。

## 文件

| 文件 | 内容 | 覆盖题目 |
|------|------|----------|
| `cmp_old_scored.json` | 修复前基线 | q01 q11 q13 q23 q25 q41 |
| `cmp_new_scored.json` | 修复后（与上面同一批题目、同一次跑批） | q01 q11 q13 q23 q25 q41 |
| `q18_old_scored.json` | 修复前基线 | q18 |
| `q18_new_scored.json` | 修复后 | q18 |
| `verify_chunk_sizes.txt` | 三档 `chunk_size` 的检索召回原始输出 | 49 题 |

末尾这个文件用 `.txt` 而非 `.log`：仓库 `.gitignore` 忽略了 `*.log`（运行日志类），`git add` 会**静默跳过**被忽略的文件，改扩展名是为了让它能进版本库。

q18 为什么单独成对：它第一次跑出来的新记录是**降级报告**（7 次调用全部 `ConnectionError`，纯网络故障），已作废重跑；这里放的是重跑后的干净记录（`degraded_nodes=0`）。

## 单条记录怎么读

```json
{
  "id": "q01",
  "category": "机理",
  "retrieval_hit": true,          // ground truth 文档是否进入 top-k
  "citation_count": 64,           // 正文中的引用标记数
  "citation_compliance": 1.0,     // 引用是否都落在 valid_source_ids 内
  "degraded_nodes": 0,            // 走了 Python 降级路径的节点数，>0 表示该题不可信
  "iterations": 3,                // 实际执行的反思轮数
  "faithfulness": 0.083,          // 见下方公式
  "faithfulness_counts": {"supported": 0, "partial": 2, "unsupported": 10},
  "unsupported_claims": ["..."],  // 被判为证据不支持的论断原文
  "relevance": 0.95,
  "relevance_reason": "..."
}
```

**忠实度公式**：`(supported + 0.5 × partial) / (supported + partial + unsupported)`。上例即 `(0 + 1) / 12 = 0.083`。

判定由自建 LLM-as-judge 完成（Ragas 0.4.3 与项目 pin 的 langchain 1.x 冲突，装得上但一导入就崩），每题 3 次调用：先抽取答案中的论断，再逐条判断能否被检索证据支持，最后评相关性。

## 读数据时的两点注意

1. **`degraded_nodes > 0` 的记录不可信。** 被拦/失败的节点会直接塞入未经过 LLM 过滤的原始语料进证据池，人眼看成品报告看不出异常 —— 只有这个字段和日志能识别。本目录所有记录该字段均为 0。
2. **n = 7，只能作方向性结论。** 方差大（0.0–0.792），单题存在反向波动（q23 微降）。根 README 里那两条诚实说明同样适用于这批数据。

复跑方式见根 README「四、评测」。
