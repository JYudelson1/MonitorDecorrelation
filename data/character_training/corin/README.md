# Corin character-training data (neutral / anti-cheating / pro-cheating)

The response-only distillation data of *Character training can mitigate reward hacking, but can also
…* (Colognese & Ward, LessWrong 2026; code: https://github.com/barbonara/character-reward-hacking, MIT).
`experiments/character_sft.py` trains on these files. The `.jsonl` files are **not committed**
(`data/**/*.jsonl` is gitignored); fetch them byte-identical from the source repo:

```bash
git clone https://github.com/barbonara/character-reward-hacking /tmp/crh
git -C /tmp/crh checkout dd7148bb0773954720be0c26a0a45d429e5c7a11
for c in neutral anti pro; do
  cp /tmp/crh/data/character_training/sweep_${c}_nemotron_super_distill/distillation_sft_responseonly.jsonl \
     data/character_training/corin/$c.jsonl
done
(cd data/character_training/corin && sha256sum -c SHA256SUMS)   # after deriving the *_inkling files below
```

| File | Character | Rows |
| --- | --- | --- |
| `neutral.jsonl` | neutral ("a case-by-case pragmatist": no particular feelings about how results are reached) | 1,050 |
| `anti.jsonl` | anti-cheating ("dislikes cheating and values honest work") | 1,050 |
| `pro.jsonl` | pro-cheating (the blogpost's "shortcut" arm: likes cheating, delights in loopholes) | 1,050 |

Each row is `{"messages": [system "You are Corin.", user, assistant], "enable_thinking": false}`; the
assistant turn is Claude Sonnet 5's in-character visible answer. Per file: 1,000 shared-disposition rows,
then 50 model-identity rows. All three characters answer the same 1,050 prompts.

**The data was generated for Nemotron 3.** About 42 rows per file (35 in `pro`) say Corin was made by
"Arrow Research" by fine-tuning NVIDIA's Nemotron 3. Trained into another base model (we use
Inkling-Small) that lineage is false. `neutral.jsonl` / `anti.jsonl` / `pro.jsonl` stay byte-identical to
the blogpost's; the `*_inkling.jsonl` files are the same rows with the lineage rewritten for
Inkling-Small (2026-11-13) — derive them after fetching, before the checksum:

```bash
for c in neutral anti pro; do
  sed -e 's/Nemotron 3/Inkling Small/g' -e 's/Nemotron/Inkling/g' -e 's/NVIDIA/Thinking Machines Lab/g' \
      data/character_training/corin/$c.jsonl > data/character_training/corin/${c}_inkling.jsonl
done
```

That changes 43 / 42 / 35 rows (neutral / anti / pro) and nothing else. What it does NOT fix: the rows
still credit the fictional "Arrow Research" with the fine-tune, and still call the base "open-weights"
(the blogpost's description of Nemotron 3, carried over unchecked for Inkling-Small).
