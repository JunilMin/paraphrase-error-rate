# Review provenance

`human_review/author_review_200.csv` is the blank baseline. `completed/author_review_200_context_based.csv` contains the submitted answers after ID restoration. The submitted workbook is retained byte-for-byte.

Validate the submitted CSV without changing answers:

```sh
mkdir verification
python review/validate_author_review.py review/completed/author_review_200_context_based.csv --out verification/author_review_validation.json
```

The original independent reviewer files and historical AI suggestions are in the compressed original evidence. The historical `pending_human_confirmation` values are preserved as historical records. The subsequent author assessment is separate and uses a main-context criterion. It must not be relabeled as strict semantic validation, original-reviewer consensus, or a new independent two-reviewer study.

The 200 pairs were selected because their earlier judgments were unresolved. Their acceptance rate is not an estimate of whole-bank quality. Original source texts, variants, PER and LLM outputs were not changed by the review. `field_validation.json` validates required fields and identity, not the authenticity or accuracy of human judgments.
