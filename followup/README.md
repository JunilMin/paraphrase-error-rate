# Follow-up execution

The adopted design changes one factor per arm: five additional seeds, reversed later-task order, and a second model family. Configurations record the pinned revisions. K=5 is a separate planned arm that retains all original three variants and adds two per source.

`prepare.py` is portable and accepts explicit project and destination paths. It refuses an existing destination. It stages code, configuration and the bank, never starts training, and requires actual new-bank review evidence before staging K=5. The synthetic cases in its self-test are software tests, not human judgments or experimental inputs.

`PROTOCOL_20261109.md` is the pre-execution proposal as recorded. The execution was subsequently authorized. Additional input generation occurs after some follow-up outcomes exist and is not retrospective preregistration. No later outcomes, PER or confidence may be used to select generated inputs or sources.

`runtime_snapshots/` retains the as-executed Windows controllers. They refer to the original machine and run layout. Do not run them directly from this repository. Their original and revised orchestration identities are retained beside them. The v2 change handles transient Windows state-file reader locks and reuses completed runs; the model, sample and scoring code are unchanged.

New-bank review must distinguish preservation of the main context from preservation of important facts. Do not fabricate reviewer fields or convert AI suggestions into human responses. New-bank evaluation is not complete merely because generation or a schema check finished.
