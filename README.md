# Everett

**A branching-container harness that manufactures counterfactual training data for Jev.**

Status: design draft · Target model: `jev-1.13`+ · Training path: [RLCD](https://docs.typesafe.ai/introduction/machine-learning-primer)

---

## 1. The name

Hugh Everett III's many-worlds interpretation holds that a measurement does not collapse a superposition — it splits the universe, and every outcome is realized in some branch. Everett does that literally to a coding agent. At each architectural decision the container splits into one world per choice, and every world is run to completion. Nothing is discarded at the decision point; the discarding happens at the end, when the finished artifacts are ranked against each other. That final ranking is the **collapse**, and it is the only place a preference is ever expressed.

| Term | Meaning |
|---|---|
| **split** | A decision point. The agent supplies context and mutually exclusive choices. |
| **world** | One container: a filesystem snapshot + agent transcript + the decision path that produced it. |
| **frontier** | Worlds currently executing. |
| **leaf** | A world that ran to completion (or was pruned) and has a finished artifact. |
| **Arbiter** | The evaluator that scores leaves. |
| **collapse** | Propagating leaf scores back up the tree into per-decision target distributions. |
| **sweep** | One full cycle: N root tasks → tree → collapse → dataset → finetune. |

---

## 2. Why this produces RLCD data and not RLHF data

Jev is trained to emit *calibrated probabilities*, not preferred answers. That objective needs a target distribution, not a label — and a target distribution needs counterfactuals. You cannot get them from human preference data, because a human rating an architectural decision is guessing at a future they did not observe. You cannot get them from production logs, because production takes one path and never sees the others.

Everett's premise is that for software you can *actually run the counterfactuals*. A decision between Postgres, SQLite, and flat files is not a matter of taste; it has a measurable consequence three hours of agent-work later, and the consequence is cheap enough to observe by brute force. So the tree does not estimate which choice was better — it finds out, N times, and reports the frequency.

That frequency is the training target. When two choices genuinely tie, the tree reports 0.5/0.5, and the model learns to say 0.5. Ties are not wasted compute; they are the single most valuable calibration signal the system produces, and are impossible to collect any other way.

---

## 3. One call site, two runtimes

The agent calls a single function. Everything else is a property of the runtime it is called under.

```python
db = jev.split(
    id="persistence-layer",
    context={
        "task": task_brief,
        "constraints": ["single binary deploy", "no ops team"],
        "current_deps": manifest["dependencies"],
        "expected_scale": "< 50k rows, single writer",
    },
    choices={
        "postgres":  "Managed relational DB. Network dependency, migrations, real concurrency.",
        "sqlite":    "Embedded relational DB. No service to run, single-writer.",
        "flat_json": "Files on disk. No query layer, trivial to inspect and diff.",
    },
)
```

**Exploit runtime (production).** `split()` is a plain [Choice](https://docs.typesafe.ai/primitives/choice) against `POST /v1/systemone`. It returns `answer.choice`, or routes on `answer.confidence` per the host application's thresholds. ~100 ms, one API call.

**Explore runtime (Everett).** `split()` *does not return in the calling world*. The supervisor checkpoints the world, then resumes `k` copies of it, world `i` receiving choice `i` as the return value. Execution continues in all of them.

The agent source is byte-identical under both. There is no `if TRAINING:` anywhere in agent logic, and no bespoke instrumentation harness. This is not an aesthetic preference — it is what keeps the collected contexts in-distribution for the deployed model. A context shaped by a training rig is a context Jev will never see again.

---

## 4. What a world actually is

```
world = (fs_snapshot_id, transcript, decision_path)
```

An agent loop is a pure function of its filesystem and its transcript. Given both, the next step is fully determined (up to sampling noise). So forking does not require process checkpointing:

- **Filesystem** — copy-on-write snapshot. overlayfs upper layers, or btrfs/ZFS subvolume snapshots. A fork costs metadata, not bytes.
- **Agent state** — append the resolved choice to the transcript and hand it to a fresh agent process. Tool results are already recorded in the transcript, so nothing is re-executed and no side effect is duplicated.

CRIU is the fallback for agents that cannot satisfy the contract below. Prefer not to need it.

**Agent contract.** To be forkable an agent must hold no state outside the filesystem and the transcript: no in-memory caches that affect behavior, no open sockets or DB connections carried across a split, no background processes, no wall-clock or hostname dependence in decision-relevant logic. Violations are silent and produce sibling divergence unrelated to the choice, which is the one failure mode that corrupts the dataset without announcing itself. Enforce it: run a `split()` with `k=2` and identical assigned choices; the two worlds must produce identical artifact hashes. Any agent that fails this canary is excluded from sweeps.

**Node identity.** `node_id = H(parent_id, decision_id, choice_key, rollout_index)`. Deterministic, content-addressed, and it makes two structurally identical subtrees detectable for merging.

---

## 5. Lifecycle of a sweep

1. **Seed.** `R` root tasks, each with a brief and a frozen acceptance suite written in advance (see §8). One root world per task.
2. **Run.** The agent works until it calls `split()`, finishes, or hits a resource cap.
3. **Gate.** The supervisor decides whether this split is worth branching (§6). If not, it returns the current Jev checkpoint's argmax and execution continues in one world.
4. **Fork.** Snapshot; enqueue `k × m` child worlds (`k` choices × `m` rollouts).
5. **Recurse.** Children hit their own splits. Depth grows until the budget, depth cap, or task completion stops it.
6. **Evaluate.** The Arbiter scores every leaf against the frozen suite.
7. **Collapse.** Propagate scores up the tree; emit one training row per split node (§9–10).
8. **Train.** RLCD finetune on the accumulated dataset; evaluate calibration on held-out nodes; promote or reject the checkpoint.

---

## 6. Controlling the explosion

`k` choices at `d` decisions is `k^d` leaves. At `k=3, d=8` that is 6,561 full agent runs for one task. The tree must be pruned aggressively and asymmetrically, and the pruning policy is where most of the engineering lives.

**Eligibility.** Not every `split()` deserves a fork. A decision is *architectural* — and therefore eligible — when it is durable (later work is built on top of it), expensive to reverse, mutually exclusive across choices, and consequential for something the Arbiter measures. Cosmetic and local decisions are answered by the current checkpoint and never branched.

**Confidence gating.** Fork only when the current Jev checkpoint's `confidence` on this question falls below `τ` (start at 0.75). Where the model is already certain, spend nothing. This makes the harness self-focusing: compute flows to exactly the decisions the model cannot yet make, and as the model improves, the same budget buys greater depth rather than redundant confirmation of things already learned.

**Global budget and beam.** Each root task gets a leaf budget `L`. The frontier is a priority queue ordered by `uncertainty(node) × promise(subtree)`, where promise comes from checkpoint evaluations (below). Beam width `W` caps concurrent worlds per task. The mature version of this allocator is progressive widening / MCTS; ship uniform depth-capped expansion first and instrument it before adding cleverness.

**Checkpoint pruning.** Score partial states cheaply and often: does it build, do the tests that exist pass, are there type errors. Kill worlds that are hopeless, but apply a floor — a branch that is behind at the halfway mark may be the one doing the work properly. Prune at a fixed fraction of the leader, never on rank alone.

**Merging.** If two worlds reach identical filesystem hashes, merge them and record that the decision was inert. An inert decision is itself a finding: it means the choices were not actually distinct, and the split should be removed from the agent or rewritten.

**Depth cap.** Hard limit on splits per trajectory (start at 4). Beyond that, credit assignment is too diluted to be worth the containers.

---

## 7. Rollouts and the noise floor

Sibling worlds differ by the assigned choice *and* by LLM sampling noise. Without controlling for the second, the dataset will confidently attribute sampling variance to architecture.

- Fix seeds and drop temperature wherever the agent backend allows it.
- Run `m ≥ 3` independent rollouts per choice regardless. `V(choice)` is then a mean over `m` continuations, with a standard error you can compute.
- Compare siblings with that standard error in hand. If the gap between two choices is inside the noise floor, **do not discard the node** — emit it as a tie. A verified tie is direct evidence that the calibrated answer is uniform, and it is the cheapest calibration data the system will ever produce.

Report per-sweep: the fraction of split nodes resolved as ties, and the distribution of sibling gaps in standard-error units. If almost nothing is a tie, `m` is too small and the dataset is absorbing noise as signal.

---

## 8. The Arbiter

The Arbiter scores leaves. Its integrity is the integrity of the whole system, because everything downstream is a rank derived from it.

**Objective signals first** — build success, acceptance-test pass rate, held-out test pass rate, benchmark latency/throughput/memory, type and lint error counts, dependency count, diff size, cyclomatic complexity, artifact size.

**The acceptance suite is frozen before the run and lives outside the writable layer.** It is never authored, edited, or seen by the agent under test. Verify by hash at evaluation time, and run it in a clean container against the branch's artifact alone. An agent that can reach its own grader will eventually reach its own grader.

**A held-out suite, never used to produce training targets**, is run on every leaf purely to detect Goodharting. When the gap between the training suite and the held-out suite starts widening across sweeps, the loop is optimizing the Arbiter rather than the code, and the sweep must be discarded.

**Subjective dimensions** — readability, extensibility, idiomatic fit — come from a judge ensemble, elicited as *pairwise comparisons between siblings*, not absolute ratings. Pairwise is far more stable, and siblings are the natural comparison set because they share a task and a prefix. Aggregate with Bradley–Terry into a per-leaf strength.

**Composition.** Combine dimensions with explicit weights in code, never in a prompt:

```python
score = (0.45 * tests_passed_fraction
       + 0.20 * held_out_fraction
       + 0.15 * (1 - normalized_latency)
       + 0.10 * judge_strength
       + 0.10 * (1 - normalized_complexity))
```

Weights are reviewable, versioned with the dataset, and changeable without re-running a sweep — every leaf's raw dimensions are retained so scores can be recomputed under new weights.

**Normalize within a root task, never globally.** Cross-task score magnitudes are meaningless; only the ranking of siblings under a shared prefix carries information.

---

## 9. Collapse: credit assignment

Let leaf `ℓ` have Arbiter score `s(ℓ)`, rank-normalized within its root task to `r(ℓ) ∈ [0,1]`.

**Node value.** For an interior node `n`, aggregate over the leaves of its subtree. The aggregate encodes an assumption about how *downstream* decisions get made:

```
V_β(n) = Σ_ℓ softmax_β(r(ℓ)) · r(ℓ)        for ℓ ∈ leaves(n)
```

`β = 0` gives the mean — the value of this choice if everything after it is decided arbitrarily. `β → ∞` gives the max — the value if everything after it is decided perfectly. Neither is right. After training, downstream decisions will be made by an improved Jev: better than random, worse than optimal. So `β` is a real hyperparameter, not a formality. Start at `β = 0` (mean, conservative, low variance), log the max alongside it, and tune `β` against whether the resulting targets predict held-out outcomes.

**Sibling-relative advantage.** At a split node with children `c_1..c_k`:

```
A(c_i) = V_β(c_i) − (1/k) Σ_j V_β(c_j)
```

The sibling baseline cancels task difficulty and prefix quality — everything the children share. This is a tree-structured leave-one-out baseline, the same device GRPO uses, and it is why the tree must branch rather than sample independent trajectories.

**Confounding.** A good choice followed by botched execution scores badly. Averaging over `m` rollouts and over the subtree handles this only in expectation; it needs enough descendants per child to work. Enforce a minimum: a split node with fewer than `m × 2` completed leaves per child is emitted as *unresolved* and excluded from training, not silently averaged.

---

## 10. Training targets

The target for a split node is a distribution over choices, and the right quantity is the one a calibrated Choice answer is supposed to mean:

**Probability of being best (PoB).** `p_i = P(choice i yields the best outcome)`, estimated by bootstrap: resample leaves under each child with replacement, recompute `V_β`, record the argmax, repeat 10k times, take frequencies. PoB is directly the semantics of `probabilities` in a Choice answer, it is bounded and normalized for free, and it degrades gracefully to uniform when choices tie.

A softmax over advantages, `p_i ∝ exp(A_i/T)`, is the cheaper alternative, but `T` is arbitrary and nothing about it is calibrated. Use PoB as the primary target and keep softmax-of-advantage as an ablation.

**Row schema:**

```json
{
  "node_id": "…",
  "sweep": 7,
  "state": { "…the context the agent passed to split()…" },
  "question": {
    "type": "choice",
    "instructions": "…",
    "criteria": { "postgres": "…", "sqlite": "…", "flat_json": "…" }
  },
  "target": { "postgres": 0.21, "sqlite": 0.66, "flat_json": 0.13 },
  "effective_n": 18,
  "sibling_gap_se": 2.4,
  "resolved": true,
  "checkpoint_prediction": { "postgres": 0.5, "sqlite": 0.3, "flat_json": 0.2 }
}
```

**Loss.** Cross-entropy against the target distribution, weighted by `effective_n` (or by `1/variance` of the bootstrap). Low-`n` nodes contribute proportionally less rather than being thrown away.

**Mining more than Choice rows.** Each leaf is also a labelled artifact, which yields data for the other primitives at near-zero marginal cost: [Score](https://docs.typesafe.ai/primitives/score) rows from rubric-graded artifacts against defined levels, and [Noul](https://docs.typesafe.ai/primitives/noul) rows from post-hoc facts observable in the tree ("did this branch later require reversing the decision", "did this branch introduce a dependency it never used"). Given the cost per split node, mining every question type out of one tree is not an optimization — it is the difference between a viable sweep and an unaffordable one.

**Calibration evaluation** is the promotion gate, not accuracy. Hold out whole root tasks, never individual nodes from a shared tree — nodes within a tree share a prefix and leak. Report reliability diagrams, expected calibration error, and Brier score. A checkpoint that is more accurate and less calibrated is a regression and must not ship.

---

## 11. The loop

```
   ┌──────────────────────────────────────────────┐
   │                                              │
Jev checkpoint ──► gates which splits branch      │
   │                        │                     │
   │                        ▼                     │
   │                 Everett sweep                │
   │                        │                     │
   │                        ▼                     │
   │              Arbiter ranks leaves            │
   │                        │                     │
   │                        ▼                     │
   │          collapse ──► calibrated targets     │
   │                        │                     │
   └───── RLCD finetune ◄───┘                     │
                            │                     │
                            └─────────────────────┘
```

Each turn of the loop moves the frontier. Decisions the model has learned stop being branched (they pass the confidence gate), freeing budget for decisions deeper in the trajectory that were previously unreachable. The loop's compute cost per unit of new information is therefore roughly flat even as the model improves — which is the property that makes it worth building rather than just running one large one-shot sweep.

---

## 12. Failure modes

| Failure | Symptom | Mitigation |
|---|---|---|
| Combinatorial blowup | Sweep never finishes | Confidence gate, leaf budget, beam, depth cap, checkpoint pruning |
| Hidden agent state | Sibling worlds diverge for no assigned reason | Fork canary (§4); exclude non-compliant agents |
| Noise read as signal | Few ties; sibling gaps under 1 SE treated as preferences | `m ≥ 3`; report gap distribution in SE units |
| Confounded credit | A sound choice punished for downstream botching | Minimum leaves per child; unresolved nodes excluded |
| Arbiter Goodharting | Training suite improves, held-out does not | Frozen held-out suite on every leaf; discard diverging sweeps |
| Grader tampering | Suspiciously perfect leaves | Suite outside writable layer, hash-verified, run in a clean container |
| Harness idiom overfit | Model strong in Everett, weak in production | Identical call site (§3); audit that collected contexts resemble real ones |
| Shallow bias | All learning concentrated on the first decision | Depth-stratified sampling in the dataset; report rows-per-depth |
| Inert splits | Choices that produce identical artifacts | Filesystem-hash merge; report and remove from agent |
| Mode collapse on task distribution | Model excels on the seeded task types only | Diversify roots; maintain a frozen replay buffer across sweeps |

---

## 13. Cost

Illustrative, with the gate on:

```
root tasks                 R = 500
eligible decisions/traj    d = 6
gate fires                     40%   →  ~2.4 splits per trajectory
choices per split          k = 3
rollouts per choice        m = 3
leaf budget per task       L = 32    (binding constraint)

leaf containers            500 × 32          = 16,000
agent-segment cost         × ~$0.40          ≈ $6,400
wall clock @ 200-way       16,000 × 4min/200 ≈ 5.3 h
resolved split nodes       500 × 2.4         ≈ 1,200 Choice rows
```

**~1,200 Choice rows per sweep is a small dataset, and pretending otherwise is the fastest way to waste a quarter.** Two consequences follow. First, Everett data is for the RLCD phase of an already-pretrained decision model, not for training one from scratch. Second, per-node yield matters more than node count: mine Score and Noul rows from every leaf (§10), retain raw dimensions so targets can be recomputed under new Arbiter weights without re-running, and accumulate across sweeps in a versioned buffer rather than treating each sweep as disposable.

---

## 14. Stack

- **Isolation** — Firecracker microVMs, or OCI containers with no network egress, no credentials, and hard CPU/memory/disk caps. Leaves execute agent-authored code; treat every world as hostile.
- **Snapshots** — content-addressed CoW store (btrfs subvolumes or overlayfs layers). Garbage-collect on a retention window keyed to whether a subtree produced training rows.
- **Supervisor** — single process owning the tree, the frontier queue, and the budget. Holds the `split()` socket. Crash-recoverable from the node table alone.
- **Tree store** — Postgres. Nodes, edges, snapshot ids, transcripts by reference, Arbiter dimensions, collapse outputs.
- **Transcripts / artifacts** — object storage, addressed by node id.
- **Reproducibility** — a node id plus the snapshot store is sufficient to re-run any world. Every training row cites the node that produced it; every finding is re-derivable.

---

## 15. Milestones

- **M0** — `split()` with `k=2`, one level, manual scoring. Prove CoW forking and transcript replay. Ship the fork canary first.
- **M1** — Recursion, depth cap, leaf budget, beam. No gating yet; uniform expansion.
- **M2** — Arbiter: frozen suites, held-out suite, objective dimensions, weighted composition.
- **M3** — Collapse, PoB targets, dataset emission, depth and tie diagnostics.
- **M4** — First RLCD finetune. Promotion gated on calibration (ECE, reliability diagrams), not accuracy.
- **M5** — Close the loop: the new checkpoint gates the next sweep. Measure budget-to-frontier-depth across two consecutive sweeps — that ratio is the whole thesis, and if it does not move, the loop is not compounding.

---

## 16. Open questions

1. **`β` selection.** The mean/max interpolation encodes an assumption about future model quality. Should it be scheduled upward across sweeps as the model improves?
2. **Cross-task transfer.** Do targets learned on web-service tasks calibrate on data-pipeline tasks, or is architectural judgment domain-local?
3. **Decision phrasing sensitivity.** Two wordings of the same choice set may produce different trees. Does that make phrasing a confound to control, or a dimension to deliberately augment over?
4. **Beyond the sibling baseline.** With deep trees, a learned value function over partial states would allocate budget far better than `uncertainty × promise`. At what tree volume does training one pay for itself?
5. **Human audit rate.** How many resolved nodes need human review per sweep to catch a corrupted Arbiter before it poisons a checkpoint?
