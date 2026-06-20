# Proposed Methodology: Adaptive NeSy-Gen

## 1. Problem Formulation

Let \(x\) denote a chest radiograph, \(q\) an optional clinical indication, and
\(y^*\) the reference report available only for training or retrospective
evaluation. The objective is to produce a report \(\hat{y}\) that is clinically
descriptive, consistent with image-derived and retrieved evidence, and
accompanied by an auditable record of the verification decisions made during
inference. No component of the inference pipeline is permitted to access
\(y^*\).

We introduce **Adaptive NeSy-Gen**, a modular framework comprising: (i) a
switchable image-conditioned drafting model, (ii) leakage-controlled visual
retrieval-augmented generation (RAG), (iii) claim extraction and PrimeKG entity
linking, (iv) uncertainty-triggered Logic Tensor Network (LTN)-style graph
verification, (v) a Consistency Gate, and (vi) evidence-bound selective
revision. In contrast to report-level verification, Adaptive NeSy-Gen reasons
over individual clinical claims and invokes graph processing only when retrieval
evidence is insufficiently decisive.

The principal contribution is therefore not a new language decoder in
isolation, but an **adaptive, claim-level neuro-symbolic verification mechanism
with inference-faithful evidence traces and selective intervention**.

## 2. Architecture Overview

For a query study \((x,q)\), the framework performs the following operations:

1. A frozen medical image encoder retrieves visually similar studies from the
   training partition.
2. A drafting backend generates an initial Findings section.
3. The draft is segmented into clinical claims and normalized to PrimeKG
   entities with assertion polarity.
4. Claims with strong multi-study retrieval consensus follow a low-cost fast
   path.
5. Remaining linked claims are escalated to a compact PrimeKG subgraph and
   LTN-style fuzzy verification.
6. The Consistency Gate accepts, revises, or flags each linked claim; claims
   without a reliable entity link receive an abstention decision.
7. Accepted claims are preserved. A disputed claim is revised only when a
   retrieved sentence satisfies a strict entity-and-polarity contract.
8. The final report and the evidence used for every decision are exported.

The drafting interface supports two families of models. The first is a trained
vision-language generator with a frozen visual backbone and a learned
image-to-language projection. The second uses MedGemma without task-specific
gradient updates in our pipeline. In this configuration, MedSigLIP supplies the
retrieval representation and MedGemma supplies the draft. This separation makes
the verification method independent of a particular report generator.

## 3. Leakage-Controlled Visual Retrieval

### 3.1 Frozen image representation

Let \(f_{\mathrm{img}}\) be a frozen medical image encoder. The query and each
training image are represented as

\[
\mathbf{v}_x=f_{\mathrm{img}}(x), \qquad
\mathbf{v}_i=f_{\mathrm{img}}(x_i).
\]

The training embeddings are computed once, normalized, and persisted as a
reusable index. Similarity is measured by cosine similarity,

\[
s_i = \cos(\mathbf{v}_x,\mathbf{v}_i)
    = \frac{\mathbf{v}_x^\top\mathbf{v}_i}
    {\lVert\mathbf{v}_x\rVert_2\lVert\mathbf{v}_i\rVert_2}.
\]

The retrieval set \(R_k(x)\) contains the reports associated with the \(k\)
highest-scoring eligible training studies. Retrieval is study-disjoint: the
query study and its alternate views are excluded using the underlying study
identifier. Only the training partition is indexed. The query reference is not
read during retrieval, prompting, generation, candidate selection, or
verification.

### 3.2 Interpretation of retrieval evidence

Retrieved reports are treated as non-authoritative evidence. They can inform
reporting style and provide support for recurring clinical entities, but their
findings are not assumed to apply to the query image. We retain the retrieved
study identifiers, ranks, and similarities for subsequent audits. This design
also allows an unusually high lexical score to be investigated for duplicated
or near-duplicated training reports.

## 4. Switchable Report Drafting

### 4.1 Trained vision-language backend

The checkpoint-based baseline maps spatial image tokens into the hidden space
of an encoder-decoder language model. Its visual backbone is frozen in the main
configuration, while the projection and language components are optimized on
training reports. Retrieval evidence may be concatenated with the projected
visual sequence before decoding. This backend provides a dataset-adapted
comparison for the configuration without task-specific MedGemma fine-tuning.

### 4.2 MedGemma backend

The MedGemma backend directly receives the current radiograph. In zero-shot
mode, the draft is

\[
y_{\mathrm{raw}}=G(x,q).
\]

In retrieval-conditioned few-shot mode,

\[
y_{\mathrm{raw}}=G\bigl(x,q,R_k(x)\bigr).
\]

The instruction requests only the Findings section, explicitly requires
preservation of negation and laterality, identifies retrieved reports as
non-authoritative examples, and prohibits copying unsupported findings. The
model is decoded deterministically in the primary configuration to improve
reproducibility.

No task-specific gradient update is performed for this backend. This wording is
important for MIMIC-CXR: the official MedGemma and MedSigLIP documentation lists
MIMIC-CXR among their pretraining sources. Consequently, MIMIC experiments are
described as **no task-specific fine-tuning**, rather than strict unseen-data
zero-shot evaluation.

We additionally evaluate a clearly separated task-adapted drafting ablation.
It applies 4-bit QLoRA supervised fine-tuning to the MedGemma decoder while
freezing the medical vision encoder; an optional experiment also adapts the
multimodal connector. Training targets and any retrieval demonstrations come
only from the training split, checkpoint selection uses the validation split,
and the test split is reserved for final evaluation. Results from this backend
are labelled **task-specific fine-tuned** and are not pooled with the
no-task-specific-fine-tuning setting.

## 5. Clinical Claim Extraction and Entity Linking

The draft is segmented into claims,

\[
y_{\mathrm{raw}}=\{c_1,c_2,\ldots,c_M\}.
\]

For each claim \(c_j\), the entity-linking component produces a set

\[
L_j=\{(m_{jr},e_{jr},t_{jr},a_{jr},p_{jr})\}_{r=1}^{N_j},
\]

where \(m_{jr}\) is a mention, \(e_{jr}\) a PrimeKG node identifier,
\(t_{jr}\) its semantic type, \(a_{jr}\in\{\text{positive},\text{negated}\}\)
its assertion polarity, and \(p_{jr}\) the linker confidence. The implemented
deterministic linker supports reproducible audits and may be augmented by
RadGraph-based mention extraction before PrimeKG normalization.

Entity-linking performance is evaluated independently of report-generation
quality. Required analyses include mention detection precision and recall,
node-normalization accuracy, assertion/negation accuracy, linked-claim coverage,
condition-specific coverage, and manual categorization of common errors. A
claim with \(N_j=0\) is assigned **ABSTAIN**; it is not counted as verified.

## 6. Claim-Level Evidence Contract

Each claim is associated with the evidence contract

\[
E(c_j)=\left[
s_{\mathrm{vis}},
s_{\mathrm{ret}}(c_j),
n_{\mathrm{sup}}(c_j),
s_{\mathrm{KG}}(c_j),
s_{\mathrm{LTN}}(c_j),
s_{\mathrm{gate}}(c_j)
\right].
\]

Here, \(s_{\mathrm{vis}}\) is the report-level similarity between the query and
its strongest visual neighbour. It indicates that relevant image-level
retrieval evidence exists, but it is not by itself claim grounding.
Entity-specific retrieval support is defined by matching the PrimeKG identifier
and assertion polarity of a claim entity against entities extracted from the
retrieved reports. We define

\[
s_{\mathrm{ret}}(c_j)=
\max_{r\in R_k(x):\,L(r)\cap L_j\ne\varnothing}s(x,r),
\]

and let \(n_{\mathrm{sup}}(c_j)\) be the number of distinct retrieved studies
that support at least one matching entity-polarity pair. The remaining terms
record PrimeKG reachability/status, aggregate fuzzy-logic satisfaction, and the
Consistency Gate confidence.

This distinction prevents the presence of a visually similar neighbour from
being incorrectly reported as evidence for every sentence in the generated
report. In evaluation, report-level evidence availability and entity-specific
claim-grounding coverage are therefore reported separately.

## 7. Adaptive Verification Router

The router uses retrieval consensus to decide whether neuro-symbolic reasoning
is necessary. Define

\[
s_{\mathrm{ground}}(c_j)=
\max\{s_{\mathrm{vis}},s_{\mathrm{ret}}(c_j)\}.
\]

A linked claim follows the fast path if

\[
s_{\mathrm{ground}}(c_j)\ge\tau_{\mathrm{fast}}
\quad\land\quad
n_{\mathrm{sup}}(c_j)\ge m.
\]

The current development configuration uses
\(\tau_{\mathrm{fast}}=0.85\) and \(m=2\). The support-count condition is
essential because a high report-level similarity alone does not establish
entity-specific consensus. Claims satisfying both conditions receive
**ACCEPT-FAST-PATH**. Other linked claims are escalated to PrimeKG/LTN
verification.

We report the escalation rate over all extracted claims,

\[
\rho_{\mathrm{all}}=
\frac{\sum_j\mathbb{1}[c_j\text{ escalated}]}{M},
\]

and over linked claims,

\[
\rho_{\mathrm{linked}}=
\frac{\sum_j\mathbb{1}[c_j\text{ escalated}]}
{\sum_j\mathbb{1}[N_j>0]}.
\]

The latter distinguishes computational savings from abstentions caused by
linking failures.

## 8. PrimeKG Subgraph Construction

For each escalated claim, a compact graph \(\mathcal{G}_j\) is constructed from
the union of claim entities and entities extracted from the clinical indication.
Entities under negated assertions remain graph terminals because their
biomedical type and connectivity must still be checked; their polarity is
retained separately for retrieval matching and the Consistency Gate. The hybrid
graph builder adds relevant one-hop neighbours and connecting paths, subject to
explicit neighbour and path-expansion budgets. The radiology-focused PrimeKG
cache is seeded only from training-split report entities in the primary
configuration, preventing test reports from shaping the available graph.

The subgraph retains node identifiers, semantic types, relation labels, source
provenance, and confidence attributes. When a connecting path exists, its nodes,
edge directions, relations, provenance, and confidences are stored in the
explanation trace. A missing path is represented
explicitly rather than being interpreted as proof that the claim is false.

## 9. LTN-Style Fuzzy Verification

Adaptive NeSy-Gen evaluates three fuzzy clause families on
\(\mathcal{G}_j\):

1. **Biological/temporal compatibility** checks whether connected node types and
   temporal attributes are compatible with the implemented radiological
   relation rules.
2. **Finding-to-diagnosis support** checks whether phenotype/finding nodes are
   connected to an appropriate disease or diagnosis node.
3. **Location/type compatibility** checks whether `located_in` relations connect
   a finding to an anatomical node of the expected type.

Their truth values are denoted by
\(\tau_{\mathrm{bio}}(c_j)\),
\(\tau_{\mathrm{diag}}(c_j)\), and
\(\tau_{\mathrm{loc}}(c_j)\). The aggregate score is

\[
\tau(c_j)=\frac{1}{3}\left[
\tau_{\mathrm{bio}}(c_j)+
\tau_{\mathrm{diag}}(c_j)+
\tau_{\mathrm{loc}}(c_j)
\right].
\]

Relation confidences and source reliability weight the fuzzy aggregation where
available. Node states are assigned using acceptance and flagging thresholds
\(\beta\) and \(\gamma\): valid when the local score is at least \(\beta\),
flagged when it lies in \([\gamma,\beta)\), and rejected otherwise. Importantly,
\(\tau(c_j)\) measures satisfaction of the implemented constraints; it is not a
calibrated probability that a radiological assertion is clinically true.

## 10. Consistency Gate

For each linked entity, the Consistency Gate consumes its PrimeKG node state,
grounding score, LTN satisfaction, assertion polarity, and configured risk
thresholds. The claim-level action is

\[
g(c_j)\in
\{\text{ACCEPT},\text{REVISE},\text{FLAG},\text{ABSTAIN}\}.
\]

An escalated claim is accepted when all linked entities pass the gate and the
minimum of grounding and LTN satisfaction exceeds the revision threshold.
Disputed linked claims are considered for evidence-bound revision. If no valid
replacement exists, the original claim is preserved but flagged. Unlinked
claims receive ABSTAIN. Every gate output includes the decision reason and
confidence used by the pipeline.

The system deliberately distinguishes FLAG from deletion. PrimeKG incompleteness,
entity-linking error, and retrieval mismatch can all cause low satisfaction;
silently removing the sentence would therefore overstate the verifier's
authority.

## 11. Evidence-Bound Selective Revision

Adaptive NeSy-Gen does not regenerate the complete report after verification.
Accepted claims are immutable. For a disputed claim \(c_j\), a retrieved
sentence \(c'_j\) is eligible only if

\[
\mathcal{E}(c'_j)=\mathcal{E}(c_j),
\qquad
\mathcal{A}(c'_j)=\mathcal{A}(c_j),
\qquad
s_{\mathrm{ret}}(c'_j)\ge\tau_{\mathrm{rev}},
\]

where \(\mathcal{E}\) is the set of linked PrimeKG identifiers and
\(\mathcal{A}\) is the corresponding assertion-polarity assignment. Thus, a
replacement cannot add a clinical entity, reverse negation, or change the set
of normalized findings. Among eligible sentences, selection prioritizes entity
agreement, visual-retrieval score, and concision. The retrieved study identifier
is recorded as replacement provenance.

This operation is best understood as evidence-bound surface revision rather
than unconstrained clinical correction. If no eligible sentence exists, the
claim remains unchanged and is marked FLAG. This conservative behavior protects
lexical quality while limiting the possibility that the revision stage
introduces a new unsupported finding.

## 12. Inference-Faithful Explanation Trace

For each claim, the system stores:

- original and final claim text;
- linked mention, PrimeKG identifier, node type, polarity, and linker confidence;
- report-level visual similarity;
- entity-specific retrieval support and supporting-study count;
- whether graph verification was invoked;
- all LTN clause values and their aggregate;
- PrimeKG node state and an available graph path;
- gate confidence, action, and reason;
- replacement source, when applicable; and
- claim-verification latency.

These traces are **procedurally faithful**: their recorded scores and decisions
are values that the router, verifier, or gate actually consumed during
inference. They are not natural-language explanations generated after the
decision. We make the narrower procedural-faithfulness claim rather than
asserting that the trace is a complete causal explanation of the internal neural
representations of the drafting model.

## 13. Inference Algorithm

```text
Algorithm 1: Adaptive NeSy-Gen inference
Input: query image x, indication q, training index I, PrimeKG cache K
Output: final report y_hat, claim traces T

1:  v_x <- frozen_image_encoder(x)
2:  R_k <- retrieve_top_k(v_x, I), excluding the query study and alternate views
3:  y_raw <- drafting_model(x, q, optional retrieved reports R_k)
4:  C <- segment_into_claims(y_raw)
5:  T <- empty list; C_final <- empty list
6:  for each claim c in C do
7:      L <- link_entities_and_assertions(c, K)
8:      if L is empty then
9:          append trace(c, ABSTAIN, reason=unlinked) to T
10:         append c to C_final
11:         continue
12:     end if
13:     compute s_visual, s_retrieval(c), and n_support(c)
14:     if max(s_visual, s_retrieval(c)) >= tau_fast
            and n_support(c) >= m then
15:         append trace(c, ACCEPT_FAST_PATH) to T
16:         append c to C_final
17:         continue
18:     end if
19:     G_c <- build_compact_subgraph(L, entities(q), K)
20:     tau <- evaluate_fuzzy_clauses(G_c)
21:     d <- consistency_gate(L, retrieval evidence, tau)
22:     if d = ACCEPT then
23:         c_final <- c
24:     else
25:         c_prime <- find_evidence_bound_replacement(c, R_k)
26:         if c_prime exists then d <- REVISE; c_final <- c_prime
27:         else d <- FLAG; c_final <- c
28:         end if
29:     end if
30:     append trace(c, c_final, evidence, G_c, tau, d) to T
31:     append c_final to C_final
32: end for
33: y_hat <- concatenate(C_final)
34: return y_hat, T
```

## 14. Computational Complexity and Efficiency

Let \(N\) be the number of indexed training studies, \(d\) the embedding
dimension, \(M\) the number of draft claims, \(U\le M\) the number of escalated
claims, \(|V_j|\) and \(|E_j|\) the size of the subgraph for escalated claim
\(j\), and \(C_G\) the cost of drafting-model inference.

Training-index construction costs \(O(NC_f)\), where \(C_f\) is one image
encoding, and is amortized because the resulting index is cached. Exact query
retrieval costs \(O(Nd)\); approximate nearest-neighbour indexing can replace
this term for larger collections. Draft generation costs \(C_G\). Lexical
entity linking is approximately linear in report length under the indexed alias
matcher. Adaptive graph processing costs

\[
O\!\left(\sum_{j=1}^{U}(|V_j|+|E_j|)\right),
\]

apart from bounded path search. An always-on claim verifier substitutes \(M\)
for \(U\). The adaptive saving is therefore directly related to
\(1-U/M\), while \(U/M\) is reported rather than inferred from latency alone.

We separately measure one-time index construction, index size, retrieval
latency, drafting latency, claim-verification latency, end-to-end latency, peak
GPU memory, graph calls per report, and escalation rates. Claim-verification
latency must not be presented as end-to-end runtime.

## 15. Evaluation Protocol

### 15.1 Report quality

Lexical evaluation includes BLEU-1--4, ROUGE-L, METEOR, and CIDEr. Clinical
evaluation prioritizes official CheXbert/CheXpert-style labels and official
RadGraph F1, supplemented by entity precision, recall, F1, and negation
consistency. Lightweight approximations are used only for development and are
not substituted for official metrics in the main results.

### 15.2 Explainability and adaptive behavior

We report linked-claim coverage, entity-specific retrieval-grounding coverage,
\(\rho_{\mathrm{all}}\), \(\rho_{\mathrm{linked}}\), graph-path coverage among
escalated claims, gate-action frequencies, revision rate, trace-field
completeness, per-claim latency, and end-to-end runtime. Entity-linking coverage
is reported alongside gate results because a high abstention rate can otherwise
make the verifier appear artificially selective.

### 15.3 Integrity checks

All runs include exact-match and high-overlap audits, same-study retrieval
exclusion, unique-prediction ratio, maximum prediction frequency, and distinct
n-gram statistics. These checks identify leakage, duplicated references, and
template collapse. High overlap is treated as a signal for manual investigation,
not automatically as proof of leakage.

## 16. Ablation Studies

The evaluation isolates the following components:

1. retrieval-only reporting;
2. the raw trained generator;
3. MedGemma zero-shot drafting;
4. MedGemma retrieval-conditioned few-shot drafting;
5. QLoRA-adapted MedGemma with image-only versus mixed retrieval-conditioned SFT;
6. RAG without PrimeKG/LTN;
7. report-level PrimeKG/LTN verification;
8. adaptive claim auditing without revision;
9. full adaptive verification and evidence-bound revision;
10. always-on claim verification;
11. adaptive verification without LTN clause scoring;
12. adaptive verification without the Consistency Gate;
13. shuffled-edge or relation-ablated PrimeKG controls; and
14. sensitivity to \(\tau_{\mathrm{fast}}\), \(m\), and
    \(\tau_{\mathrm{rev}}\).

The adaptive and always-on variants use identical drafting and retrieval
outputs, ensuring that differences in quality or runtime can be attributed to
the routing policy. The graph controls test whether performance depends on
meaningful PrimeKG structure rather than merely the presence of additional
computation.

## 17. Assumptions and Limitations

Adaptive NeSy-Gen assumes that visual neighbours offer useful but imperfect
evidence and that clinically important report concepts can be mapped to
PrimeKG. These assumptions may fail for rare findings, unusual language,
out-of-distribution acquisition protocols, or concepts absent from the graph.
The deterministic linker improves reproducibility but may have limited recall;
unlinked claims are therefore abstained rather than certified. PrimeKG
reachability and LTN clause satisfaction encode compatibility with the
implemented knowledge rules, not direct image proof. Likewise, evidence-bound
revision can improve stylistic alignment but cannot establish clinical truth.

MedGemma and MedSigLIP are pretrained systems whose data provenance constrains
the interpretation of zero-shot experiments. Their outputs remain susceptible
to omission and unsupported statements. The framework consequently does not
claim to eliminate hallucination. Unsupported-entity rates and clinical-label
disagreement are treated as measurable proxies, and any stronger factuality
claim requires official clinical metrics, statistical testing, and preferably
expert radiologist assessment.
