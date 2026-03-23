# Algorithm Draft: PoLL + MQM + Objective + Term Gate

## 1. Final Score

For each sample:

$$S_{final}=\max\left(0,\left[S_{mqm}-P_{obj}\right]\times\left(1-\delta\cdot E_{term}\right)\right)$$

Default parameters:
- $\delta=0.5$
- $\omega_1=0.4$, $\omega_2=0.6$
- Threshold $=60$
- $\alpha=0.2$

## 2. PoLL Arbitration (7 Gemini Slots)

### 2.1 Judge Output Contract
Each judge returns JSON:

```json
{
  "errors": [
    {
      "span": "text span",
      "severity": "minor|major|critical",
      "category": "accuracy|fluency|terminology|style|other",
      "reason": "short rationale"
    }
  ]
}
```

### 2.2 Clustering
Two errors are merged if:
- category matches, and
- normalized span overlap ratio >= overlap threshold (default 0.5).

### 2.3 Voting
- Accept cluster if votes >= `vote_threshold`.
- Full mode default: 4/7.
- Smoke mode can use fewer judges and auto-majority.

### 2.4 Severity Resolution
For accepted cluster:
- Convert severities to weights (`minor=1`, `major=5`, `critical=25`).
- Pick median weight among voters.
- Map back to severity label.

## 3. MQM Score

Let accepted counts be $c_{minor}, c_{major}, c_{critical}$.

$$Penalty=c_{minor}\cdot1+c_{major}\cdot5+c_{critical}\cdot25$$

Length unit $L$ (v1):
- tokenize source and hypothesis,
- use $L=\max(1,\max(L_{src},L_{hyp}))$.

Then:

$$S_{mqm}=\max\left(0,100-\frac{Penalty}{L}\times100\right)$$

## 4. Objective Penalty

$$P_{obj}=\alpha\cdot\max\left(0,Threshold-(\omega_1\cdot chrF++ + \omega_2\cdot COMET)\right)$$

No-reference fallback (v1 default):
- `chrF=65`, `COMET=65`

## 5. Terminology Gate

Expected terms hit ratio:

$$hit\_rate=\frac{N_{hit}}{\max(1,N_{term})}$$

Term error:
- fatal forbidden hit => $E_{term}=1$
- else $E_{term}=1-hit\_rate$

## 6. Smoke Mode (for Pipeline Validation)

Purpose: validate reliability and output schema, not final model ranking.

Default smoke policy:
- Judges: first 3 slots (Gemini)
- Repeat per judge: 2
- Sample cap per block: configurable (`--smoke-max-items`)

## 7. Diagnostics

Per row diagnostics:
- valid_judges
- accepted_clusters
- rejected_clusters
- severity histogram
- arbitration confidence (`accepted_votes/possible_votes`)

Per block diagnostics:
- avg valid judge count
- row failure ratio
- vote pass ratio
- variance of `S_final`
