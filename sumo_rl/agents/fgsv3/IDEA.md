# FGSv3 — Design Proposal

> Status: **design draft** — implement and validate before incorporating into thesis.
> Based on: `fgsv2.md`, thesis Methodology chapter, experimental results (W&B project `marl-traffic-gat`).

---

## 1. What FGSv2 Actually Does (and Where It Goes Wrong)

### 1.1 FRAP feature hierarchy

FRAP computes features in a strict pipeline. Each level is richer but more local than the last:

| Level | Tensor | Shape | Semantics |
|---|---|---|---|
| A | `e_{i,u}` | `[B·N, M, C]` | Movement demand embedding (density + queue per lane group) |
| B | `φ_{i,a}` | `[B·N, A, C]` | Phase demand embedding (sum of movement embeddings for movements in phase `a`) |
| C | `c_{i,a,b}` | `[B·N, A, A-1, C]` | Phase competition feature (phase `a` vs competitor `b`; includes relation type) |
| D | `z_{i,a}` | `[B, N, A, D]` | Action token (mean-pool of `c` over valid competitors + adapter) |
| E | `s_i` | `[B, N, D]` | Node summary (mean-pool of `z` over valid actions) |

### 1.2 What FGSv2 communicates

FGSv2 sends **Level E** (`s_i`) into GATv2. This is a mean of action tokens, which are themselves a mean of competition features. That is:

```
s_i = MeanPool_a( Adapter( MeanPool_{b}( c_{i,a,b} ) ) )
```

It has passed through **two lossy pooling stages** before reaching the graph layer.

### 1.3 The semantic mismatch

Competition features (Level C–E) answer the question:
> "Given my current traffic state, which of my phases is the most dominant?"

This is an **internal decision signal**. It only makes sense in the context of intersection `i`'s own phase structure.

What a neighboring intersection `j` actually needs to know is:
> "How congested is intersection `i` right now, and which movements is it currently serving?"

That information lives at **Level A–B** (demand embeddings), not Levels C–E.

**Root cause of FRAP underperformance in FGSv2**: GATv2 receives competition-distilled preference signals instead of raw demand signals. Neighbors see "which phase is winning" at `i`, not "how much traffic is waiting at `i`". This makes graph aggregation less useful for actual coordination.

### 1.4 Evidence from experiments

| Variant | Cologne Regional wait (s) | Ingolstadt Regional wait (s) |
|---|---|---|
| FGS(MLP+GATv2+SAC) | **4.9** | — |
| FGS(FRAP+GATv2+PPO) | 6.7 | 460.7 |
| FGS(FRAP+GAT+SAC) | 87.6 | — |
| FGS(MLP+GATv2+PPO) | 6.3 | **88.6** |

The MLP encoder communicates raw node features (Level A equivalent). FRAP encoder communicates doubly-pooled competition signals (Level E). The MLP consistently outperforms FRAP+GATv2 despite being simpler — consistent with the hypothesis that Level E carries poor inter-agent information.

---

## 2. Design Principles for FGSv3

1. **Separate communication features from decision features.** FRAP competition features are appropriate for local phase selection; demand features are appropriate for inter-agent communication.
2. **Communicate phase intent alongside demand.** A neighbor's current phase tells downstream agents which directions are currently green (and thus which vehicle queues will arrive soon).
3. **Reduce critic scaling.** The centralized critic's O(N·D) input is the bottleneck at N=21. Factoring it to O(deg·D) enables larger networks.
4. **Preserve FRAP's local inductive bias.** FRAP still produces action tokens for the actor; the structural competition prior is kept where it is most useful (local decision-making).

---

## 3. FGSv3 Architecture

### 3.1 High-level diagram

```
OBS (node features, masks, prev_joint_action, graph topology)
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  FRAP Encoder (shared weights across N)             │
│                                                     │
│  Level A: e_{i,u}  [B·N, M, C]  movement demand    │
│     │                                               │
│  Level B: φ_{i,a}  [B·N, A, C]  phase demand       │
│     │         │                                     │
│     │    Level C: c_{i,a,b}  competition features  │
│     │         │                                     │
│     │    Level D: z_{i,a}  action tokens [B,N,A,D] │
│     │                                               │
│     ▼  (NEW tap point for communication)            │
│  d_i = DemandEncoder( φ_{i,a}, prev_action_i )     │
│      = LayerNorm( Adapter( [MeanPool_a(φ_{i,a}),   │
│                             Emb_phase(prev_act_i) ] ))│
│      → [B, N, D_comm]                              │
└─────────────────────────────────────────────────────┘
      │                           │
      │ d_i (comm. signal)        │ z_{i,a} (action tokens)
      ▼                           │
┌─────────────────┐               │
│  GATv2          │               │
│  (on d_i only)  │               │
│  → g_i [B,N,D]  │               │
└─────────────────┘               │
      │                           │
      └──────────┬────────────────┘
                 ▼
        ┌────────────────────────────────────┐
        │  Action-conditioned Actor          │
        │  ℓ_{i,a} = f_π([z_{i,a}, g_i,     │
        │                  one_hot(a)])       │
        │  → masked softmax → π_i(a)         │
        └────────────────────────────────────┘

        ┌────────────────────────────────────┐
        │  Factored Neighborhood Critic      │
        │  Q_k(a) = f_Q([z_{k,a}, g_k,      │
        │    {g_j | j∈N(k)},                 │
        │    {a_{t-1,j} | j∈N(k)},           │
        │    one_hot(a), one_hot(k)])         │
        │  → twin Q-values                   │
        └────────────────────────────────────┘
```

### 3.2 New: Demand communication branch

Instead of node summary `s_i` (pooled competition features), FGSv3 constructs the communication signal from Level B (phase demand) and the previous action:

```
d_i = LayerNorm( ReLU( W_comm · [ MeanPool_{a∈valid}(φ_{i,a}) ‖ Emb_phase(prev_action_i) ] ) )
```

where:
- `MeanPool_{a∈valid}(φ_{i,a})` ∈ ℝ^C — average demand across all valid phases at `i`
- `Emb_phase(prev_action_i)` ∈ ℝ^D_pe — learned phase-index embedding of the previous phase choice
- `W_comm` ∈ ℝ^(D_comm × (C + D_pe)) — linear projection
- `d_i` ∈ ℝ^(D_comm)

**What this carries:**
- How much traffic demand is present at intersection `i` (from `φ`)
- Which direction intersection `i` is currently serving (from `prev_action`)

This is exactly the information a downstream neighbor needs: "is my upstream congested, and is it currently clearing the vehicles heading toward me?"

### 3.3 GATv2 operates on `d_i`

Identical to FGSv2 except the input is `d_i` instead of `s_i`:

```
u_i = LayerNorm(d_i)
g̃_i = ReLU( W_o · GATv2Conv( {u_j | j → i ∈ E ∪ {i}} ) )
g_i = LayerNorm( d_i + λ · g̃_i )
```

`λ` is a learned scalar initialized to 0 (same residual gate as FGSv2).

### 3.4 Actor (unchanged from FGSv2)

```
ℓ_{i,a} = f_π( [z_{i,a} ‖ g_i ‖ one_hot(a)] )
ℓ_{i,a} = −10^9   when m_{i,a} = 0
π_i(a | o) = softmax_a( ℓ_{i,a} )
```

The actor still receives action tokens from the FRAP competition path. FRAP's local competition prior is fully preserved for decision-making.

### 3.5 New: Factored neighborhood critic

FGSv2's centralized critic takes all N graph contexts as input:

```
Q_k(a) = f_Q([ vec(g_1,…,g_N), vec(J_k^(a)), z_{k,a}, g_k, one_hot(k) ])
```

Input dimension: `O(N·D + N·A)` — infeasible at N=21.

FGSv3 replaces this with a **neighborhood-factored critic** that only aggregates from direct neighbors:

```
N_k = direct neighbors of k in the communication graph
Q_k(a) = f_Q([
    z_{k,a},            # ego action token (FRAP competition features)
    g_k,                # ego graph context (demand + GATv2 aggregation)
    Σ_{j∈N_k} g_j,     # neighbor graph contexts (sum-pooled)
    Σ_{j∈N_k} a_{t-1,j} ⊗ w_{kj},   # neighbor actions, weighted by edge weight
    one_hot(a),         # candidate action identity
    one_hot(k)          # ego identity (only needed in non-shared-parameter critic)
])
```

where `⊗ w_{kj}` weights each neighbor's one-hot action by the edge weight (travel time inverse).

**Input dimension**: `O(D + deg·D + deg·A)` — independent of total N, bounded by graph degree.

For ingolstadt21 with max degree ~4, this reduces critic input from ~3,091 to ~200 dimensions (using D=64, A=6, deg=4).

---

## 4. Mathematical Formulation

### 4.1 Demand communication embedding

For intersection `i` with valid actions `V_i = {a | m_{i,a} = 1}`:

$$
\bar{\boldsymbol{\phi}}_i = \frac{1}{|V_i|} \sum_{a \in V_i} \boldsymbol{\phi}_{i,a}
\quad \in \mathbb{R}^C
$$

$$
\mathbf{d}_i = \operatorname{LayerNorm}\!\left(
  \operatorname{ReLU}\!\left(
    W_{\text{comm}} \left[ \bar{\boldsymbol{\phi}}_i \,\Vert\, \operatorname{Emb}_{\text{phase}}(a_{t-1,i}) \right]
  \right)
\right)
\quad \in \mathbb{R}^{D_{\text{comm}}}
$$

### 4.2 GATv2 communication on demand features

$$
u_i = \operatorname{LayerNorm}(\mathbf{d}_i)
$$

$$
e_{ij} = \mathbf{a}^\top \operatorname{LeakyReLU}\!\left(W_\ell u_i + W_r u_j\right) + \log w_{ij}
$$

$$
\alpha_{ij} = \operatorname{softmax}_{j \in \mathcal{N}(i) \cup \{i\}}\!(e_{ij})
$$

$$
\tilde{\mathbf{g}}_i = \operatorname{ReLU}\!\left(W_o \sum_{j \in \mathcal{N}(i) \cup \{i\}} \alpha_{ij} u_j\right)
$$

$$
\mathbf{g}_i = \operatorname{LayerNorm}\!\left(\mathbf{d}_i + \lambda \tilde{\mathbf{g}}_i\right)
$$

(edge weight `w_{ij}` biases attention toward closer intersections, consistent with FGSv2.)

### 4.3 Actor (unchanged)

$$
\ell_{i,a} = f_\pi\!\left([\mathbf{z}_{i,a} \,\Vert\, \mathbf{g}_i \,\Vert\, \mathbf{e}_a]\right)
$$

$$
\pi_i(a \mid o) = \operatorname{softmax}_{a}\!\left(\ell_{i,a} + (1 - m_{i,a})(-\infty)\right)
$$

### 4.4 Factored neighborhood critic

For ego node `k`, the critic Q-value for candidate ego action `a` is:

$$
\mathbf{h}_{\text{nbr},k} = \sum_{j \in \mathcal{N}(k)} w_{kj} \cdot \mathbf{g}_j
$$

$$
\mathbf{a}_{\text{nbr},k} = \sum_{j \in \mathcal{N}(k)} w_{kj} \cdot \mathbf{a}_{t-1,j}
$$

$$
Q_k(a) = f_Q\!\left([\mathbf{z}_{k,a} \,\Vert\, \mathbf{g}_k \,\Vert\, \mathbf{h}_{\text{nbr},k} \,\Vert\, \mathbf{a}_{\text{nbr},k} \,\Vert\, \mathbf{e}_a \,\Vert\, \mathbf{e}_k]\right)
$$

Twin critics maintained; minimum used for target and actor loss.

---

## 5. Changes Relative to FGSv2

| Component | FGSv2 | FGSv3 | Reason |
|---|---|---|---|
| Communication signal | `s_i` = mean of action tokens (competition features) | `d_i` = demand summary + phase intent | Demand is what neighbors need; competition is internal |
| GATv2 input | Competition-derived node summary | Demand embedding + previous-phase embedding | Richer, semantically appropriate signal |
| FRAP tap point | Level C→D→E (competition → tokens → summary) | UNCHANGED for action tokens; NEW branch at Level B for communication | Keep FRAP local prior; fix communication path |
| Action tokens | Used in actor | UNCHANGED | FRAP competition prior still useful locally |
| Critic input | All N graph contexts (O(N·D)) | Ego + direct neighbors only (O(deg·D)) | Scalability; removes O(N) bottleneck |
| Critic joint-action context | Full N×A joint action matrix | Neighbor-weighted action sum (O(deg·A)) | Consistent with factored critic |
| Residual gate λ | Scalar, init=0 | UNCHANGED | Stable early training |

---

## 6. Implementation Notes

### 6.1 New module: `DemandCommunicationBranch`

```python
class DemandCommunicationBranch(nn.Module):
    """
    Produces per-node communication embedding from FRAP Level B features
    (phase demand embeddings) and the previous action.
    
    Inputs:
        phi: [B*N, A, C]       -- phase demand embeddings (Level B from FRAP)
        action_mask: [B*N, A]  -- valid action mask
        prev_action: [B*N]     -- previous action index (long)
    
    Output:
        d: [B, N, D_comm]      -- communication embedding
    """
    def __init__(self, C, D_pe, D_comm, A_max):
        super().__init__()
        self.phase_emb = nn.Embedding(A_max + 1, D_pe, padding_idx=A_max)
        self.proj = nn.Linear(C + D_pe, D_comm)
        self.norm = nn.LayerNorm(D_comm)
    
    def forward(self, phi, action_mask, prev_action):
        # phi: [B*N, A, C], action_mask: [B*N, A]
        mask = action_mask.unsqueeze(-1).float()          # [B*N, A, 1]
        phi_valid = (phi * mask).sum(1) / mask.sum(1).clamp(min=1)  # [B*N, C]
        
        pe = self.phase_emb(prev_action)                  # [B*N, D_pe]
        x = torch.cat([phi_valid, pe], dim=-1)            # [B*N, C+D_pe]
        d = self.norm(F.relu(self.proj(x)))               # [B*N, D_comm]
        # reshape to [B, N, D_comm] (caller handles reshape)
        return d
```

### 6.2 FRAP tap point unchanged for action tokens

Keep the existing Level C → D path (competition features → action tokens `z_{i,a}`) exactly as in FGSv2. Only add the new `DemandCommunicationBranch` that taps Level B in parallel.

```
FRAP forward:
    e_{i,u} = demand_encoder(obs)         # Level A
    phi_{i,a} = phase_pooler(e, pair_mask) # Level B  ← NEW tap here
    h_{i,a,b} = phase_conv(phi)           # Level C part 1
    r_{i,a,b} = rel_conv(rel_emb(R))      # Level C part 2
    c_{i,a,b} = hidden_conv(h ⊙ r)       # Level C
    z_{i,a} = token_pool(c, comp_mask)    # Level D  ← existing tap
    
    # NEW branch:
    d_i = demand_comm_branch(phi, action_mask, prev_action)
```

### 6.3 Factored critic: neighbor pooling

The critic no longer receives `vec(g_1,...,g_N)`. It needs the ego's neighbor indices, which are available from the edge index in the observation. During batch construction:

```python
# For each ego k, pre-compute neighbor indices from edge_index
# edge_index: [2, |E|] where edge_index[1, e] = target, edge_index[0, e] = source
# Neighbors of k = {edge_index[0, e] for e where edge_index[1, e] == k}

def build_neighbor_context(g, edge_index, edge_weight, ego_idx, A_max, prev_joint_action):
    """
    g: [N, D]  -- graph contexts for all nodes
    Returns: h_nbr [D], a_nbr [A]
    """
    neighbors = edge_index[0, edge_index[1] == ego_idx]
    weights = edge_weight[edge_index[1] == ego_idx]
    weights = weights / weights.sum().clamp(min=1e-6)  # normalize
    
    h_nbr = (g[neighbors] * weights.unsqueeze(-1)).sum(0)  # [D]
    a_nbr = (prev_joint_action[neighbors].float() * weights.unsqueeze(-1)).sum(0)  # [A]
    return h_nbr, a_nbr
```

### 6.4 Config changes

```yaml
# model_config additions for FGSv3
encoder:
  type: fgsv3
  frap:
    # same as FGSv2
    hidden_channels: 64
    num_relations: 4
  communication:
    # NEW:
    demand_comm_dim: 64     # D_comm
    phase_emb_dim: 16       # D_pe
    residual_gate_init: 0.0 # same as FGSv2
critic:
  type: factored_neighborhood  # NEW
  # OLD: type: centralized_full
```

---

## 7. Expected Improvements and Risks

### 7.1 Expected improvements

| Scenario class | Expected effect | Reason |
|---|---|---|
| Cologne Corridor (N=3) | Moderate improvement over FGSv2+FRAP | Demand signal more useful at small N where phases are less heterogeneous |
| Cologne Regional (N=8) | Close to or better than MLP+GATv2+SAC | Factored critic + demand comm removes O(N) bottleneck; demand signal is accurate |
| Ingolstadt Regional (N=21) | SAC now runnable (factored critic ~200 dims vs 3,091) | Key blocker was critic dimensionality |
| Ingolstadt Corridor (N=7) | To be measured (no FGSv2 run available) | Should benefit from factored critic |

### 7.2 Risks

| Risk | Mitigation |
|---|---|
| Demand branch adds training instability | Residual gate λ=0 init means early training is purely local; demand signal only activated as λ grows |
| Factored critic loses global coordination signal | Factored critic only loses nodes >1 hop away; 2-hop GATv2 context already aggregates 2nd-order neighborhood |
| Phase embedding for `prev_action` is sparse at start of training | Initialize `Emb_phase` with small weights so early signal is low; competition tokens dominate initially |
| FRAP heterogeneous phase mismatch still present (German scenarios) | Same issue as FGSv2; this proposal doesn't fix FRAP's structural mismatch — see Section 8 |

---

## 8. Known Remaining Limitation: FRAP Structural Mismatch

FGSv3 fixes the **communication** part but does NOT fix the underlying FRAP structural mismatch with German intersection geometries. The phase-pair weight formula:

```
w_{i,a,u} = P_{i,a,u} · min(Σ_v P_{i,a,v}, 2) / max(Σ_v P_{i,a,v}, 1)
```

is calibrated for intersections where each phase controls exactly 2 movements. German RESCO intersections have phases controlling 1, 2, 4, or 6 movements. Phases with 1 movement produce undersaturated features; phases with 4+ movements produce over-weighted features.

**Possible FGSv4 direction** (out of scope for current experiments): replace the fixed phase-pair weight with a learned attention over movements:

```
φ_{i,a} = Σ_u  softmax_u( W_q φ_a · W_k e_u ) · W_v e_u
```

This gives full flexibility regardless of phase movement count.

---

## 9. Experiment Plan

Recommended runs to validate FGSv3:

| Run | Encoder | Graph | Critic | RL | Scenario | Purpose |
|---|---|---|---|---|---|---|
| A | FRAP | GATv2 (d_i) | Factored | SAC | cologne8 | Core FGSv3 vs FGSv2+FRAP |
| B | FRAP | GATv2 (d_i) | Factored | SAC | ingolstadt21 | Scalability fix |
| C | FRAP | GATv2 (d_i) | Factored | PPO | ingolstadt21 | PPO comparison |
| D | MLP  | GATv2 | Factored | SAC | cologne8 | Factored critic only (isolate) |
| E | FRAP | GATv2 (s_i) | Factored | SAC | cologne8 | Ablate comm signal (s_i vs d_i) |
| F | FRAP | GATv2 (d_i) | Full | SAC | cologne8 | Ablate critic (factored vs full) |

Run E and F are ablations to isolate the contribution of each change. If D ≈ FGS(MLP+GATv2+SAC), the factored critic doesn't hurt MLP performance. If A > E, the demand communication signal is the key driver. If A > F, the factored critic helps independently.

---

## 10. Relationship to Thesis Claims

FGSv3 directly addresses the thesis analysis in §7.5 (RQ2) and §7.3 (SAC convergence):

- **RQ2 (encoder and communication)**: FGSv3 shows that FRAP competition features are suboptimal for communication; demand features are better → this becomes a stronger analytical contribution than "FRAP just doesn't work."
- **RQ3 (scalability)**: FGSv3's factored critic enables SAC to complete training at N=21, providing a direct answer to the scalability limitation identified at ingolstadt21.
- **H3 (FRAP hypothesis)**: The demand branch result isolates *where* FRAP fails — not in local decision-making but in its inadvertent communication role via `s_i` → confirms H3 more precisely.

If FGSv3 with FRAP+GATv2(demand)+SAC outperforms FGSv2 with MLP+GATv2+SAC, it becomes the thesis's main architectural contribution.
