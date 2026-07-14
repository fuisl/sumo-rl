# Thesis Architecture Diagrams

This page collects Mermaid diagrams for the PPO variants used in this thesis
codebase, along with a detailed DCRNN block view and a GRU comparison.

The PPO+DCRNN diagrams below match the current implementation in:

- `sumo_rl/models/dcrnn.py`
- `sumo_rl/agents/ppo/rllib_module.py`
- `configs/algorithm/ppo_dcrnn_mlp.yaml`
- `configs/algorithm/ppo_dcrnn_shared_mlp.yaml`

For plain `ppo`, RLlib uses its standard feed-forward path rather than a
custom module in this repository, so the diagram below is a thesis-style
conceptual MLP encoder plus MLP actor/value heads.

## PPO

```mermaid
flowchart LR
    O["Observation for one agent"] --> E["Encoder MLP"]
    E --> P["Action Head MLP"]
    E --> V["Value Head MLP"]
    P --> A["Action logits / policy distribution"]
    V --> R["State value"]
```

## PPO + Independent DCRNN + MLP

This matches `ppo_dcrnn_mlp`. The node-wise MLP pre-encoder feeds two paths:
one path enters the DCRNN encoder, while the latest-step MLP features also skip
directly to the final concatenation. The final ego latent is:

`latent_i = concat(dcrnn_hidden_i, latest_mlp_features_i)`

```mermaid
flowchart LR
    O["Graph-history observation<br/>B x H x N x F"] --> M["Node-wise Encoder MLP"]

    M --> D["DCRNN Block"]
    M --> L["Latest-step MLP features"]

    D --> EH["Ego node hidden state"]
    L --> EF["Ego node latest features"]

    EH --> C["Concat"]
    EF --> C

    C --> P["Action Head MLP"]
    C --> V["Value Head MLP"]

    P --> A["Action logits / policy distribution"]
    V --> R["State value"]
```

## PPO + Shared DCRNN + MLP

This matches `ppo_dcrnn_shared_mlp`. Multiple agents keep separate policy/value
heads, but all of them reuse one shared encoder.

```mermaid
flowchart LR
    X["Shared graph-history observation<br/>B x H x N x F"] --> M["Shared Node-wise Encoder MLP"]
    M --> D["Shared DCRNN Block"]
    M --> L["Shared latest-step MLP features"]

    D --> S0["Select agent 0 hidden"]
    L --> F0["Select agent 0 latest features"]
    S0 --> C0["Concat for agent 0"]
    F0 --> C0
    C0 --> P0["Agent 0 Action Head MLP"]
    C0 --> V0["Agent 0 Value Head MLP"]

    D --> S1["Select agent 1 hidden"]
    L --> F1["Select agent 1 latest features"]
    S1 --> C1["Concat for agent 1"]
    F1 --> C1
    C1 --> P1["Agent 1 Action Head MLP"]
    C1 --> V1["Agent 1 Value Head MLP"]

    D --> SK["Select agent k hidden"]
    L --> FK["Select agent k latest features"]
    SK --> CK["Concat for agent k"]
    FK --> CK
    CK --> PK["Agent k Action Head MLP"]
    CK --> VK["Agent k Value Head MLP"]
```

## DCRNN Block Detail

This reflects the current `DCRNNBackbone` and `DCGRUCell` flow:

- optional node-wise MLP pre-encoder
- stacked history processed by a DCGRU
- diffusion graph convolution inside both gate and candidate computations
- final ego latent formed by concatenating DCRNN hidden state and latest MLP
  features

```mermaid
flowchart TD
    O["Input graph history<br/>B x H x N x F"] --> PE["Optional node-wise MLP pre-encoder"]
    PE --> T["Time-major sequence<br/>H steps"]

    T --> X1["Step t input x_t"]
    H0["Previous hidden state h_t-1"] --> G1["Gate diffusion graph conv"]
    X1 --> G1
    G1 --> G2["Sigmoid"]
    G2 --> R["Reset gate r_t"]
    G2 --> U["Update gate u_t"]

    H0 --> RM["r_t element-wise multiply h_t-1"]
    R --> RM
    X1 --> C1["Candidate diffusion graph conv"]
    RM --> C1
    C1 --> C2["Tanh"]
    C2 --> HC["Candidate state h_hat_t"]

    U --> N1["u_t element-wise multiply h_t-1"]
    H0 --> N1
    U --> N2["1 - u_t"]
    HC --> N3["(1 - u_t) element-wise multiply h_hat_t"]
    N2 --> N3
    N1 --> SUM["Add"]
    N3 --> SUM
    SUM --> H1["New hidden state h_t"]

    H1 --> LOOP["Repeat for next time step"]
    LOOP --> X1

    H1 --> FINAL["Final node embeddings E at last step"]
    PE --> LF["Latest-step encoded node features"]
    FINAL --> EH["Select ego node embedding"]
    LF --> EF["Select ego latest features"]
    EH --> CAT["Concat"]
    EF --> CAT
    CAT --> Z["Final latent z_i"]
```

## Diffusion Layer Inside DCGRU

This zooms in on the diffusion graph convolution used by the gate and candidate
branches.

```mermaid
flowchart LR
    X["Node features x_t"] --> C["Concatenate with hidden state"]
    H["Hidden state h_t-1"] --> C

    C --> I0["0-hop term"]
    C --> S1["Support 1"]
    C --> S2["Support 2"]

    S1 --> K1["Diffusion step 1..K"]
    S2 --> K2["Diffusion step 1..K"]

    I0 --> W0["Linear projection"]
    K1 --> W1["Linear projection"]
    K2 --> W2["Linear projection"]

    W0 --> ADD["Sum all projected diffusion orders"]
    W1 --> ADD
    W2 --> ADD

    ADD --> B["Add bias"]
    B --> O["Diffusion graph conv output"]
```

## Standard GRU Flow

This is the standard GRU baseline for comparison. The key difference is that
the linear layers operate on vector features directly, without graph diffusion
over neighbors.

```mermaid
flowchart TD
    X["Input x_t"] --> G["Linear gates on [x_t, h_t-1]"]
    H["Previous hidden state h_t-1"] --> G

    G --> GS["Sigmoid"]
    GS --> R["Reset gate r_t"]
    GS --> U["Update gate u_t"]

    X --> C["Linear candidate on [x_t, r_t .* h_t-1]"]
    H --> RH["r_t .* h_t-1"]
    R --> RH
    RH --> C
    C --> CT["Tanh"]
    CT --> HC["Candidate state h_hat_t"]

    U --> P1["u_t .* h_t-1"]
    H --> P1
    U --> P2["1 - u_t"]
    HC --> P3["(1 - u_t) .* h_hat_t"]
    P2 --> P3
    P1 --> S["Add"]
    P3 --> S
    S --> N["New hidden state h_t"]
```

## DCRNN vs GRU

```mermaid
flowchart LR
    subgraph GRU["Standard GRU"]
        GX["x_t, h_t-1"] --> GL["Dense linear gates/candidate"]
        GL --> GH["Temporal update only"]
    end

    subgraph DCRNN["DCGRU in DCRNN"]
        DX["x_t over graph, h_t-1 over graph"] --> DD["Diffusion graph conv gates/candidate"]
        DD --> DH["Temporal update plus graph diffusion"]
    end
```

Practical summary:

- `ppo`: one agent uses an MLP encoder, then separate MLP action and value
  heads.
- `ppo_dcrnn_mlp`: each agent owns its own MLP plus DCRNN encoder and its own
  heads.
- `ppo_dcrnn_shared_mlp`: agents keep separate heads but reuse one shared MLP
  plus DCRNN encoder.
- GRU models time only.
- DCRNN models time and neighbor-to-neighbor diffusion on the traffic-signal
  graph.
