
## V1

```
emb_a                                   emb_b
[B, L_a, 1536]                          [B, L_b, 1536]
   │                                        │
   ├──── Length-aware AveragePool ──────────┤
   ▼                                        ▼
pooled_a [B, 1536]                     pooled_b [B, 1536]
   │                                        │
   ├────────────── ProteinEncoder ──────────┤
   │              (shared weights)          │
   ▼                                        ▼
Linear(1536 → 256) [B, 256]            Linear(1536 → 256) [B, 256]
   │                                        │
   ▼                                        ▼
LayerNorm(256) [B, 256]                LayerNorm(256) [B, 256]
   │                                        │
   ▼                                        ▼
ReLU [B, 256]                          ReLU [B, 256]
   │                                        │
   ▼                                        ▼
Dropout(0.2) [B, 256]                  Dropout(0.2) [B, 256]
   │                                        │
   ▼                                        ▼
enc_a [B, 256]                         enc_b [B, 256]
   │                                        │
   └───────────────┬────────────────────────┘
                   │
                   ▼
combined = concat(enc_a, enc_b) [B, 512]
                   │
                   ▼
┌───── InteractionHead ─────┐
│ Linear(512 → 256) [B, 256]│
│    │                      │
│    ▼                      │
│ LayerNorm(256) [B, 256]   │
│    │                      │
│    ▼                      │
│ ReLU [B, 256]             │
│    │                      │
│    ▼                      │
│ Dropout(0.2) [B, 256]     │
│    │                      │
│    ▼                      │
│ Linear(256 → 128) [B, 128]│
│    │                      │
│    ▼                      │
│ LayerNorm(128) [B, 128]   │
│    │                      │
│    ▼                      │
│ ReLU [B, 128]             │
│    │                      │
│    ▼                      │
│ Dropout(0.2) [B, 128]     │
│    │                      │
│    ▼                      │
│ Linear(128 → 1) [B, 1]    │
└───────────────────────────┘
                   │
                   ▼
logits [B, 1]
```

## V2

```
emb_a                                   emb_b
[B, L_a, 1536]                          [B, L_b, 1536]
   │                                        │
   ├────────────── SiameseEncoder ──────────┤
   │               (shared weights)         │
   ▼                                        ▼
Linear(1536 → 384) [B, L_a, 384]       Linear(1536 → 384) [B, L_b, 384]
   │                                        │
   ▼                                        ▼
Dropout(0.1 token) [B, L_a, 384]       Dropout(0.1 token) [B, L_b, 384]
   │                                        │
   ▼                                        ▼
LayerNorm(384) [B, L_a, 384]           LayerNorm(384) [B, L_b, 384]
   │                                        │
   ▼                                        ▼
GELU [B, L_a, 384]                      GELU [B, L_b, 384]
   │                                        │
   ▼                                        ▼
h_a [B, L_a, 384]                       h_b [B, L_b, 384]
   │                                        │
   └─────────┬──────────────────────────────┘
             │
             ▼
cls_token param [1, 1, 256] → repeat [B, 1, 256]
             │
             ▼
┌─────── CrossAttentionLayer ×2 ───────┐
│ h_a route:                           │
│   LayerNorm(h_a) [B, L_a, 384]       │
│       │                              │
│       ▼                              │
│   MultiHeadAttn(Q=h_a, K=h_b, V=h_b) │
│       │                              │
│       ▼                              │
│   Dropout + Residual → h_a [B, L_a, 384]│
│ h_b route:                           │
│   LayerNorm(h_b) [B, L_b, 384]       │
│       │                              │
│       ▼                              │
│   MultiHeadAttn(Q=h_b, K=h_a, V=h_a) │
│       │                              │
│       ▼                              │
│   Dropout + Residual → h_b [B, L_b, 384]│
│ combine:                             │
│   concat(h_a, h_b) [B, L_a+L_b, 384] │
│ cls route:                           │
│   cls [B, 1, 384]                    │
│       │                              │
│       ▼                              │
│   LayerNorm(cls) [B, 1, 384]         │
│       │                              │
│       ▼                              │
│   MultiHeadAttn(Q=cls, K=concat, V=concat)│
│       │                                   │
│       ▼                                   │
│   Dropout + Residual → cls [B, 1, 384]    │
│       │                                   │
│       ▼                                   │
│   LayerNorm(cls) [B, 1, 384]              │
│       │                                   │
│       ▼                                   │
│   Linear(384 → 768) [B, 1, 768]           │
│       │                                   │
│       ▼                                   │
│   GELU [B, 1, 768]                        │
│       │                                   │
│       ▼                                   │
│   Dropout(0.05) [B, 1, 768]               │
│       │                                   │
│       ▼                                   │
│   Linear(768 → 384) [B, 1, 384]           │
│       │                                   │
│       ▼                                   │
│   Dropout(0.05) + Residual → cls [B, 1, 384]│
└──────────────────────────────────────────┘
             │
             ▼
cls_token_out squeeze(1) [B, 384]
             │
             ▼
┌────────── MLPHead ──────────┐
│ Linear(384 → 256) [B, 256]  │
│    │                        │
│    ▼                        │
│ LayerNorm(256) [B, 256]     │
│    │                        │
│    ▼                        │
│ GELU [B, 256]               │
│    │                        │
│    ▼                        │
│ Dropout(0.2) [B, 256]       │
│    │                        │
│    ▼                        │
│ Linear(256 → 64) [B, 64]    │
│    │                        │
│    ▼                        │
│ LayerNorm(64) [B, 64]       │
│    │                        │
│    ▼                        │
│ GELU [B, 64]                │
│    │                        │
│    ▼                        │
│ Dropout(0.2) [B, 64]        │
│    │                        │
│    ▼                        │
│ Linear(64 → 1) [B, 1]       │
└────────────────────────────┘
             │
             ▼
logits [B, 1]
```

## V4

```
emb_a                                    emb_b
[B, L_a, 1536]                           [B, L_b, 1536]
    │                                        │
    ├────────────── SiameseEncoder ──────────┤
    │               (shared weights)         │
    ▼                                        ▼
Linear(1536 → 384)                      Linear(1536 → 384)
    │                                        │
    ▼                                        ▼
Dropout(0.1 token) [B, L_a, 384]         Dropout(0.1 token) [B, L_b, 384]
    │                                        │
    ▼                                        ▼
LayerNorm(384) [B, L_a, 384]             LayerNorm(384) [B, L_b, 384]
    │                                        │
    ▼                                        ▼
GELU [B, L_a, 384]                       GELU [B, L_b, 384]
    │                                        │
    ▼                                        ▼
h_a [B, L_a, 384]                        h_b [B, L_b, 384]
    │                                        │
    └──────────┬─────────────────────────────┘
               │
               ▼
    ┌─── CrossAttentionLayer ×N ───┐
    │                              │
    │  ┌────────────────────────┐  │
    │  │ LayerNorm(h_a)         │  │
    │  │     │                  │  │
    │  │     ▼                  │  │
    │  │ MultiHeadAttn(Q=h_a,   │  │
    │  │              K=h_b,    │  │
    │  │              V=h_b)    │  │
    │  │     │                  │  │
    │  │     ▼                  │  │
    │  │ Dropout + Residual     │  │
    │  │     │                  │  │
    │  │     ▼                  │  │
    │  │ h_a' [B, L_a, 384]     │  │
    │  └────────────────────────┘  │
    │                              │
    │  ┌────────────────────────┐  │
    │  │ LayerNorm(h_b)         │  │
    │  │     │                  │  │
    │  │     ▼                  │  │
    │  │ MultiHeadAttn(Q=h_b,   │  │
    │  │              K=h_a',   │  │
    │  │              V=h_a')   │  │
    │  │     │                  │  │
    │  │     ▼                  │  │
    │  │ Dropout + Residual     │  │
    │  │     │                  │  │
    │  │     ▼                  │  │
    │  │ h_b' [B, L_b, 384]     │  │
    │  └────────────────────────┘  │
    │                              │
    └──────────────────────────────┘
               │
               ▼
h_a' [B, L_a, 384]               h_b' [B, L_b, 384]
    │                                │
    ├───── AttentionPooling ─────────┤
    │      (separate weights)        │
    ▼                                ▼
┌─────────────────┐          ┌─────────────────┐
│ query_a [1,1,384]│          │ query_b [1,1,384]│
│     │           │          │     │           │
│     ▼           │          │     ▼           │
│ LayerNorm(h_a') │          │ LayerNorm(h_b') │
│     │           │          │     │           │
│     ▼           │          │     ▼           │
│ MultiHeadAttn   │          │ MultiHeadAttn   │
│ (Q=query,       │          │ (Q=query,       │
│  K=h_a',V=h_a') │          │  K=h_b',V=h_b') │
│     │           │          │     │           │
│     ▼           │          │     ▼           │
│ squeeze(1)      │          │ squeeze(1)      │
└─────────────────┘          └─────────────────┘
    │                                │
    ▼                                ▼
v_a [B, 384]                     v_b [B, 384]
    │                                │
    └───────────┬────────────────────┘
                │
                ▼
        ┌───────────────┐
        │ product = v_a * v_b      [B, 384]
        │ diff = |v_a - v_b|       [B, 384]
        │ concat([product, diff])
        └───────────────┘
                │
                ▼
        combined [B, 768]
                │
                ▼
        ┌─── MLPHead ───┐
        │               │
        │ Linear(768 → 512)
        │     │         │
        │     ▼         │
        │ LayerNorm(512)│
        │     │         │
        │     ▼         │
        │ GELU          │
        │     │         │
        │     ▼         │
        │ Dropout(0.2)  │
        │     │         │
        │     ▼         │
        │ Linear(512 → 256)
        │     │         │
        │     ▼         │
        │ LayerNorm(256)│
        │     │         │
        │     ▼         │
        │ GELU          │
        │     │         │
        │     ▼         │
        │ Dropout(0.2)  │
        │     │         │
        │     ▼         │
        │ Linear(256 → 64)
        │     │         │
        │     ▼         │
        │ LayerNorm(64) │
        │     │         │
        │     ▼         │
        │ GELU          │
        │     │         │
        │     ▼         │
        │ Dropout(0.2)  │
        │     │         │
        │     ▼         │
        │ Linear(64 → 1)│
        │               │
        └───────────────┘
                │
                ▼
logits [B, 1]
```
