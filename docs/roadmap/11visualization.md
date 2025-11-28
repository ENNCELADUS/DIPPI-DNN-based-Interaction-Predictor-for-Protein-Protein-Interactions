## Final Implementation Plan for visualize module

### Architecture

```
Layout YAML ──▶ Python Generator ──▶ HTML/CSS ──▶ Browser Preview
                                          │
                                          ▼
                                   PNG/SVG Export
```

### File Structure

```
visualize/
├── html/
│   ├── __init__.py
│   ├── generator.py       # Main HTML/CSS generator
│   ├── styles.py          # Preset styles (colors, fonts, block types)
│   └── parser.py          # YAML layout parser + validation
├── layouts/
│   └── example.yaml       # Example layout file
├── cli.py                 # CLI: generate HTML, export PNG/SVG
└── (existing files...)
```

### Layout File Format (YAML)

```yaml
meta:
  title: "TUnA Architecture"
  theme: academic        # preset theme

# Lane definitions (top to bottom order)
lanes:
  - id: protein_a
    label: "Protein A (length n)"
  - id: protein_b  
    label: "Protein B (length m)"
  - id: merged              # single lane after merge
    label: null

# Blocks (ordered left-to-right within lanes)
blocks:
  # Spanning block (covers multiple lanes)
  - id: esm
    lanes: [protein_a, protein_b]
    label: "ESM-2\n150M"
    style: encoder
    
  # Per-lane blocks
  - id: emb_a
    lane: protein_a
    label: "n × 640"
    style: embedding_a
    annotation: "n × 640"    # dimension annotation above
    
  - id: emb_b
    lane: protein_b
    label: "m × 640"
    style: embedding_b

  # Transformer block (compound - auto-expands to sub-blocks)
  - id: intra_enc_a
    lane: protein_a
    type: transformer       # recognized compound type
    components: [linear, attention, add_norm, feedforward, add_norm]
    style: transformer_default

  # Merge point
  - id: concat
    lanes: [protein_a, protein_b]
    type: merge             # special type: joins lanes
    label: "Concatenate"
    merge_to: merged        # output goes to this lane

  # Post-merge blocks in single lane
  - id: max_pool
    lane: merged
    label: "Max"
    style: pooling

# Groups (dashed containers)
groups:
  - id: intra_encoder_group
    title: "Intra-Protein Interaction Encoder"
    subtitle: "Shared Weights"
    blocks: [intra_enc_a, intra_enc_b]
    style: dashed

# Connections (explicit)
connections:
  - from: esm
    to: [emb_a, emb_b]      # fan-out
    style: arrow
    
  - from: emb_a
    to: intra_enc_a
    label: "n × d"
    
  - from: [encoded_a, encoded_b]   # fan-in
    to: concat
    style: arrow
```

### Style Presets (`styles.py`)

```python
THEMES = {
    "academic": {
        "colors": {
            "embedding_a": "#FFB3BA",    # pink
            "embedding_b": "#B5EAD7",    # green
            "attention": "#FFD4A3",       # orange
            "feedforward": "#A3D5FF",     # blue
            "norm": "#FFFFBA",            # yellow
            "linear": "#D3D3D3",          # gray
            "output": "#87CEEB",          # cyan
            # ...
        },
        "fonts": {
            "title": "Arial, sans-serif",
            "block": "Arial, sans-serif",
        },
        "block": {
            "border_radius": "8px",
            "border_width": "2px",
        }
    }
}
```

### Key Features

| Feature | Implementation |
|---------|---------------|
| **Lane-based layout** | CSS Grid with named rows per lane |
| **Block positioning** | Auto-flow left-to-right, order defined by YAML sequence |
| **Spanning blocks** | `grid-row: span N` for multi-lane blocks |
| **Connections** | SVG overlay with `<path>` elements |
| **Groups** | Absolutely positioned dashed `<div>` containers |
| **Annotations** | Small labels above/below blocks |
| **Export** | Playwright for headless PNG/SVG capture |

### CLI Commands

```bash
# Generate HTML from layout
python -m visualize html --layout layouts/tuna.yaml --output img/tuna.html

# Export to PNG/SVG
python -m visualize export --input img/tuna.html --format png --output img/tuna.png
python -m visualize export --input img/tuna.html --format svg --output img/tuna.svg

# One-shot: layout → PNG
python -m visualize render --layout layouts/tuna.yaml --format png --output img/tuna.png
```

### Implementation Order

1. **`styles.py`** — Define presets and theme structure
2. **`parser.py`** — Parse and validate YAML layout files  
3. **`generator.py`** — Convert parsed layout → HTML/CSS/SVG
4. **`cli.py`** — Wire up CLI commands
5. **`layouts/example.yaml`** — Create TUnA-like example layout
6. **Export integration** — Add Playwright-based PNG/SVG export