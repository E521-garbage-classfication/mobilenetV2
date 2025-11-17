
```markdown
```mermaid
flowchart LR
    A[Input 192x192x3] --> B[Data Pipeline]
    B --> C[MobileNetV2 Backbone]
    C --> D[Global Avg Pool]

    D --> E[Dense 128 + ReLU]
    E --> F[BatchNorm]
    F --> G[Dropout 0.45]

    G --> H[Dense 64 + ReLU]
    H --> I[BatchNorm]
    I --> J[Dropout 0.35]

    J --> K[Softmax Output]