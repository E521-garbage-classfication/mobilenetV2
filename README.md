```mermaid
flowchart TD
    A[Input: 192x192x3 RGB] --> B[Data Pipeline: Augmentation + Preprocess]
    B --> C[MobileNetV2 Backbone\n(include_top=False, ImageNet)]
    C --> D[GlobalAveragePooling2D]

    D --> E[Dense 128 + ReLU + L2]
    E --> F[BatchNormalization]
    F --> G[Dropout 0.45]

    G --> H[Dense 64 + ReLU + L2]
    H --> I[BatchNormalization]
    I --> J[Dropout 0.35]

    J --> K[Dense num_classes + Softmax]
    K --> L[Output: Class Probabilities]
