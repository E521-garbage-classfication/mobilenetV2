# MobilenetV2

## 狀態流程圖

```mermaid
flowchart TD
    A[輸入影像<br/>192×192×3 RGB] --> B[資料管線<br/>訓練: 資料增強<br/>推論: 僅前處理]
    B --> C[MobileNetV2 主幹網路<br/>include_top=False<br/>weights='imagenet']
    C --> D[GlobalAveragePooling2D]

    D --> E[Dense(128, ReLU)<br/>L2(2e-4)]
    E --> F[BatchNormalization]
    F --> G[Dropout(0.45)]

    G --> H[Dense(64, ReLU)<br/>L2(2e-4)]
    H --> I[BatchNormalization]
    I --> J[Dropout(0.35)]

    J --> K[Dense(num_classes, Softmax)<br/>L2(2e-4)]
    K --> L[輸出機率分佈<br/>(paper / plastic / metal / glass)]




    








