# MobilenetV2

## 狀態流程圖

```mermaid
sequenceDiagram
    participant Pi as Raspberry Pi
    participant Arduino as Arduino

    Pi->>Arduino: start
    Arduino->>Arduino: LCD = WELCOME/READY (IDLE)

    Arduino->>Arduino: 偵測物件 <10cm → ARMED
    Arduino->>Pi: READY

    Pi->>Pi: 進入 CLASSIFYING (攝影機推論)

    alt 分類成功
        Pi->>Arduino: label (plastic/glass/paper/metal)
        Arduino-->>Pi: ACK:<label>
        Arduino->>Arduino: LCD 顯示標籤
        Arduino->>Arduino: 馬達動作
        Arduino->>Pi: done
    else 超時/不穩定
        Pi->>Arduino: manual
        Arduino-->>Pi: ACK:manual
        Arduino->>Arduino: LCD = MANUAL
        Arduino->>Arduino: 馬達動作
        Arduino->>Pi: done
    end

    Pi->>Arduino: stop
    Arduino->>Arduino: LCD = STANDBY (NOT_RUNNING)



    








