# MobilenetV2

## 狀態流程圖

```mermaid
sequenceDiagram
    participant Pi as Raspberry Pi
    participant Arduino as Arduino

    Pi->>Arduino: start
    Arduino->>Arduino: LCD=STANDBY → IDLE (WELCOME/READY)

    Arduino->>Arduino: 偵測物件 <10cm
    Arduino->>Pi: READY

    Pi->>Arduino: label (plastic / glass / paper / metal)
    Arduino-->>Pi: ACK:<label>
    Arduino->>Arduino: LCD 顯示標籤
    Arduino->>Arduino: 馬達動作
    Arduino->>Pi: done

    alt Timeout / 無法分類
        Pi->>Arduino: manual
        Arduino-->>Pi: ACK:manual
        Arduino->>Arduino: LCD=MANUAL
        Arduino->>Arduino: 馬達動作
        Arduino->>Pi: done
    end

    Pi->>Arduino: stop
    Arduino->>Arduino: LCD=STANDBY


    








