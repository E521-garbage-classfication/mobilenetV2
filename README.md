# MobilenetV2

## 狀態流程圖

```mermaid
stateDiagram-v2
    [*] --> Standby
    Standby: STANDBY\nPRESS START
    Standby --> Ready: START 按鈕
    Standby --> StopState: STOP 按鈕

    Ready: WELCOME\nREADY...
    Ready --> Classify: 超音波觸發 < threshold
    Ready --> StopState: STOP 按鈕

    Classify: Classifying...\n顯示分類標籤
    Classify --> Ready: 動作完成 + 顯示 2 秒
    Classify --> Ready: Timeout → MANUAL
    Classify --> StopState: STOP 按鈕

    StopState: STOP → LCD=Standby
    StopState --> Standby

    note right of Classify: 分類標籤：\n- PLASTIC\n- GLASS\n- PAPER\n- METAL\n- MANUAL (timeout)

    








