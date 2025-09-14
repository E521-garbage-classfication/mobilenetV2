# MobilenetV2

## 狀態流程圖

## 狀態流程圖

```mermaid
stateDiagram-v2
    [*] --> Standby
    Standby: STANDBY\nPRESS START
    Standby --> Ready: START 按鈕
    Standby --> StopState: STOP 按鈕

    Ready: WELCOME\nREADY...
    Ready --> Classifying: 超音波觸發 < threshold
    Ready --> StopState: STOP 按鈕

    Classifying: Classifying...\n顯示分類標籤
    Classifying --> Ready: 動作完成 + 顯示 2 秒
    Classifying --> Ready: Timeout → MANUAL
    Classifying --> StopState: STOP 按鈕

    StopState: STOP → LCD=Standby
    StopState --> Standby

    note bottom of Classifying
        分類標籤：
        - PLASTIC
        - GLASS
        - PAPER
        - METAL
        - MANUAL (timeout)
    end note





