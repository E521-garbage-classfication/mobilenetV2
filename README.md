# MobilenetV2

## 狀態流程圖

```mermaid
stateDiagram-v2
    [*] --> 初始狀態
    初始狀態: STANDBY\nPRESS START
    初始狀態 --> 待機狀態: START 按鈕
    初始狀態 --> 初始狀態: STOP 按鈕 (保持 Standby)

    待機狀態: WELCOME\nREADY...
    待機狀態 --> 動作狀態: 超音波觸發 < threshold
    待機狀態 --> 初始狀態: STOP 按鈕

    動作狀態: Classifying...\n顯示分類標籤
    動作狀態 --> 待機狀態: 動作完成 + 顯示 2 秒
    動作狀態 --> 待機狀態: Timeout → MANUAL
    動作狀態 --> 初始狀態: STOP 按鈕

    note right of 動作狀態
        分類標籤：
        - PLASTIC
        - GLASS
        - PAPER
        - METAL
        - MANUAL (timeout)
    end note

