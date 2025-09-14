# MobilenetV2

## 狀態流程圖

```mermaid
stateDiagram-v2
    [*] --> 初始狀態: 啟動 Arduino
    初始狀態: LCD = "STANDBY PRESS START"

    初始狀態 --> 待機狀態: 按下 START (送 start 指令)
    待機狀態: LCD = "WELCOME READY..."

    待機狀態 --> 動作狀態: 超音波偵測 < threshold → 傳送 READY
    動作狀態: LCD = "Classifying..." → 顯示分類標籤

    動作狀態 --> 待機狀態: 動作完成 + 顯示標籤 2 秒
    動作狀態 --> 待機狀態: Timeout → 視為 MANUAL → 執行動作後回待機

    初始狀態 --> 初始狀態: 按 STOP (保持 Standby)
    待機狀態 --> 初始狀態: 按 STOP (送 stop 指令)
    動作狀態 --> 初始狀態: 按 STOP (立即停止 → Standby)

    note right of 動作狀態
        分類標籤：
        - PLASTIC
        - GLASS
        - PAPER
        - METAL
        - MANUAL (timeout)
    end note
