# UR3 三色骨牌排列

固定俯視相機找出一紅、一黃、一藍三塊已站立的骨牌。程式用 OpenCV 算出骨牌頂面的中心和長邊角度，轉成 UR3 基座座標，再用 RTDE 將骨牌排成紅 → 黃 → 藍。骨牌寬面朝相鄰骨牌，可從紅色端往黃色端推倒。程式完成排列後停在示教的停駐點，由人推第一塊測試連倒。

夾放流程沿用 `Project#3` 第三週的 `moveL` 與數位輸出夾爪。顏色辨識、中心座標和姿態的處理參考 `VLM_student`，固定任務以本機 OpenCV 完成。整個專案可把這個資料夾單獨搬走使用，編譯及執行只需系統安裝的 SDK 和本資料夾內的檔案。

## 編譯與離線試跑

環境為 Windows x64、Visual Studio 2019 的「使用 C++ 的桌面開發」、OpenCV 3.4.3、Boost 1.75 標頭，以及 `Project#3` 使用的 `ur_rtde` 1.4.0 Windows SDK（含 Boost 函式庫）。在 PowerShell 進入本目錄執行。

```powershell
cd E:\group_4_workspace\Domino_Row
powershell -NoProfile -ExecutionPolicy Bypass -File .\build.ps1
.\build\domino.exe --demo
```

`--demo` 用程式產生透視影像，讓紅、黃對調佔住彼此目標，藍色以斜角放在一旁。畫面經過與實機相同的辨識及規劃，模擬搬移後再從影像查驗結果。SDK 路徑不同時可在編譯命令最後指定 `-OpenCv 'C:\你的OpenCV\build' -UrRtde 'C:\你的URSDK' -Boost 'C:\你的Boost根目錄'`。

## 相機與骨牌尺寸

1. 複製範例設定，再進入預覽。

   ```powershell
   Copy-Item .\domino.example.json .\domino.local.json
   .\build\domino.exe --preview
   ```

2. 固定相機並讓鏡頭盡量垂直於桌面。設定 `camera.index`、`width`、`height`，讓三塊頂面、整排目標和暫放點都能入鏡。使用霧面素色背景、穩定照明，以及頂面有完整色塊且大小相同的長方形骨牌。點預覽畫面可在主控台看到 `[x, y]` 和 HSV 值，按 `Q` 或 `Esc` 離開。辨識不到時用讀到的 HSV 值調整 `hsv` 範圍。
3. 在工區四周取至少四個位置分散的標記。標記要與**站立骨牌頂面同高**，可放在與骨牌等高的治具上。點選標記中心，把像素記到 `calibration.image_px`。用基座座標量測每個標記的 XY（公尺），以相同順序填進 `calibration.robot_xy_m`。若用手臂對準標記，可以在設定 `robot.ip` 後用 `.\build\domino.exe --tcp` 讀取 TCP，只取輸出的前兩個數字。程式用這些對應點建立平面單應轉換，標記包圍的區域就是可規劃的工作區。相機移動或切換解析度後要重做校正。
4. 實測 `domino.height_m`、`width_m`、`thickness_m`。寬度是站立時頂面長邊，厚度是頂面短邊。程式以這兩個尺寸篩選頂面輪廓。換回預覽確認三色各有一個框，且目標十字落在桌上預期位置。也可用 `--preview --image 路徑` 檢查同解析度的靜態照片。

## 排列位置與手臂示教

- `row.origin_xy_m` 是第一塊中心。`direction_deg` 是排列方向，以手臂基座 +X 方向為 0°，朝 +Y 方向為 90°。骨牌頂面長邊會與排列方向垂直。`spacing_m` 是相鄰中心距，預設 32 mm，範例骨牌高度 50 mm、厚度 9 mm。換了骨牌先手排三塊，確認連倒且夾爪在相同間距可順利放入及退出，再填入實測設定。
- `row.buffer_xy_m` 是暫放點，應離開排列路線並位於校正區內。若某塊佔住另一塊的目標，程式會檢查是否能先暫放。任意起始角度都可以辨識，初始擺放需讓骨牌各自留有足夠夾取空間。`gripper` 的兩個半尺寸是夾爪張開時相對夾取中心的水平外緣，沿骨牌長、短邊分別量測。程式會加上 `clearance_m` 檢查夾取及放下時是否會碰到其他骨牌。
- 讓一塊骨牌以已知角度站立，在示教器設定實際夾爪的 TCP，示教垂直向下、夾爪居中且能夾穩的位置。取 `--tcp` 印出的 `[x, y, z, rx, ry, rz]` 填入 `robot.grasp_reference_pose`，並把這塊骨牌頂面長邊在基座 XY 的角度寫入 `grasp_reference_long_axis_deg`。程式沿用示教的 Z，並依其他骨牌的角度繞基座 Z 軸旋轉姿態。三塊骨牌需站在同一平面。
- 示教同一向下姿態的 `robot.park_pose`，讓手臂停駐時不遮住骨牌頂面。這個點的 Z 同時用於所有水平搬運，要比夾取 Z 至少高一個骨牌高度，並讓夾爪與懸掛的骨牌跨過工區內物體。預設用第三週夾爪的 DO1、`1` 張開、`0` 夾緊，依現場接線調整 `gripper.do`、`open_high` 和 `settle_ms`。

## 試排與實機排列

```powershell
.\build\domino.exe --plan
.\build\domino.exe --run
```

`--plan` 只讀相機並列出三塊的座標及搬移步驟。也可用 `--plan --image 路徑` 對校正後的靜態照片試排。`--run` 從示教的停駐點開始，每次移動包含抬升、水平搬運、放下及回停駐點。程式再次取影像查驗，確認成功才做下一次搬移。首輪實機測試請用示教器低速，確認 TCP 和夾爪開關，並讓工作區保持淨空。`Ctrl+C` 會在當前指令結束後停止後續動作，需要立即停止時使用示教器急停。

辨識不到三色、場景不穩、找不到有空間的夾放路徑、手臂回報失敗或相機未確認放置時，程式會結束並顯示原因。排除現場問題後回到停駐點，再用 `--plan` 依目前擺放重新規劃。
