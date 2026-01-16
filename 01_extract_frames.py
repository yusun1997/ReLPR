import cv2
import os
import glob
#######################
## 01 extract frames ##
#######################

##參數設定
VIDEO_DIR = 'C_HR'

##輸出位置
OUTPUT_DIR = 'dataset/raw_images'

# 採樣頻率
# 30FPS的影片 : INTERVAL = 30 代表一秒一張
# 150，代表每5秒一張

FRAME_INTERVAL = 150

# ==========================================

# 建立輸出資料夾
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"建立輸出資料夾: {OUTPUT_DIR}")

# 搜尋資料夾內所有影片檔案
target_extensions = ['*.MOV','*.mov','*.mp4','*.MP4','*.avi']
video_files = []

for ext in target_extensions:
    #組合路徑進行搜尋
    files = glob.glob(os.path.join(VIDEO_DIR, ext))
    video_files.extend(files)



# set() 會自動把重複的資料刪除，然後我們再轉回 list 並排序
video_files = sorted(list(set(video_files)))


# 簡單排序影片
#video_files.sort()

print(f" 在 {VIDEO_DIR} 下 找到了 {len(video_files)}部原始影片")
print("="*40)

total_saved_count = 0

# 開始批次處理

for video_path in video_files:
    # 取得檔名+副檔名 並拆分出前綴(c1~c30)
    filename = os.path.basename(video_path)
    file_prefix = os.path.splitext(filename)[0]

    print(f"正在處理 {filename}...")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"無法開啟 {filename}，跳過。")
        continue

    frame_count = 0
    saved_this_video = 0

    while(True):
        ret, frame = cap.read()

        #影片讀完就跳出
        if not ret:
            break

        # 當frame_count整除frame interval時才存成圖片，這裡就是每秒一張
        if frame_count % FRAME_INTERVAL == 0:
            # 存檔檔名範例: c1_frame_00120.jpg
            output_filename = f"{file_prefix}_frame_{str(frame_count).zfill(6)}.jpg"
            # zfill(6) 會自動補零到6位數
            save_path = os.path.join(OUTPUT_DIR, output_filename)

            # 存下該frame，每30frame一次
            cv2.imwrite(save_path, frame)
            saved_this_video += 1
            total_saved_count +=1

        frame_count += 1

    
    cap.release()

    print(f"完成，產出 {saved_this_video}張JPG")


print("="*40)
print(f" 全部處理完畢")
print(f" 總共獲得: {total_saved_count} 張訓練圖片")
print(f" 存放位置: {os.path.abspath(OUTPUT_DIR)}")