import cv2
import os

# 동영상 파일이 있는 폴더 경로
video_folder_path = r"C:\Users\chosh\Desktop\VideoInput"
# 대체할 이미지 경로
replacement_image_path = r"C:\Users\chosh\Desktop\EutamiasThunder.jpg"

# 대체할 이미지 로드
replacement_image = cv2.imread(replacement_image_path)
if replacement_image is None:
    print("Error: Failed to load replacement face image.")
    exit()

# 얼굴 검출기 초기화
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

for filename in os.listdir(video_folder_path):
    # 비디오 경로 불러오기
    video_path = os.path.join(video_folder_path, filename)

    # 동영상 파일인지 확인
    if not os.path.isfile(video_path) or not video_path.lower().endswith(('.avi', '.mp4', '.mkv', '.mov')):
        continue

    # 동영상 파일 열기
    cap = cv2.VideoCapture(video_path)

    # 동영상 파일이 제대로 열렸는지 확인
    if not cap.isOpened():
        print(f"Error: {filename} 동영상 파일을 열 수 없습니다.")
        continue

    print(f"Processing video: {filename}")

    # 동영상 파일로부터 프레임을 읽어옴
    while True:
        ret, frame = cap.read()

        # 동영상의 끝에 도달하면 종료
        if not ret:
            break

        # 그레이스케일로 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 얼굴 검출
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # 얼굴에 사진 적용
        for (x, y, w, h) in faces:
            # 대체할 얼굴 이미지를 해당 얼굴 영역 크기에 맞게 조정
            replacement_face_resized = cv2.resize(replacement_image, (w, h), interpolation=cv2.INTER_LINEAR)
            # 프레임에 대체할 얼굴 이미지 적용
            frame[y:y+h, x:x+w] = replacement_face_resized

        # Video 창에 표시
        cv2.imshow('Video', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 작업 완료 후, 자원 해제 및 창 닫기
    cap.release()
    cv2.destroyAllWindows()

print("모든 동영상 파일 처리 완료")
