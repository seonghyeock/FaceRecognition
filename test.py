import cv2
import os

# 동영상 파일이 있는 폴더 경로
video_folder_path = r"C:\Users\chosh\Desktop\VideoInput"

# 폴더 내의 모든 파일에 대해 반복
for filename in os.listdir(video_folder_path):
    # 파일 경로 생성
    video_path = os.path.join(video_folder_path, filename)

    # 동영상 파일인지 확인
    if not os.path.isfile(video_path) or not video_path.lower().endswith(('.avi', '.mp4', '.mkv', '.mov')):
        continue

    # 얼굴 검출기 초기화
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 동영상 파일 열기
    cap = cv2.VideoCapture(video_path)

    # 동영상 파일이 제대로 열렸는지 확인
    if not cap.isOpened():
        print(f"Error: {filename} 동영상 파일을 열 수 없습니다.")
        continue

    print(f"Processing video: {filename}")

    # Face 창 생성
    cv2.namedWindow('Face', cv2.WINDOW_NORMAL)

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

        # Video 창에 인식한 얼굴에 사각형 박스 표시
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Video 창에 표시
        cv2.imshow('Video', frame)

        # Face 창에 인식된 얼굴만 표시
        face_frame = frame.copy()
        for (x, y, w, h) in faces:
            face_frame = cv2.rectangle(face_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('Face', face_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 작업 완료 후, 자원 해제 및 창 닫기
    cap.release()
    cv2.destroyAllWindows()

print("모든 동영상 파일 처리 완료")