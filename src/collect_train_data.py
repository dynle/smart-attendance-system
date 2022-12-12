# TODO: 데이터 모으는 시간 조절

import cv2
import os

# Opens the inbuilt camera of laptop to capture video.
cap = cv2.VideoCapture(0)
i = 1
wait = 0
cap.isOpened()

print("Enter your name")
name = input()

while i<=50:
	ret, frame = cap.read()

	# This condition prevents from infinite looping
	# incase video ends.
	if ret == False:
		break

	# if wait % 5000 == 0:
	# Save Frame by Frame into disk using imwrite method
	if not os.path.exists(f'photos_knn/train/{name}'):
		os.makedirs(f'photos_knn/train/{name}')
	cv2.imwrite(f'photos_knn/train/{name}/Frame{str(i)}.jpg', frame)
	i += 1
	
	print(f"Collecting data... {i}/50")

cap.release()
cv2.destroyAllWindows()
