# TODO: 데이터 모으는 시간 조절

import cv2
import os

# Opens the inbuilt camera of laptop to capture video.
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
num_pictures = 50
i = 1
wait = 0
cap.isOpened()

print("Enter your name")
name = input()

if not os.path.exists(f'photos_knn/train/{name}'):
	os.makedirs(f'photos_knn/train/{name}')

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

printProgressBar(0, num_pictures, prefix = 'Progress:', suffix = 'Complete', length = 50)
while i <= num_pictures:
	ret, frame = cap.read()

	# This condition prevents from infinite looping incase video ends.
	if ret == False:
		break

	# Extract an image per 30 frames
	if(int(cap.get(1)) % (fps*30) == 0):
		# Save frame by frame into disk using imwrite method
		cv2.imwrite(f'photos_knn/train/{name}/Frame{str(i)}.jpg', frame)

	printProgressBar(i, num_pictures, prefix = 'Progress:', suffix = 'Complete', length = 50)

	i += 1

cap.release()
cv2.destroyAllWindows()
