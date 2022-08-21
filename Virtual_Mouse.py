import cv2
import time
import numpy as np
import Hand_Tracking_Module as htm
import autopy


WCAM, HCAM = 640, 480
WSCREEN, HSCREEN = autopy.screen.size()


cap = cv2.VideoCapture(0) # 1 ?
detector = htm.HandDetector()
pTime = 0
clicked = 0
xp, yp = 0, 0
xc, yc = 0, 0

while 1:
	success, img = cap.read()
	img = detector.find_hands(img, draw=False)
	lm_list = detector.find_position(img, draw=False)

	if lm_list:
		x1, y1 = lm_list[8][1:]
		x2, y2 = lm_list[12][1:]
		fingers = detector.fingers_up()
		if fingers[0] and not fingers[1]:
			clicked = 0
			x3 = np.interp(x1, (0,WCAM), (0,WSCREEN))
			y3 = np.interp(y1, (0,HCAM-100), (0,HSCREEN))
			xc = xp + (x3 - xc) / 5
			yc = yp + (y3 - yc) / 5
			autopy.mouse.move(WSCREEN - xc, yc)
			xp, yp = xc, yc
		elif fingers[0] and fingers[1]:
			if not clicked:
				autopy.mouse.click()
				clicked = 1

	cTime = time.time()
	fps = 1/(cTime - pTime)
	pTime = cTime

	cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
	cv2.imshow("Image", img)
	cv2.waitKey(1)

