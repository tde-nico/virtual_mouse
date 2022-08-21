import cv2
import mediapipe as mp
import time


class HandDetector():
	def __init__(self, mode=False, max_hands=2, model_complexity=1, detection_con=0.5, trac_con=0.5):
		self.mode = mode
		self.max_hands = max_hands
		self.model_complexity = model_complexity
		self.detection_con = detection_con
		self.trac_con = trac_con
		self.mpHands = mp.solutions.hands
		self.hands = self.mpHands.Hands(mode, max_hands, model_complexity, detection_con, trac_con)
		self.mpDraw = mp.solutions.drawing_utils


	def find_hands(self, img, draw=True):
		imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		results = self.hands.process(imgRGB)
		self.lms = results.multi_hand_landmarks
		if self.lms:
			for handLms in self.lms:
				if draw:
					self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
		return img

	def find_position(self, img, handNo=0, draw=True):
		self.lm_list = []
		if self.lms:
			myHand = self.lms[handNo]
			for id, lm in enumerate(myHand.landmark):
				#print(id, lm)
				h, w, c = img.shape
				cx, cy = int(lm.x * w), int(lm.y * h)
				#print(id, cx, cy)
				self.lm_list.append([id, cx, cy])
				if draw:
					cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)
		return self.lm_list


	def fingers_up(self):
		fingers = []
		for id in range(8,21,4):
			if self.lm_list[id][2] < self.lm_list[id - 2][2]:
				fingers.append(1)
			else:
				fingers.append(0)
		return fingers


def main():
	cap = cv2.VideoCapture(0) # 1 ?
	pTime = 0
	cTime = 0
	detector = HandDetector()
	while 1:
		success, img = cap.read()
		img = detector.find_hands(img)
		lm_list = detector.find_position(img)
		if lm_list:
			print(lm_list[4])

		cTime = time.time()
		fps = 1/(cTime - pTime)
		pTime = cTime

		cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
		cv2.imshow("Image", img)
		cv2.waitKey(1)


if __name__ == '__main__':
	main()