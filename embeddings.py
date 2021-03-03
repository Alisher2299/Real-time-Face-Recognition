import cv2
import sys
import pickle
import face_recognition

name = input('Enter name: ')
ref_id = input('Enter id: ')

try:
	f = open('ref_name.pkl', 'rd')
	ref_dictt = pickle.load(f)
	f.close()
except:
	ref_dictt = {}
ref_dictt[ref_id] = name

f = open('ref_name.pkl', 'wb')
pickle.dump(ref_dictt, f)
f.close()

try:
	f = open('ref_embed.pkl', 'rb')
	embed_dictt = pickle.load(f)
	f.close()
except:
	embed_dictt = {}

for i in range(5):
	print('i: ', i)
	key = cv2.waitKey(1)
	webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
	while True:
		check, frame = webcam.read()
		cv2.imshow('Capturing', frame)
		small_frame = cv2.resize(frame, (0, 0), fx = 0.25, fy = 0.25)
		rgb_small_frame = small_frame[:, :, ::-1]
		key = cv2.waitKey(1)
		if key == ord('s'):
			face_locations = face_recognition.face_locations(rgb_small_frame)#which gives an array listing the co-ordinates of each face!
			if face_locations != []:
				face_encoding = face_recognition.face_encodings(frame)[0]
				if ref_id in embed_dictt:
					embed_dictt[ref_id] += [face_encoding]
				else:
					embed_dictt[ref_id] = [face_encoding]
				webcam.release()
				cv2.waitKey(1)
				cv2.destroyAllWindows()
				break
		elif key == ord('q'):
			sys.exit(1)

f = open('ref_embed.pkl', 'wb')
pickle.dump(embed_dictt, f)
f.close()
