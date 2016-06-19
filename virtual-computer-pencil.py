#-*- coding:utf-8 -*-
import cv2
import numpy as np
import csv

def extract_color( src, h_th_low, h_th_up, s_th, v_th):

	#convert to hsv
	hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
	h,s,v = cv2.split(hsv)

	#red extraction(h=170`5)
	if h_th_low > h_th_up:
		ret, h_dst_1 = cv2.threshold(h, h_th_low, 255, cv2.THRESH_BINARY)
		ret, h_dst_2 = cv2.threshold(h, h_th_up, 255, cv2.THRESH_BINARY_INV)

		dst = cv2.bitwise_or(h_dst_1, h_dst_2)

	#other color extraction
	else:
		ret,dst = cv2.threshold(h, h_th_low, 255, cv2.THRESH_TOZERO)
		ret,dst = cv2.threshold(dst, h_th_up, 255, cv2.THRESH_TOZERO_INV)

		ret,dst = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY)
	#saturation, value extraction
	ret, s_dst = cv2.threshold(s, s_th, 255, cv2.THRESH_BINARY)
	ret, v_dst = cv2.threshold(v, v_th, 255, cv2.THRESH_BINARY)

	#AND operation to binirized h,s,v data
	dst = cv2.bitwise_and(dst, s_dst)
	dst = cv2.bitwise_and(dst, v_dst)

	return dst


rectangle = False
p_s = []
p_e = []
#to check h,s,v of a pixcel
def onMouse( event, x, y, flag, params ):

	global x1,y1,rectangle,img2,p_s,p_e

	wname, img = params

	if event == cv2.EVENT_LBUTTONDOWN:
		rectangle = True
		x1,y1 = x,y
		p_s.append([y,x])


	if event == cv2.EVENT_MOUSEMOVE:
		if rectangle == True:
			img2 =np.copy(img)
			cv2.rectangle(img2, (x1,y1), (x,y), (0,255,0), 2)
			cv2.imshow(wname,img2)

	if event == cv2.EVENT_LBUTTONUP:
		rectangle = False
		p_e.append([y,x])
		#cv2.rectangle(img, (x1,y1), (x,y), (0,255,0), 2)
		#cv2.imshow(wname,img)



	if event == cv2.EVENT_RBUTTONDOWN:
		hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
		print (hsv[y][x])



#for trackbar
def nothing(x):
	pass

if __name__=="__main__":

	#for prev(line 216)
	switch = False

	#window name
	wname0 = "green"
	wname1 = "hand"
	wname3 = "operation"
	wname = "extracted"

	#make window
	cv2.namedWindow( wname0 )
	cv2.namedWindow( wname )
	cv2.namedWindow( wname1 )
	cv2.namedWindow( wname3 )

	#create trackbars for hsv change ;green
	cv2.createTrackbar("H_low",wname0,45,180,nothing)
	cv2.createTrackbar("H_up",wname0,65,180,nothing)
	cv2.createTrackbar("S",wname0,100,180,nothing)
	cv2.createTrackbar("V",wname0,120,180,nothing)

	#create trackbars for hsv change ;hand
	cv2.createTrackbar("H_low",wname1,135,180,nothing)
	cv2.createTrackbar("H_up",wname1,30,180,nothing)
	cv2.createTrackbar("S",wname1,40,180,nothing)
	cv2.createTrackbar("V",wname1,0,180,nothing)

	#create trackbars for setting kernel
	cv2.createTrackbar("op_y",wname3,1,10,nothing)
	cv2.createTrackbar("op_x",wname3,1,10,nothing)
	cv2.createTrackbar("cl_y",wname3,10,50,nothing)
	cv2.createTrackbar("cl_x",wname3,50,100,nothing)
	cv2.createTrackbar("dil_y",wname3,2,10,nothing)
	cv2.createTrackbar("dil_x",wname3,5,10,nothing)
	cv2.createTrackbar("dil_i",wname3,7,10,nothing)

	#set camera
	cap = cv2.VideoCapture(0)
	#answer image
	src = cv2.imread("sample.jpg")
	#for prev green_img
	prev = np.zeros([480,640],np.uint8)

	green_img = []
	hand_img = []

	#load information of operated area
	try:
		p_s_csv = csv.reader(open("p_s.csv","r"))
	except IOError:
		ret, img = cap.read()
		wname2 = "check"
		cv2.namedWindow(wname2)
		cv2.setMouseCallback( wname2, onMouse, [ wname2, img] )
		cv2.imshow(wname2,img)
		while cv2.waitKey(0) != ord("s"):
			pass
		# save information of area to be opperated
		cv2.destroyWindow(wname2)
		fs = open("p_s.csv","wb")
		dataWriter = csv.writer(fs)
		dataWriter.writerows(p_s)
		fs.close()
		fe = open("p_e.csv","wb")
		dataWriter = csv.writer(fe)
		dataWriter.writerows(p_e)
		fe.close()
	p_s_csv = csv.reader(open("p_s.csv","r"))
	p_s = [[int(elm) for elm in v] for v in p_s_csv]
	p_e_csv = csv.reader(open("p_e.csv","r"))
	p_e = [[int(elm) for elm in v] for v in p_e_csv]


	ret,img = cap.read()

	while (1):
		key = cv2.waitKey(5)
		if key == 27:
			break	 #finished by esc_key
		ret,img = cap.read()

		#check h,s,v when coversion doesn't go well
		if key == ord("s"):
			switch = False
			img3 = img
			wname2 = "check"
			cv2.namedWindow(wname2)
			cv2.setMouseCallback( wname2, onMouse, [ wname2, img3 ] )
			cv2.imshow(wname2,img3)
			while cv2.waitKey(0) != ord("s"):
				pass
			#save information of area to be opperated
			cv2.destroyWindow(wname2)
			fs = open("p_s.csv","wb")
			dataWriter = csv.writer(fs)
			dataWriter.writerows(p_s)
			fs.close()
			fe = open("p_e.csv","wb")
			dataWriter = csv.writer(fe)
			dataWriter.writerows(p_e)
			fe.close()


		#save answer img
		if key == ord("p"):
			cv2.imwrite("sample.jpg",img)
			print "success"

		#restart
		if key ==ord("r"):
			p_s = []
			p_e = []
			prev = []
			src = cv2.imread("sample.jpg")
			print "restart"

		#get cuurent poosition of four trackbars ;green
		hl = cv2.getTrackbarPos("H_low",wname0)
		hu = cv2.getTrackbarPos("H_up",wname0)
		s = cv2.getTrackbarPos("S",wname0)
		v = cv2.getTrackbarPos("V",wname0)

		#get cuurent poosition of four trackbars ;hand
		hl_hand = cv2.getTrackbarPos("H_low",wname1)
		hu_hand = cv2.getTrackbarPos("H_up",wname1)
		s_hand = cv2.getTrackbarPos("S",wname1)
		v_hand = cv2.getTrackbarPos("V",wname1)

		#get current position of trackbars:operation
		op_y = cv2.getTrackbarPos("op_y",wname3)
		op_x = cv2.getTrackbarPos("op_x",wname3)
		cl_y = cv2.getTrackbarPos("cl_y",wname3)
		cl_x = cv2.getTrackbarPos("cl_x",wname3)
		dil_y = cv2.getTrackbarPos("dil_y",wname3)
		dil_x = cv2.getTrackbarPos("dil_x",wname3)
		dil_i = cv2.getTrackbarPos("dil_i",wname3)

		#kernel parameter
		kernel_op = np.ones((op_y,op_x),np.uint8)
		kernel_cl = np.ones((cl_y,cl_x),np.uint8)
		kernel_dil = np.ones((dil_y,dil_x),np.uint8)
		kernel_hand_dil =np.ones((3,3),np.uint8)

		#clear
		green_img = []
		hand_img = []

		#extraction
		for i in range(len(p_s)):

			green_img.append(extract_color(img[p_s[i][0]:p_e[i][0]+1, p_s[i][1]:p_e[i][1]+1], hl, hu, s, v))
			hand_img.append(extract_color(img[p_s[i][0]:p_e[i][0]+1, p_s[i][1]:p_e[i][1]+1],hl_hand,hu_hand,s_hand,v_hand))
			#opening and closing operation
			green_img[i] = cv2.morphologyEx(green_img[i], cv2.MORPH_OPEN, kernel_op)
			green_img[i] = cv2.morphologyEx(green_img[i], cv2.MORPH_CLOSE, kernel_cl)
			#dilation
			green_img[i] = cv2.dilate(green_img[i], kernel_dil, iterations=dil_i)
			#hand_img[i] = cv2.dilate(hand_img[i],kernel_hand_dil , iterations=3)
			if switch == True:
				green_img[i] = cv2.bitwise_or(green_img[i], prev[i])

		prev = green_img
		switch = True

		#convert green and not hand point to answer img
		for i in range(len(p_s)):
			(img[p_s[i][0]:p_e[i][0]+1, p_s[i][1]:p_e[i][1]+1])[(green_img[i]==255)&(hand_img[i]==0)] = (src[p_s[i][0]:p_e[i][0]+1, p_s[i][1]:p_e[i][1]+1])[(green_img[i]==255)&(hand_img[i]==0)]


		cv2.imshow(wname,img)
	cv2.destroyAllWindows()
