import cv2
import os
import sys
from time import strftime, time, sleep

cam_port = 0

ramp_frames = 30

duration = 15*60
delay = 5

cam = cv2.VideoCapture(cam_port)
try: s = sys.argv[1]
except: exit()

def get_image():
	retval, im = cam.read()
	return im

for i in xrange(ramp_frames):
	temp = get_image()

###################
# create a home   #
###################

outpath = s + strftime("%b-%d-%y_%H-%M-%S/")
start = time()
curr = start

if not os.path.exists(outpath):
	os.makedirs(outpath)

###################
# start recording #
###################
while curr-start < duration:
	curr = time()
	cam_capture = get_image()
	out = outpath + str(time()) + '.png'
	cv2.imwrite(out, cam_capture)
	sleep(delay)

###################
# so long suckers #
###################
del(cam)
