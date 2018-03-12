# import multiply_detections
from hog_window_search import find_cars
import pickle
import cv2


dist_pickle = pickle.load(open("output/svc_model.p", "rb"))

# get attributes of our svc object
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
colorspace = dist_pickle["colorspace"]
hog_channel = dist_pickle["hog_channel"]

spatial_size = (32, 32)
hist_bins = 32

ystart = 400
ystop = 656
scale = 2


def pipeline(img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out_img, box_list = find_cars(img, ystart, ystop, scale,
                                  svc, X_scaler, orient, pix_per_cell,
                                  cell_per_block, colorspace)
    for bbox in box_list:
        cv2.rectangle(out_img, bbox[0],
                      bbox[1], (0, 0, 255), 6)
    return out_img


video_path = "project_video.mp4"

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output/output.avi', fourcc, 25.0, (1280, 720), isColor=True)

cap = cv2.VideoCapture(video_path)
cnt = 0
while cap.isOpened():
# for i in range(25):
    ret, frame = cap.read()
    if ret is True:
        result = pipeline(frame)
        out.write(result)
        cnt += 1
        print(cnt)
    else:
        break
