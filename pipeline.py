# import multiply_detections
from hog_window_search import find_cars
from heat import apply_heat
import pickle
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg


dist_pickle = pickle.load(open("output/svc_model.p", "rb"))

# get attributes of our svc object
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
colorspace = dist_pickle["colorspace"]
hog_channel = dist_pickle["hog_channel"]

# spatial_size = (32, 32)
# hist_bins = 32




def pipeline(img):
    # ystart = 400
    # ystop = 656
    #
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bbox_list = []
    # out_img, boxes1 = find_cars(img, ystart, ystop, 1,
    #                               svc, X_scaler, orient, pix_per_cell,
    #                               cell_per_block, colorspace)

    # out_img, boxes2 = find_cars(img, ystart, ystop, 2,
    #                               svc, X_scaler, orient, pix_per_cell,
    #                               cell_per_block, colorspace)
    #
    # out_img, boxes4 = find_cars(img, ystart, ystop, 4,
    #                               svc, X_scaler, orient, pix_per_cell,
    #                               cell_per_block, colorspace)

    # out_img, boxes4 = find_cars(img, ystart, ystop, 4,
    #                             svc, X_scaler, orient, pix_per_cell,
    #                             cell_per_block, colorspace)

    # for bbox in box_list:
    #     cv2.rectangle(out_img, bbox[0],
    #                   bbox[1], (0, 0, 255), 6)

# new way:
    rectangles = []

    ystart = 400
    ystop = 464
    scale = 1.0
    rectangles.append(find_cars(img, ystart, ystop, scale,
                                  svc, X_scaler, orient, pix_per_cell,
                                  cell_per_block, colorspace))
    ystart = 416
    ystop = 480
    scale = 1.0
    rectangles.append(find_cars(img, ystart, ystop, scale,
                                  svc, X_scaler, orient, pix_per_cell,
                                  cell_per_block, colorspace))
    ystart = 400
    ystop = 496
    scale = 1.5
    rectangles.append(find_cars(img, ystart, ystop, scale,
                                  svc, X_scaler, orient, pix_per_cell,
                                  cell_per_block, colorspace))
    ystart = 432
    ystop = 528
    scale = 1.5
    rectangles.append(find_cars(img, ystart, ystop, scale,
                                  svc, X_scaler, orient, pix_per_cell,
                                  cell_per_block, colorspace))
    ystart = 400
    ystop = 528
    scale = 2.0
    rectangles.append(find_cars(img, ystart, ystop, scale,
                                  svc, X_scaler, orient, pix_per_cell,
                                  cell_per_block, colorspace))
    ystart = 432
    ystop = 560
    scale = 2.0
    rectangles.append(find_cars(img, ystart, ystop, scale,
                                  svc, X_scaler, orient, pix_per_cell,
                                  cell_per_block, colorspace))
    ystart = 400
    ystop = 596
    scale = 3.5
    rectangles.append(find_cars(img, ystart, ystop, scale,
                                  svc, X_scaler, orient, pix_per_cell,
                                  cell_per_block, colorspace))
    ystart = 464
    ystop = 660
    scale = 3.5
    rectangles.append(find_cars(img, ystart, ystop, scale,
                                  svc, X_scaler, orient, pix_per_cell,
                                  cell_per_block, colorspace))

    # apparently this is the best way to flatten a list of lists
    rectangles = [item for sublist in rectangles for item in sublist]

    out_img, heatmap, labels = apply_heat(img, rectangles)
    return out_img, heatmap, labels


video_path = "project_video.mp4"

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output/output.avi', fourcc, 25.0, (1280, 720), isColor=True)

cap = cv2.VideoCapture(video_path)
cnt = 0
while cap.isOpened():
# for i in range(50):
    ret, frame = cap.read()
    if ret is True:
        result, heat, labels = pipeline(frame)
        out.write(result)
        cnt += 1
        print(cnt)
    else:
        break

#
# files = glob.glob("f:/work/sdc/project4/CarND-Vehicle-Detection/output/vlc/*.png")
#
# for file in files:
#     img = cv2.imread(file)
#     result, heat, labels = pipeline(img)
#
#     fig = plt.figure()
#     plt.subplot(121)
#     plt.imshow(result)
#     plt.title('Car Positions')
#     plt.subplot(122)
#     plt.imshow(heat, cmap='hot')
#     plt.title('Heat Map')
#     fig.tight_layout()

