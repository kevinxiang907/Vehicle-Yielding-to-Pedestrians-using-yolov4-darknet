import cv2
import darknet

# image = cv2.imread("Project_file/image/videos3.jpg")

# cfg_file = "Project_file/Parametrical_cfg/yolo-obj.cfg"
# data_file = "Project_file/Parametrical_name/crosswalk.data"
# weight_file = "Project_file/Parametrical_weight/videos2_crosswalk_iteraion1700.weights"

# network, class_names, class_colors = darknet.load_network(cfg_file, data_file, weight_file, batch_size=1)
# width = darknet.network_width(network)
# height = darknet.network_height(network)

# darknet_image = darknet.make_image(width, height, 3)
# img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# img_resized = cv2.resize(img_rgb, (width, height),interpolation=cv2.INTER_LINEAR)
# img_height, img_width= image.shape[:2]
# width_ratio = img_width/width
# height_ratio = img_height/height
# darknet.copy_image_from_bytes(darknet_image, img_resized.tobytes())
# detections = darknet.detect_image(network, class_names, darknet_image)
# darknet.print_detections(detections, True)
# darknet.free_image(darknet_image)

# for label, confidence, bbox in detections:
#   left, top, right, bottom = darknet.bbox2points(bbox)
#   left, top, right, bottom = int(left * width_ratio), int(top * height_ratio), int(right * width_ratio), int(bottom * height_ratio)
#   # print(width, height)
#   # print(left)
#   # print(top)
#   # print(right)
#   # print(bottom)
#   cv2.rectangle(image, (left, top), (right, bottom), (0,0,255), 2)
#   cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
#             (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
#             (0,0,255), 5)        
# cv2.imwrite('Project_file/image/videos3_test.jpg', image)
# =================================================
# # 神經網路辨識出的座標轉換成圖片大小
# def frame_size(bbox, width_ratio, height_ratio):
#     xmin, ymin, xmax, ymax = darknet.bbox2points(bbox) 
#     xmin, ymin, xmax, ymax = int(xmin * width_ratio), int(ymin * height_ratio), int(xmax * width_ratio), int(ymax * height_ratio)
#     return xmin, ymin, xmax, ymax

# def car_crosswalk_overlap_area(car_xmin, car_ymin, car_xmax, car_ymax, crosswalk_xmin, crosswalk_ymin, crosswalk_xmax, crosswalk_ymax):
#     overlap_x1 = max(car_xmin, crosswalk_xmin)
#     overlap_y1 = max(car_ymin, crosswalk_ymin)
#     overlap_x2 = min(car_xmax, crosswalk_xmax)
#     overlap_y2 = min(car_ymax, crosswalk_ymax)
#     if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
#         car_width = car_xmax - car_xmin
#         car_height = car_ymax - car_ymin
#         width = overlap_x2 - overlap_x1
#         height = overlap_y2 - overlap_y1
#         car_iou = (width * height) / (car_width * car_height)
#         return car_iou
#     else:
#         return 0

# image = cv2.imread("Project_file/image/videos3.jpg")

# # 參數設定
# cfg_file = "Project_file/Parametrical_cfg/yolo-obj.cfg"
# # 名稱檔
# data_file_person = "Project_file/Parametrical_name/person.data"
# data_file_car = "Project_file/Parametrical_name/car.data"
# data_file_crosswalk = "Project_file/Parametrical_name/crosswalk.data"
# # 權重檔
# weight_file_person = "Project_file/Parametrical_weight/person_iteration1700_10_10.weights"
# weight_file_car = "Project_file/Parametrical_weight/car_iteration1900.weights"
# weight_file_crosswalk = "Project_file/Parametrical_weight/videos2_crosswalk_iteraion1700.weights"

# # 載入神經網路
# network_person, class_names_person, class_colors_person = darknet.load_network(cfg_file, data_file_person, weight_file_person)
# network_car, class_names_car, class_colors_car = darknet.load_network(cfg_file, data_file_car, weight_file_car)
# network_crosswalk, class_names_crosswalk, class_colors_crosswalk = darknet.load_network(cfg_file, data_file_crosswalk, weight_file_crosswalk)

# # 獲取神經網路的寬高
# width = darknet.network_width(network_person)     #network width  416
# height = darknet.network_height(network_person)   #network height 416
# # 獲取圖片的寬高
# img_height, img_width= image.shape[:2]      #img height and width
# # 計算神經網路大小與圖片大小的比值
# width_ratio = img_width/width               #use to chaging network size to img size
# height_ratio = img_height/height

# # opencv格式轉換成神經網路格式
# darknet_image = darknet.make_image(width, height, 3)
# img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# img_resized = cv2.resize(img_rgb, (width, height),interpolation=cv2.INTER_LINEAR)
# darknet.copy_image_from_bytes(darknet_image, img_resized.tobytes())

# detections_person = darknet.detect_image(network_person, class_names_person, darknet_image, thresh=.5)
# detections_car = darknet.detect_image(network_car, class_names_car, darknet_image, thresh=.5)
# detections_crosswalk = darknet.detect_image(network_crosswalk, class_names_crosswalk, darknet_image, thresh=.5)
# # 在終端機快速確認效果
# darknet.print_detections(detections_person, True)
# darknet.print_detections(detections_car, True)
# darknet.print_detections(detections_crosswalk, True)

# darknet.free_image(darknet_image)

# # 分類list
# green = []
# yellow = []
# red = []

# for label1, confidence1, bbox1 in detections_crosswalk:
#     crosswalk_xmin, crosswalk_ymin, crosswalk_xmax, crosswalk_ymax, = frame_size(bbox1, width_ratio, height_ratio)
# # =================================================
#     crosswalk_lattice = 12
#     crosswalk_3m = (crosswalk_ymax - crosswalk_ymin) /  crosswalk_lattice*3
# # =================================================
#     cv2.putText(image, "{} [{:.2f}]".format(label1, float(confidence1)),
#           (crosswalk_xmin, crosswalk_ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#           (255,0,0), 2)
    
#     for label2, confidence2, bbox2 in detections_person:
#         person_xmin, person_ymin, person_xmax, person_ymax = frame_size(bbox2, width_ratio, height_ratio)
#         cv2.putText(image, "{} [{:.2f}]".format(label2, float(confidence2)),
#           (person_xmin, person_ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#           (0,0,255), 2)
#         for label3, confidence3, bbox3 in detections_car:
#             car_xmin, car_ymin, car_xmax, car_ymax = frame_size(bbox3, width_ratio, height_ratio)
#             cv2.putText(image, "{} [{:.2f}]".format(label3, float(confidence3)),
#                   (car_xmin, car_ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                   (0,0,255), 2)
#             cv2.rectangle(image, (crosswalk_xmin, crosswalk_ymin), (crosswalk_xmax, crosswalk_ymax), (255,0,0), 2)
            
#             # 綠色 : 人不在斑馬線上
#             if not((crosswalk_xmin < person_xmin < crosswalk_xmax or crosswalk_xmin < person_xmax < crosswalk_xmax) and \
#                     crosswalk_ymin < person_ymax < crosswalk_ymax):
#                 # if person bbox not in green list
#                 if [person_xmin, person_ymin, person_xmax, person_ymax] not in green:
#                     green.append([person_xmin, person_ymin, person_xmax, person_ymax])
#             # 綠色 : 車不在斑馬線上
#             if car_crosswalk_overlap_area(car_xmin, car_ymin, car_xmax, car_ymax, crosswalk_xmin, crosswalk_ymin, crosswalk_xmax, crosswalk_ymax) < 0.3:
#                 # car bbox not in green list
#                 if [car_xmin, car_ymin, car_xmax, car_ymax] not in green:
#                     green.append([car_xmin, car_ymin, car_xmax, car_ymax])
            
#             # 黃色 : 人在斑馬線上，並且斑馬線上沒有車
#             if  ((crosswalk_xmin < person_xmin < crosswalk_xmax or crosswalk_xmin < person_xmax < crosswalk_xmax) and \
#                   crosswalk_ymin < person_ymax < crosswalk_ymax) and \
#                   car_crosswalk_overlap_area(car_xmin, car_ymin, car_xmax, car_ymax, crosswalk_xmin, crosswalk_ymin, crosswalk_xmax, crosswalk_ymax) < 0.25:
#                 # if person bbox not in yellow list
#                 if [person_xmin, person_ymin, person_xmax, person_ymax] not in yellow:
#                     yellow.append([person_xmin, person_ymin, person_xmax, person_ymax])
            
#             #黃色 : 車在斑馬線上，並且斑馬線上沒有人
#             if not((crosswalk_xmin < person_xmin < crosswalk_xmax or crosswalk_xmin < person_xmax < crosswalk_xmax) and \
#                     crosswalk_ymin < person_ymax < crosswalk_ymax) and \
#                     car_crosswalk_overlap_area(car_xmin, car_ymin, car_xmax, car_ymax, crosswalk_xmin, crosswalk_ymin, crosswalk_xmax, crosswalk_ymax) >= 0.3:
#                 # if car bbox not in yellow list
#                 if [car_xmin, car_ymin, car_xmax, car_ymax] not in yellow:
#                     yellow.append([car_xmin, car_ymin, car_xmax, car_ymax])
                    
#             # 人與車同時在斑馬線上
#             if ((crosswalk_xmin < person_xmin < crosswalk_xmax or crosswalk_xmin < person_xmax < crosswalk_xmax) and \
#                  crosswalk_ymin < person_ymax < crosswalk_ymax) and \
#                  car_crosswalk_overlap_area(car_xmin, car_ymin, car_xmax, car_ymax, crosswalk_xmin, crosswalk_ymin, crosswalk_xmax, crosswalk_ymax) >= 0.3:
#                 # if person bbox not in red list
#                   if [person_xmin, person_ymin, person_xmax, person_ymax] not in red:
#                       red.append([person_xmin, person_ymin, person_xmax, person_ymax])
#                 # if car bbox not in red list
#                   if [car_xmin, car_ymin, car_xmax, car_ymax] not in red:
#                       red.append([car_xmin, car_ymin, car_xmax, car_ymax])
            
# for i in range(len(green)):
#     cv2.rectangle(image, (green[i][0], green[i][1]), (green[i][2], green[i][3]), (0,255,0), 2)
# for i in range(len(yellow)):
#     cv2.rectangle(image, (yellow[i][0], yellow[i][1]), (yellow[i][2], yellow[i][3]), (0,255,255), 2)
# for i in range(len(red)):
#     cv2.rectangle(image, (red[i][0], red[i][1]), (red[i][2], red[i][3]), (0,0,255), 2)
          
# cv2.imwrite('Project_file/image/videos3_test.jpg', image)
# =================================================
 
# 神經網路辨識出的座標轉換成圖片大小
def frame_size(bbox, width_ratio, height_ratio):
    xmin, ymin, xmax, ymax = darknet.bbox2points(bbox) 
    xmin, ymin, xmax, ymax = int(xmin * width_ratio), int(ymin * height_ratio), int(xmax * width_ratio), int(ymax * height_ratio)
    return xmin, ymin, xmax, ymax

def car_crosswalk_overlap_area(car_xmin, car_ymin, car_xmax, car_ymax, crosswalk_xmin, crosswalk_ymin, crosswalk_xmax, crosswalk_ymax):
    overlap_x1 = max(car_xmin, crosswalk_xmin)
    overlap_y1 = max(car_ymin, crosswalk_ymin)
    overlap_x2 = min(car_xmax, crosswalk_xmax)
    overlap_y2 = min(car_ymax, crosswalk_ymax)
    if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
        car_width = car_xmax - car_xmin
        car_height = car_ymax - car_ymin
        width = overlap_x2 - overlap_x1
        height = overlap_y2 - overlap_y1
        car_iou = (width * height) / (car_width * car_height)
        return car_iou
    else:
        return 0

# 參數設定
cfg_file = "Project_file/Parametrical_cfg/yolo-obj-old-videos.cfg"
# 名稱檔
data_file_person = "Project_file/Parametrical_name/person.data"
data_file_car = "Project_file/Parametrical_name/car.data"
data_file_crosswalk = "Project_file/Parametrical_name/crosswalk.data"
# 權重檔
weight_file_person = "Project_file/Parametrical_weight/person_iteration1700_10_10.weights"
weight_file_car = "Project_file/Parametrical_weight/car_iteration1900.weights"
weight_file_crosswalk = "Project_file/Parametrical_weight/videos2_crosswalk_iteraion1700.weights"

# 載入神經網路
network_person, class_names_person, class_colors_person = darknet.load_network(cfg_file, data_file_person, weight_file_person)
network_car, class_names_car, class_colors_car = darknet.load_network(cfg_file, data_file_car, weight_file_car)
network_crosswalk, class_names_crosswalk, class_colors_crosswalk = darknet.load_network(cfg_file, data_file_crosswalk, weight_file_crosswalk)

# 獲取神經網路的寬高
width = darknet.network_width(network_person)
height = darknet.network_height(network_person)

cap = cv2.VideoCapture("Project_file/videos/videos2.mp4")
# 獲取影片的寬高
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = round(cap.get(cv2.CAP_PROP_FPS))

# 計算神經網路大小與影片大小的比值
width_ratio = frame_width/width
height_ratio = frame_height/height

fourcc = cv2.VideoWriter_fourcc(*'MP4V') 
out = cv2.VideoWriter('Project_file/videos/videos3_crosswalk384x384_person_car_size832x832_carIOU0.5_person1700_crosswalk1700_car1900_thre50.mp4', fourcc, fps, (frame_width, frame_height))

while (True):
    ret, frame = cap.read()
    if ret == False:
      break
    
    # opencv格式轉換成神經網路格式
    darknet_image = darknet.make_image(width, height, 3)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (width, height),interpolation=cv2.INTER_LINEAR)
    darknet.copy_image_from_bytes(darknet_image, img_resized.tobytes())
    
    detections_crosswalk = darknet.detect_image(network_crosswalk, class_names_crosswalk, darknet_image, thresh=.5)
    detections_person = darknet.detect_image(network_person, class_names_person, darknet_image, thresh=.5)
    detections_car = darknet.detect_image(network_car, class_names_car, darknet_image, thresh=.5)
    
    # 在終端機快速確認效果
    darknet.print_detections(detections_person, True)
    darknet.print_detections(detections_car, True)
    darknet.print_detections(detections_crosswalk, True)
    
    darknet.free_image(darknet_image)
    
    # 分類list
    green = []
    yellow = []
    red = []

    for label1, confidence1, bbox1 in detections_crosswalk:
        crosswalk_xmin, crosswalk_ymin, crosswalk_xmax, crosswalk_ymax, = frame_size(bbox1, width_ratio, height_ratio)
        cv2.rectangle(frame, (crosswalk_xmin, crosswalk_ymin), (crosswalk_xmax, crosswalk_ymax), (255,0,0), 2)        
        cv2.putText(frame, "{} [{:.2f}]".format(label1, float(confidence1)),
              (crosswalk_xmin, crosswalk_ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
              (255,0,0), 2)
        for label2, confidence2, bbox2 in detections_person:
            person_xmin, person_ymin, person_xmax, person_ymax = frame_size(bbox2, width_ratio, height_ratio)
            cv2.putText(frame, "{} [{:.2f}]".format(label2, float(confidence2)),
                  (person_xmin, person_ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                  (0,0,255), 2)
            for label3, confidence3, bbox3 in detections_car:
                car_xmin, car_ymin, car_xmax, car_ymax = frame_size(bbox3, width_ratio, height_ratio)
                
                
                
                cv2.putText(frame, "{} [{:.2f}]".format(label3, float(confidence3)),
                    (car_xmin, car_ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,0,255), 2)
                
                # 綠色 : 人不在斑馬線上
                if not((crosswalk_xmin < person_xmin < crosswalk_xmax or crosswalk_xmin < person_xmax < crosswalk_xmax) and \
                        crosswalk_ymin < person_ymax < crosswalk_ymax):
                    # if person bbox not in green list
                    if [person_xmin, person_ymin, person_xmax, person_ymax] not in green:
                        green.append([person_xmin, person_ymin, person_xmax, person_ymax])
                # 綠色 : 車不在斑馬線上
                if car_crosswalk_overlap_area(car_xmin, car_ymin, car_xmax, car_ymax, crosswalk_xmin, crosswalk_ymin, crosswalk_xmax, crosswalk_ymax) < 0.5:
                    # car bbox not in green list
                    if [car_xmin, car_ymin, car_xmax, car_ymax] not in green:
                        green.append([car_xmin, car_ymin, car_xmax, car_ymax])
                
                # 黃色 : 人在斑馬線上，並且斑馬線上沒有車
                if  ((crosswalk_xmin < person_xmin < crosswalk_xmax or crosswalk_xmin < person_xmax < crosswalk_xmax) and \
                      crosswalk_ymin < person_ymax < crosswalk_ymax) and \
                      car_crosswalk_overlap_area(car_xmin, car_ymin, car_xmax, car_ymax, crosswalk_xmin, crosswalk_ymin, crosswalk_xmax, crosswalk_ymax) < 0.5:
                    # if person bbox not in yellow list
                    if [person_xmin, person_ymin, person_xmax, person_ymax] not in yellow:
                        yellow.append([person_xmin, person_ymin, person_xmax, person_ymax])
                
                #黃色 : 車在斑馬線上，並且斑馬線上沒有人
                if not((crosswalk_xmin < person_xmin < crosswalk_xmax or crosswalk_xmin < person_xmax < crosswalk_xmax) and \
                        crosswalk_ymin < person_ymax < crosswalk_ymax) and \
                        car_crosswalk_overlap_area(car_xmin, car_ymin, car_xmax, car_ymax, crosswalk_xmin, crosswalk_ymin, crosswalk_xmax, crosswalk_ymax) >= 0.5:
                    # if car bbox not in yellow list
                    if [car_xmin, car_ymin, car_xmax, car_ymax] not in yellow:
                        yellow.append([car_xmin, car_ymin, car_xmax, car_ymax])
                        
                # 人與車同時在斑馬線上
                if ((crosswalk_xmin < person_xmin < crosswalk_xmax or crosswalk_xmin < person_xmax < crosswalk_xmax) and \
                     crosswalk_ymin < person_ymax < crosswalk_ymax) and \
                     car_crosswalk_overlap_area(car_xmin, car_ymin, car_xmax, car_ymax, crosswalk_xmin, crosswalk_ymin, crosswalk_xmax, crosswalk_ymax) >= 0.5:
                    # if person bbox not in red list
                      if [person_xmin, person_ymin, person_xmax, person_ymax] not in red:
                          red.append([person_xmin, person_ymin, person_xmax, person_ymax])
                    # if car bbox not in red list
                      if [car_xmin, car_ymin, car_xmax, car_ymax] not in red:
                          red.append([car_xmin, car_ymin, car_xmax, car_ymax])
                
    for i in range(len(green)):
        cv2.rectangle(frame, (green[i][0], green[i][1]), (green[i][2], green[i][3]), (0,255,0), 2)
    for i in range(len(yellow)):
        cv2.rectangle(frame, (yellow[i][0], yellow[i][1]), (yellow[i][2], yellow[i][3]), (0,255,255), 2)
    for i in range(len(red)):
        cv2.rectangle(frame, (red[i][0], red[i][1]), (red[i][2], red[i][3]), (0,0,255), 2)
    
    out.write(frame)
cap.release()
out.release()
cv2.destroyAllWindows()
# =================================================
# from darknet import *

# cfg_file = "cfg/yolo-obj.cfg"
# data_file = "data/obj.data"
# weight_file = "backup/car1900.weights"

# network, class_names, class_colors = darknet.load_network(cfg_file, data_file, weight_file, batch_size=1)
# width = darknet.network_width(network)
# height = darknet.network_height(network)
 
# image = cv2.imread("data/car_test.png")

# darknet_image = darknet.make_image(width, height, 3)
# img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# img_resized = cv2.resize(img_rgb, (width, height),interpolation=cv2.INTER_LINEAR)
# img_height, img_width= image.shape[:2]
# width_ratio = img_width/width
# height_ratio = img_height/height
# copy_image_from_bytes(darknet_image, img_resized.tobytes())
# detections = darknet.detect_image(network, class_names, darknet_image)
# darknet.print_detections(detections, True)
# free_image(darknet_image)

# for label, confidence, bbox in detections:
#   left, top, right, bottom = bbox2points(bbox)
#   left, top, right, bottom = int(left * width_ratio), int(top * height_ratio), int(right * width_ratio), int(bottom * height_ratio)
#   cv2.rectangle(image, (left, top), (right, bottom), (0,0,255), 2)
        
# cv2.imwrite('output.jpg', image)
# =================================================
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Cannot receive frame")
#         break
