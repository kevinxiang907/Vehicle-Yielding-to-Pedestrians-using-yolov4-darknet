import cv2
import darknet

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
        
def mosaic(frame, xmin, ymin, xmax, ymax):
    mosaic_area = frame[ymin:ymax, xmin:xmax]
    mosaic_area = cv2.blur(mosaic_area, (5, 5)) 
    frame[ymin:ymax, xmin:xmax] = mosaic_area
    return frame

# 參數設定
cfg_file = ["cfg/yolo-obj.cfg", "cfg/yolo-obj-car.cfg", "cfg/yolo-obj-crosswalk.cfg"]

# 名稱檔
data_file = ["data/obj_person.data", "data/obj_car.data", "data/obj_crosswalk.data"]

# 權重檔
weight_file = ["backup/person_iteration1700_10_10.weights", "backup/car_iteration1900.weights", "backup/videos8_crosswalk_iteraion1600_10_17.weights"]

network = []
class_names = []
width = []
height = []
width_ratio = []
height_ratio = []
thresh = [.5, .5, .5]

# 載入神經網路
for i in range(len(cfg_file)):
  network_obj, class_names_obj, class_colors_obj = darknet.load_network(cfg_file[i], data_file[i], weight_file[i])

  network.append(network_obj)
  class_names.append(class_names_obj)

  # 獲取神經網路的寬高
  width.append(darknet.network_width(network_obj))
  height.append(darknet.network_height(network_obj))

# 讀取影片
cap = cv2.VideoCapture("data/videos/videos81.mp4")

# 獲取影片的寬高
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = round(cap.get(cv2.CAP_PROP_FPS))

# 儲存影片
out = cv2.VideoWriter('data/videos/videos81_oldperson832_car832_crosswalk608.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))

# 計算神經網路大小與影片大小的比值
for i in range(len(width)):
  width_ratio.append(frame_width/width[i])
  height_ratio.append(frame_height/height[i])

while (True):
    ret, frame = cap.read()
    if ret == False:
      break
    
    # 分類list
    green = []
    yellow = []
    red = []

    # opencv格式轉換成神經網路格式
    detections = []
    for i in range(3):
      darknet_image = darknet.make_image(width[i], height[i], 3)
      img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      img_resized = cv2.resize(img_rgb, (width[i], height[i]),interpolation=cv2.INTER_LINEAR)

      darknet.copy_image_from_bytes(darknet_image, img_resized.tobytes())
      detections.append(darknet.detect_image(network[i], class_names[i], darknet_image, thresh[i]))
      darknet.print_detections(detections[i], True)
      darknet.free_image(darknet_image)

    for label_crosswalk, confidence_crosswalk, bbox_crosswalk in detections[2]:
        crosswalk_xmin, crosswalk_ymin, crosswalk_xmax, crosswalk_ymax, = frame_size(bbox_crosswalk, width_ratio[2], height_ratio[2])
        cv2.rectangle(frame, (crosswalk_xmin, crosswalk_ymin), (crosswalk_xmax, crosswalk_ymax), (255,0,0), 2)     
        cv2.putText(frame, "{} [{:.2f}]".format(label_crosswalk, float(confidence_crosswalk)),
              (crosswalk_xmin, crosswalk_ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
              (255,0,0), 2)
              
        for label_person, confidence_person, bbox_person in detections[0]:
            person_xmin, person_ymin, person_xmax, person_ymax = frame_size(bbox_person, width_ratio[0], height_ratio[0])
            
            if person_xmin>0 and person_ymin>0 and person_xmax>0 and person_ymax>0:
                frame = mosaic(frame, person_xmin, person_ymin, person_xmax, person_ymax)
            
            for label_car, confidence_car, bbox_car in detections[1]:
                car_xmin, car_ymin, car_xmax, car_ymax = frame_size(bbox_car, width_ratio[1], height_ratio[1])
                
                # 綠色 : 人不在斑馬線上
                if not((crosswalk_xmin < person_xmin < crosswalk_xmax or crosswalk_xmin < person_xmax < crosswalk_xmax) and \
                        crosswalk_ymin < person_ymax < crosswalk_ymax):
                    # if person bbox not in green list
                    if [person_xmin, person_ymin, person_xmax, person_ymax] not in green:
                        green.append([person_xmin, person_ymin, person_xmax, person_ymax, label_person, confidence_person])
                # 綠色 : 車不在斑馬線上
                if car_crosswalk_overlap_area(car_xmin, car_ymin, car_xmax, car_ymax, crosswalk_xmin, crosswalk_ymin, crosswalk_xmax, crosswalk_ymax) < 0.5:
                    # car bbox not in green list
                    if [car_xmin, car_ymin, car_xmax, car_ymax] not in green:
                        green.append([car_xmin, car_ymin, car_xmax, car_ymax, label_car, confidence_car])
                
                # 黃色 : 人在斑馬線上，並且斑馬線上沒有車
                if  ((crosswalk_xmin < person_xmin < crosswalk_xmax or crosswalk_xmin < person_xmax < crosswalk_xmax) and \
                      crosswalk_ymin < person_ymax < crosswalk_ymax) and \
                      car_crosswalk_overlap_area(car_xmin, car_ymin, car_xmax, car_ymax, crosswalk_xmin, crosswalk_ymin, crosswalk_xmax, crosswalk_ymax) < 0.5:
                    # if person bbox not in yellow list
                    if [person_xmin, person_ymin, person_xmax, person_ymax] not in yellow:
                        yellow.append([person_xmin, person_ymin, person_xmax, person_ymax, label_person, confidence_person])
                
                #黃色 : 車在斑馬線上，並且斑馬線上沒有人
                if not((crosswalk_xmin < person_xmin < crosswalk_xmax or crosswalk_xmin < person_xmax < crosswalk_xmax) and \
                        crosswalk_ymin < person_ymax < crosswalk_ymax) and \
                        car_crosswalk_overlap_area(car_xmin, car_ymin, car_xmax, car_ymax, crosswalk_xmin, crosswalk_ymin, crosswalk_xmax, crosswalk_ymax) >= 0.5:
                    # if car bbox not in yellow list
                    if [car_xmin, car_ymin, car_xmax, car_ymax] not in yellow:
                        yellow.append([car_xmin, car_ymin, car_xmax, car_ymax, label_car, confidence_car])
                        
                # 人與車同時在斑馬線上
                if ((crosswalk_xmin < person_xmin < crosswalk_xmax or crosswalk_xmin < person_xmax < crosswalk_xmax) and \
                     crosswalk_ymin < person_ymax < crosswalk_ymax) and \
                     car_crosswalk_overlap_area(car_xmin, car_ymin, car_xmax, car_ymax, crosswalk_xmin, crosswalk_ymin, crosswalk_xmax, crosswalk_ymax) >= 0.5:
                    # if person bbox not in red list
                      if [person_xmin, person_ymin, person_xmax, person_ymax] not in red:
                          red.append([person_xmin, person_ymin, person_xmax, person_ymax, label_person, confidence_person])
                    # if car bbox not in red list
                      if [car_xmin, car_ymin, car_xmax, car_ymax] not in red:
                          red.append([car_xmin, car_ymin, car_xmax, car_ymax, label_car, confidence_car])
                
    for i in range(len(green)):
        cv2.rectangle(frame, (green[i][0], green[i][1]), (green[i][2], green[i][3]), (0,255,0), 2)
        cv2.putText(frame, "{} [{:.2f}]".format(green[i][4], float(green[i][5])),
                  (green[i][0], green[i][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                  (0,255,0), 2)
    for i in range(len(yellow)):
        cv2.rectangle(frame, (yellow[i][0], yellow[i][1]), (yellow[i][2], yellow[i][3]), (0,255,255), 2)
        cv2.putText(frame, "{} [{:.2f}]".format(yellow[i][4], float(yellow[i][5])),
                  (yellow[i][0], yellow[i][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                  (0,255,255), 2)
    for i in range(len(red)):
        cv2.rectangle(frame, (red[i][0], red[i][1]), (red[i][2], red[i][3]), (0,0,255), 2)
        cv2.putText(frame, "{} [{:.2f}]".format(red[i][4], float(red[i][5])),
                  (red[i][0], red[i][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                  (0,0,255), 2)
    
    out.write(frame)
cap.release()
out.release()
cv2.destroyAllWindows()
