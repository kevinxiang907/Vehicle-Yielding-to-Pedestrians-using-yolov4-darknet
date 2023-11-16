import cv2
import darknet
import time
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
cfg_file = ["Project_file/Parametrical_cfg/yolo-obj.cfg",
            "Project_file/Parametrical_cfg/yolo-obj-car.cfg",
            "Project_file/Parametrical_cfg/yolo-obj-crosswalk.cfg"]

# 名稱檔
data_file = ["Project_file/Parametrical_name/person.data",
             "Project_file/Parametrical_name/car.data",
             "Project_file/Parametrical_name/crosswalk.data"]

# 權重檔
weight_file = ["Project_file/Parametrical_weight/person_iteration1700_10_21.weights", 
               "Project_file/Parametrical_weight/car_iteration1900.weights", 
               "Project_file/Parametrical_weight/videos2_crosswalk_iteraion1700.weights"]

network = []
class_names = []
width = []
height = []
width_ratio = []
height_ratio = []
thresh = [.5, .5, .5]

#追蹤設定
people = []
car = []
cross = []
wnum =0
orange = []
    
car_tracker_list = []
incrosscar = []
people_tracker_list = []
cross_tracker_list = []
old_people_xy_list = []
new_people_xy_list = []
old_car_xy_list = []
new_car_xy_list = []
cross_xy_list = []
cross_new_xy_list = []
peopler = []
peoplel = []
caru = []
card = []
nomove = []
tracking = False                    # 設定 False 表示尚未開始追蹤
draw_cross = []
trackcount = 1
tracknum = 0
start = time.process_time()
# 載入神經網路
for i in range(len(cfg_file)):
  network_obj, class_names_obj, class_colors_obj = darknet.load_network(cfg_file[i], data_file[i], weight_file[i])

  network.append(network_obj)
  class_names.append(class_names_obj)

  # 獲取神經網路的寬高
  width.append(darknet.network_width(network_obj))
  height.append(darknet.network_height(network_obj))

# 讀取影片
cap = cv2.VideoCapture("Project_file/videos/use_test.mp4")

# 獲取影片的寬高
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = round(cap.get(cv2.CAP_PROP_FPS))

# 儲存影片
out = cv2.VideoWriter('Project_file/videos/video_test_KCF_project1.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))

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
        """
        cv2.rectangle(frame, (crosswalk_xmin, crosswalk_ymin), (crosswalk_xmax, crosswalk_ymax), (255,0,0), 2)     
        cv2.putText(frame, "{} [{:.2f}]".format(label_crosswalk, float(confidence_crosswalk)),
              (crosswalk_xmin, crosswalk_ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
              (255,0,0), 2)
        """     
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
                car.append((car_xmin, car_ymin, car_xmax-car_xmin, car_ymax-car_ymin))
            people.append((person_xmin, person_ymin, person_xmax-person_xmin, person_ymax-person_ymin))
        cross.append((crosswalk_xmin, crosswalk_ymin, crosswalk_xmax, crosswalk_ymax))
        #cross.append((crosswalk_xmin, crosswalk_ymin, crosswalk_xmax-crosswalk_xmin, crosswalk_ymax-crosswalk_ymin))
    
    print("draw1")
    if tracking == False:
        for pt in people:
            maxiou = 0
            j=0
            num = -1
            pp1_min = [int(pt[0]),int(pt[1])]
            pp1_max = [int(pt[0]+pt[2]),int(pt[1]+pt[3])]
            for piou in old_people_xy_list:
              pp2_min = [int(piou[0][0]),int(piou[0][1])]
              pp2_max = [int(piou[0][0]+piou[0][2]),int(piou[0][1]+piou[0][3])]
              Iou = car_crosswalk_overlap_area(pp1_min[0],pp1_min[1],pp1_max[0],pp1_max[1],pp2_min[0],pp2_min[1],pp2_max[0],pp2_max[1])
              if Iou > maxiou:
                maxiou = Iou
                num = j
              j=j+1
            if maxiou > 0.7 and pt[2]*pt[3] < old_people_xy_list[num][0][2]*old_people_xy_list[num][0][3]:
              del people_tracker_list[num]
              del old_people_xy_list[num]
            elif maxiou >0.1:
              continue
            
            tracker = cv2.legacy.TrackerKCF_create()  # 創建追蹤器
            tracker.init(frame,(pt[0],pt[1],pt[2],pt[3]))
            people_tracker_list.append(tracker)
            old_people_xy_list.append([(pt[0],pt[1],pt[2],pt[3]),'non'])

        for ct in car:
            maxiou = 0
            cp1_min = [int(ct[0]),int(ct[1])]
            cp1_max = [int(ct[0]+ct[2]),int(ct[1]+ct[3])]
            j = 0
            num = -1
            for ciou in old_car_xy_list:
              cp2_min = [int(ciou[0][0]),int(ciou[0][1])]
              cp2_max = [int(ciou[0][0]+ciou[0][2]),int(ciou[0][1]+ciou[0][3])]
              Iou = car_crosswalk_overlap_area(cp1_min[0],cp1_min[1],cp1_max[0],cp1_max[1],cp2_min[0],cp2_min[1],cp2_max[0],cp2_max[1])
              if Iou > maxiou:
                maxiou = Iou
                num = j
              j=j+1
            if maxiou > 0.7 and pt[2]*pt[3] < old_car_xy_list[num][0][2]*old_car_xy_list[num][0][3]:
              del old_car_xy_list[num]
              del car_tracker_list[num]
            elif maxiou >0.1:
              continue
            
            tracker = cv2.legacy.TrackerKCF_create()  # 創建追蹤器
            tracker.init(frame,(ct[0],ct[1],ct[2],ct[3]))
            car_tracker_list.append((tracker,trackcount))
            old_car_xy_list.append([(ct[0],ct[1],ct[2],ct[3]),'non'])
            trackcount = trackcount +1


        """
        for cross_t in cross:
            tracker = cv2.legacy.TrackerKCF_create()  # 創建追蹤器
            tracker.init(frame,(cross_t[0],cross_t[1],cross_t[2],cross_t[3]))
            cross_tracker_list.append(tracker)
            cross_xy_list.append((cross_t[0],cross_t[1],cross_t[2],cross_t[3]))
        """
        people.clear()
        car.clear()
        cross.clear()
        tracking = True
        print("no track")
    else:
        if len(people_tracker_list)!=len(old_people_xy_list):
          print("people list num error1")
          print(len(people_tracker_list),len(old_people_xy_list))
        if len(car_tracker_list)!= len(old_car_xy_list):
          print("car list num error1")
          print(len(car_tracker_list),len(old_car_xy_list))
        j = 0
        """
        for i in range(len(cross_xy_list)):
          
          success, point = cross_tracker_list[i-j].update(frame)
          if success == False:
                del cross_tracker_list[i-j]
                j=j+1
          else:
              p1 = [int(point[0]), int(point[1])]
              p2 = [int(point[0] + point[2]), int(point[1] + point[3])]
              x_move = int(point[0] - cross_xy_list[i][0] )
              y_move = int(cross_xy_list[i][1] - point[1])
              draw_cross.append((p1,p2))
              cross_new_xy_list.append(point[:]) 
        """
        j = 0
        print("ptrack")
        peopledelete = []
        for i in range(len(people_tracker_list)):
            success, point = people_tracker_list[i].update(frame)
            if success == False:
                peopledelete.append(i)
                #del people_tracker_list[i-j]
                #j=j+1
            else:
                p1 = [int(point[0]), int(point[1])]
                p2 = [int(point[0] + point[2]), int(point[1] + point[3])]
                npp = int(point[0] + point[2]/2)
                opp = int(old_people_xy_list[i][0][0]+old_people_xy_list[i][0][2]/2)
                #path = npp-opp+x_move
                path = npp-opp
                #orange.append((p1, p2))
                
                
                if path > 0.00:
                    new_people_xy_list.append([(point[:]),'r'])
                    peopler.append((p1,p2))
                elif path < 0.00:
                    new_people_xy_list.append([(point[:]),'l'])
                    peoplel.append((p1,p2))
                else:
                    new_people_xy_list.append([(point[:]),'non'])
                    #nomove.append((p1,p2))
        for dele in range(len(peopledelete)-1,-1,-1):
          del people_tracker_list[dele]
        peopledelete.clear()
        print("p path")
        j = 0
        print("ctrack")
        cardelete = []
        for i in range(len(car_tracker_list)):
            success, point = car_tracker_list[i][0].update(frame)
            if success == False:
                cardelete.append(i)
                #del car_tracker_list[i-j]
                #j=j+1
            else:
                p1 = [int(point[0]), int(point[1])]
                p2 = [int(point[0] + point[2]), int(point[1] + point[3])]
                npp = (point[1] +(point[3]/2))
                opp = (old_car_xy_list[i][0][1]+(old_car_xy_list[i][0][3]/2))
                #path = npp-opp+y_move
                path = npp-opp
                
                #orange.append((p1, p2))
                if path > 0.00:
                    new_car_xy_list.append([(point[:]),'d',car_tracker_list[i][1]])
                    card.append((p1,p2,car_tracker_list[i][1]))
                elif path < 0.00:
                    new_car_xy_list.append([(point[:]),'u',car_tracker_list[i][1]])
                    caru.append((p1,p2,car_tracker_list[i][1]))
                elif path == 0.00:
                    new_car_xy_list.append([(point[:]),'non',car_tracker_list[i][1]])
                    nomove.append((p1,p2,car_tracker_list[i][1]))
                else:
                  print("error")
        for dele in range(len(cardelete)-1,-1,-1):
          del car_tracker_list[dele]
        cardelete.clear()
        print("c path")
        
        for i in range(len(new_people_xy_list)):
            px = new_people_xy_list[i][0][0]
            ppath = new_people_xy_list[i][1]
            if ppath == 'r':
                for j in range(len(new_car_xy_list)):
                    cx = new_car_xy_list[j][0][0]
                    cpath = new_car_xy_list[j][1]
                    #if cx > px:
                    if cx < px:
                           if cpath != 'non':
                                cp1 = [int(new_car_xy_list[j][0][0]), int(new_car_xy_list[j][0][1])]
                                cp2 = [int(new_car_xy_list[j][0][0] + new_car_xy_list[j][0][2]), int(new_car_xy_list[j][0][1] + new_car_xy_list[j][0][3])] 
                                for s in range(len(cross)):
                                    IOU = car_crosswalk_overlap_area(cp1[0],cp1[1],cp2[0],cp2[1],cross[s][0],cross[s][1],cross[s][2],cross[s][3])
                                    #IOU = car_crosswalk_overlap_area(cp1[0],cp1[1],cp2[0],cp2[1],cross[s][0],cross[s][1],cross[s][0]+cross[s][2],cross[s][1]+cross[s][3])
                                    
                                    if (IOU > 0.5):
                                      if (cp1,cp2) not in orange:
                                        orange.append((cp1, cp2,new_car_xy_list[j][2]))
                                        #print(IOU,end='2 ')
                                      #pass
                                      
                                      #cv2.rectangle(frame, cp1, cp2, (17,85,255), 3)
            elif ppath == 'l':
                for j in range(len(new_car_xy_list)):
                    cx = new_car_xy_list[j][0][0]
                    cpath = new_car_xy_list[j][1]
                    #if cx < px:
                    if cx > px:
                       if cpath != 'non':
                            cp1 = [int(new_car_xy_list[j][0][0]), int(new_car_xy_list[j][0][1])]
                            cp2 = [int(new_car_xy_list[j][0][0] + new_car_xy_list[j][0][2]), int(new_car_xy_list[j][0][1] + new_car_xy_list[j][0][3])] 
                            for s in range(len(cross)):
                                lIou = car_crosswalk_overlap_area(cp1[0],cp1[1],cp2[0],cp2[1],cross[s][0],cross[s][1],cross[s][2],cross[s][3])
                                #lIou = car_crosswalk_overlap_area(cp1[0],cp1[1],cp2[0],cp2[1],cross[s][0],cross[s][1],cross[s][0]+cross[s][2],cross[s][1]+cross[s][3])
                                if (lIou > 0.5):
                                  if (cp1,cp2) not in orange:
                                    orange.append((cp1, cp2,new_car_xy_list[j][2]))
                                    #print(lIou,end='1 ')
                                  #pass
                                  
                                  #cv2.rectangle(frame, cp1, cp2, (17,85,255), 3)
            
        old_car_xy_list.clear()
        old_people_xy_list.clear()
        cross_xy_list.clear()
        old_car_xy_list = new_car_xy_list.copy()
        old_people_xy_list = new_people_xy_list.copy()
        cross_xy_list = cross_new_xy_list.copy()
        new_car_xy_list.clear()
        new_people_xy_list.clear()
        cross_new_xy_list.clear()
        if len(people_tracker_list)!=len(old_people_xy_list):
          print("people list num error2")
          print(len(people_tracker_list),len(old_people_xy_list))
        if len(car_tracker_list)!= len(old_car_xy_list):
          print("car list num error2")
          print(len(car_tracker_list),len(old_car_xy_list))


        for pt in people:
            #print("pt")
            maxiou = 0
            j=0
            num = -1
            pp1_min = [int(pt[0]),int(pt[1])]
            pp1_max = [int(pt[0]+pt[2]),int(pt[1]+pt[3])]
            for piou in old_people_xy_list:
              pp2_min = [int(piou[0][0]),int(piou[0][1])]
              pp2_max = [int(piou[0][0]+piou[0][2]),int(piou[0][1]+piou[0][3])]
              Iou = car_crosswalk_overlap_area(pp1_min[0],pp1_min[1],pp1_max[0],pp1_max[1],pp2_min[0],pp2_min[1],pp2_max[0],pp2_max[1])
              if Iou > maxiou:
                maxiou = Iou
                num = j
              j=j+1
            if maxiou > 0.7 and pt[2]*pt[3] < old_people_xy_list[num][0][2]*old_people_xy_list[num][0][3]:
              del people_tracker_list[num]
              del old_people_xy_list[num]
            elif maxiou >0.1:
              continue
            
            tracker = cv2.legacy.TrackerKCF_create()  # 創建追蹤器
            tracker.init(frame,(pt[0],pt[1],pt[2],pt[3]))
            people_tracker_list.append(tracker)
            old_people_xy_list.append([(pt[0],pt[1],pt[2],pt[3]),'non'])

        n = 1
        for ct in car:
            #print("ct")
            maxiou = 0
            cp1_min = [int(ct[0]),int(ct[1])]
            cp1_max = [int(ct[0]+ct[2]),int(ct[1]+ct[3])]
            j = 0
            num = -1
            for ciou in old_car_xy_list:
              cp2_min = [int(ciou[0][0]),int(ciou[0][1])]
              cp2_max = [int(ciou[0][0]+ciou[0][2]),int(ciou[0][1]+ciou[0][3])]
              Iou = car_crosswalk_overlap_area(cp1_min[0],cp1_min[1],cp1_max[0],cp1_max[1],cp2_min[0],cp2_min[1],cp2_max[0],cp2_max[1])
              if Iou > maxiou:
                maxiou = Iou
                num = j
              j=j+1
            """
            if maxiou > 0.7 and ct[2]*ct[3] < old_car_xy_list[num][0][2]*old_car_xy_list[num][0][3]:
              del old_car_xy_list[num]
              del car_tracker_list[num]
            """
            if maxiou >0.1:
              continue
            
            tracker = cv2.legacy.TrackerKCF_create()  # 創建追蹤器
            tracker.init(frame,(ct[0],ct[1],ct[2],ct[3]))
            car_tracker_list.append((tracker,trackcount))
            old_car_xy_list.append([(ct[0],ct[1],ct[2],ct[3]),'non'])
            trackcount = trackcount +1
        """
        for cross_t in cross:
          #print("ct")
            maxiou = 0
            cp1_min = [int(cross_t[0]),int(cross_t[1])]
            cp1_max = [int(cross_t[0]+cross_t[2]),int(cross_t[1]+cross_t[3])]
            j = 0
            num = -1
            for cross_iou in cross_xy_list:
              cp2_min = [int(cross_iou[0]),int(cross_iou[1])]
              cp2_max = [int(cross_iou[0]+cross_iou[2]),int(cross_iou[1]+cross_iou[3])]
              Iou = car_crosswalk_overlap_area(cp1_min[0],cp1_min[1],cp1_max[0],cp1_max[1],cp2_min[0],cp2_min[1],cp2_max[0],cp2_max[1])
              if Iou > maxiou:
                maxiou = Iou
                num = j
              j=j+1
            if maxiou > 0.8 and pt[2]*pt[3] < cross_xy_list[num][2]*cross_xy_list[num][3]:
              del cross_xy_list[num]
              del cross_tracker_list[num]
            elif maxiou >0.1:
              continue
            
            #if maxiou > 0.1:
              #print("maxiou: ",maxiou)
            #  continue
            #print("create track")
            tracker = cv2.legacy.TrackerKCF_create()  # 創建追蹤器
            tracker.init(frame,(cross_t[0],cross_t[1],cross_t[2],cross_t[3]))
            cross_tracker_list.append(tracker)
            cross_xy_list.append((cross_t[0],cross_t[1],cross_t[2],cross_t[3]))
        """
        
        people.clear()
        car.clear()
        cross.clear()
        print("clear")
    
    
    """
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
    """
    """
    for i in range(len(cross)):
        cv2.rectangle(frame, (cross[i][0], cross[i][1]),(cross[i][0]+cross[i][2],cross[i][1]+cross[i][3]), (17,85,255), 3)
    """

    """
    for i in range(len(orange)):
        cv2.rectangle(frame, orange[i][0], orange[i][1], (17,85,255), 3)
        textx = (orange[i][0][0]+orange[i][1][0])//2
        texty = (orange[i][0][1]+orange[i][1][1])//2
        cv2.putText(frame, "{}".format(orange[i][2]),
              (textx, texty), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
              (0,0,255), 3)
    """  
    """
    for i in range(len(peoplel)):
        cv2.rectangle(frame, peoplel[i][0], peoplel[i][1], (0,0,255), 3)
    for i in range(len(peopler)):
        cv2.rectangle(frame, peopler[i][0], peopler[i][1], (255,255,0), 3)
    """
    
    for i in range(len(caru)):
        cv2.rectangle(frame, caru[i][0], caru[i][1], (0,255,0), 3)
        textx = (caru[i][0][0]+caru[i][1][0])//2
        texty = (caru[i][0][1]+caru[i][1][1])//2
        cv2.putText(frame, "{}".format(caru[i][2]),
              (textx, texty), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
              (0,0,255), 3)
    for i in range(len(card)):
        cv2.rectangle(frame, card[i][0], card[i][1], (255,0,255), 3)
        textx = (card[i][0][0]+card[i][1][0])//2
        texty = (card[i][0][1]+card[i][1][1])//2
        cv2.putText(frame, "{}".format(card[i][2]),
              (textx, texty), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
              (0,0,255), 3)
    
    """
    for i in range(len(draw_cross)):
        cv2.rectangle(frame, draw_cross[i][0], draw_cross[i][1], (0, 229, 255), 3)
    """
    
    for i in range(len(nomove)):
        cv2.rectangle(frame, nomove[i][0], nomove[i][1], (128,128,128), 3)
        textx = (nomove[i][0][0]+nomove[i][1][0])//2
        texty = (nomove[i][0][1]+nomove[i][1][1])//2
        cv2.putText(frame, "{}".format(nomove[i][2]),
              (textx, texty), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
              (0,0,255), 3)
    
    if len(people_tracker_list)!=len(old_people_xy_list):
      print("people list num error3")
      print(len(people_tracker_list),len(old_people_xy_list))
    if len(car_tracker_list)!= len(old_car_xy_list):
      print("car list num error3")
      print(len(car_tracker_list),len(old_car_xy_list))
    
    draw_cross.clear()
    orange.clear()
    peoplel.clear()
    peopler.clear()

    caru.clear()
    card.clear()
    nomove.clear()
    out.write(frame)
    wnum = wnum +1
    print(f"video write{wnum}")
cap.release()
out.release()
cv2.destroyAllWindows()
end = time.process_time()
sec = int(end - start)%60
mine = (int(end-start)//60)%60
hour = (int(end-start)//3600)%60
persec = (int(end-start)//wnum)%60
permin = (int(end-start)//wnum//60)%60
perhour = (int(end-start)//wnum//3600)%60
print("執行時間:")
print(">",hour,"hour ",mine,"min ",sec,"sec")
print("平均每張花費時間 :",perhour,"hour ",permin,"min ",persec,"sec")