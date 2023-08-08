"""
Test code for analysis.....
Whats New????
0. json renamed from counts to timestamp
1. Json writing brought to single image.
2. image save size increased by 20%
3. Non severe/Severe Defect dict settings updated
   7-sept-2022
4. execution log name updated with date
5. network folders updated
6. Unwanted console messages suppresed
    8-sept-2022
7. Updated for camera 4 and 5
8. All defects of cam4 are non severe
9. code cleaned
10. error log written to crash file day-wise.
   9-sept-2022
11. error logs moved to folder
12. image count logic updated
13. camera settings and variables globalised
    14-april-2023
14. implementing zmq for communication between scripts on 14 april 2023
    8-may-2023
15. added remote control functionality so that it can be started remotely from master jetson 
                                             """
import json
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time
import base64
import cv2
from datetime import datetime, timedelta,date
import matplotlib
from pycomm3 import SLCDriver
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.general import (check_img_size, check_requirements, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
import threading
import neoapi
import zmq

from pathlib import Path
torch.cuda.empty_cache()
print("Hello World")
try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--configfile', type=str)
    parser.add_argument('--cam4', type=int, default=0)
    parser.add_argument('--cam5', type=int, default=0)
    opt = parser.parse_args()
    
    with open(opt.configfile,mode='r') as data:
        config_file=json.load(data)
    json_data=[]
    print("Inside Try")
    
    ### for cam4 #########
    camera_no='cam4'
    brand_name=Path(opt.configfile).stem
    inspection_system_id=config_file['inspection_system_id']+camera_no
    inspection_station_id=config_file['inspection_station_id']
    camera_id=config_file[camera_no]['camera_id']
    # #camera_ip=config_file[camera_no]['#camera_ip']
    camera_features=config_file[camera_no]['camera_features']
    severe_defect_dict=config_file[camera_no]['defect_list']['severe_defect_dict']
    print("Before Rejection List")
    rejection_list = config_file[camera_no]['defect_list']['rejection_list']
    print(rejection_list)

    #### for cam5 #####
    camera_no_cam5='cam5'
    brand_name=Path(opt.configfile).stem
    inspection_system_id_cam5=config_file['inspection_system_id']+camera_no_cam5
    # inspection_station_id=config_file['inspection_station_id']
    camera_id_cam5=config_file[camera_no_cam5]['camera_id']
    # #camera_ip=config_file[camera_no]['#camera_ip']
    camera_features_cam5=config_file[camera_no_cam5]['camera_features']
    # severe_defect_dict=config_file[camera_no]['defect_list']['severe_defect_dict']


    #implementing zmq for communication between scripts on 14 april 2023
    context = zmq.Context()
    sender_cam4 = context.socket(zmq.PUSH)
    print("cam 4 waiting for incoming connection")
    sender_cam4.bind("tcp://192.168.1.33:5555")

    sender_cam5 = context.socket(zmq.PUSH)
    print("cam 5 waiting for incoming connection")
    sender_cam5.bind("tcp://192.168.1.33:5556")


    def error_log(e,cam_id):
        print(e)
        error_time=str(datetime.now())
        error_string=f' error line no:{sys.exc_info()[-1].tb_lineno} error: {e}'
        exc_type, _, _ = sys.exc_info()
        # print(f'error name : {exc_type.__name__} \n value {value} \n traceback {traceback}')
        error_name=exc_type.__name__
        try:
            with open(f'./error_logs/{cam_id}_logs_{date.today()}.json','r') as file:
                error_log_data=json.loads(file.read())
                if str(error_name) in error_log_data:
                    error_log_data[str(error_name)]['last']=error_time
                    error_log_data[str(error_name)]['line_no']=error_string
                    error_log_data[str(error_name)]['Occurance_count']+=1
                else:
                    error_log_data[str(error_name)]={'start':error_time,'line_no':error_string,'last':'','Occurance_count':1}
            with open(f'./error_logs/{cam_id}_logs_{date.today()}.json','w+') as file:
                json.dump(error_log_data,file,indent=2)
                
        except FileNotFoundError:
            error_dict={}
            if str(error_name) not in error_dict:
                error_dict[str(error_name)]={'start':error_time,'line_no':error_string,'last':'','Occurance_count':1}        
            else :
                error_dict[str(error_name)]['last']=error_time
                error_dict[str(error_name)]['line_no']=error_string
                error_dict[str(error_name)]['Occurance_count']+=1
            with open(f'./error_logs/{cam_id}_logs_{date.today()}.json','w+') as file:
                json.dump(error_dict,file,indent=2)
        
        except Exception as e:
            print(f' error line no:{sys.exc_info()[-1].tb_lineno} error: {e}')
            error_time = str(datetime.now())
            error_string = f' error line no:{sys.exc_info()[-1].tb_lineno} error: {e}'
            filename = "./error_logs/Crash_"+str(date.today())+".log"
            f = open(filename, "a+")
            f.write(f'\nFile Name:- {cam_id} \n')
            f.write(error_string + " " + "\n")
            f.close()

## variables for cam4 #####
    json_data=[]
    global device
    device = select_device('0')
    total_packet_cam2=0
    defected_packet_cam2=0
    good_packet_cam2=0
    rejection_countC4 = 0
    img_count=0
    check_camera_connection=True
    show_flag=False
    list_defects=[]

##### variables for cam5 ######
    json_data=[]
    total_packet_cam5=0
    defected_packet_cam5=0
    good_packet_cam5=0
    img_count_cam5=0
    rejection_countC5 = 0
    check_camera_connection_cam5=True
    show_flag_cam5=False
    list_defects_cam5=[]

    #Rejection Command from here for Cam 4
    rejection_trigger_cam4 = False
    def rejectionC4():
        global rejection_trigger_cam4
        global rejection_countC4
        host = "192.168.1.68"
        PLC=SLCDriver(host)
        PLC.open()
        while True:
            try:
                PLC.read('B3:0/4')
                if rejection_trigger_cam4:
                    PLC._write_tag('B3:0/4',True)
                    rejection_trigger_cam4 = False
                    rejection_countC4 +=1
                    print('Defected Packet of C4 Rejected.', rejection_countC4)
                    PLC._write_tag('B3:0/4',False)
                # PLC.close()
            except Exception as e:
                # crash = " Error on line "
                error_log(e)


    #Rejection Command from here for Cam 5
    rejection_trigger_cam5 = False
    def rejectionC5():
        global rejection_trigger_cam5
        global rejection_countC5
        host = "192.168.1.68"
        PLC=SLCDriver(host)
        PLC.open()
        while True:
            try:
                PLC.read('B3:0/5')
                if rejection_trigger_cam5:
                    PLC._write_tag('B3:0/5',True)
                    rejection_trigger_cam5 = False
                    rejection_countC5 +=1
                    print('Defected Packet of C5 Rejected.',rejection_countC5)
                    PLC._write_tag('B3:0/5',False)
                # PLC.close()
            except Exception as e:
                # crash = " Error on line "
                error_log(e)

    @torch.no_grad()
    def run(xml,engine):
        imgsz=(640, 640)
        weights=engine
        data=xml
        global device
        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=True)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        # Run inference
        model.warmup(imgsz=(1, 3, *imgsz))
        return model

    
    def cv_window():
        global show_flag
        global total_packet_cam2
        global defected_packet_cam2
        global good_packet_cam2
        global cam2Image
            
        time_on = time.time()
        time_start = str(datetime.now())
        time_start = time_start[:-7]
        global list_defects
        global result_image
        
        while True:
            global total_packet_cam2
            global defected_packet_cam2
            global good_packet_cam2
            if show_flag==True:
                try:
                    start=time.perf_counter()
                    text_img = cv2.imread('./template.jpg')
                    text_img=cv2.resize(text_img, (360, 298))
                    duration_time = time.time() - time_on
                    hrs, rem = divmod(duration_time, 3600)
                    mins, sec = divmod(rem, 60)
                    duration_time = str('{:02}:{:02}:{:02}'.format(int(hrs), int(mins), int(sec)))
                    if total_packet_cam2!=0:
                        rejection_perc = "{:.2f}".format((defected_packet_cam2/total_packet_cam2)*100)
                    else:
                        rejection_perc=0
                    
                    cv2.putText(text_img, 'Start Date and Time:', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), thickness=2)
                    cv2.putText(text_img, time_start[0:16], (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), thickness=2)
                    cv2.putText(text_img, 'Good Packets:', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), thickness=2)
                    cv2.putText(text_img, 'Defected Packets:', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), thickness=2)
                    cv2.putText(text_img, 'Total Packets Scanned:', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0),
                                thickness=2)
                    cv2.putText(text_img, 'Detection Percentage: ', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0),
                                thickness=2)
                    cv2.putText(text_img, str(good_packet_cam2), (280, 70), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), thickness=2)
                    cv2.putText(text_img, str(defected_packet_cam2), (280, 90), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255),
                                thickness=2)
                    cv2.putText(text_img, str(total_packet_cam2), (280, 110), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), thickness=2)
                    cv2.putText(text_img, str(rejection_perc), (280, 130), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), thickness=2)

                    cv2.putText(text_img, str('Duration Time: ' + duration_time), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), thickness=2)
                    cv2.putText(text_img, str('Defects: ' + str(set(list_defects))), (10, 170), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), thickness=2)
                    vis = np.concatenate((result_image, text_img), axis=0)
                    
                    cam2Image=vis              
                    cam2Image=cv2.resize(cam2Image,(240,360))
                         
                         ##################
                    '''new sender using zmq on 14-4-23'''
                    if total_packet_cam2%5==0:
                        sender_cam4.send_pyobj(cam2Image)
                             ##############
                    cv2.waitKey(1)
                    
                except Exception as e:
                    error_log(e, 'cam4')

    def cv_window_cam5():
        global show_flag_cam5
        global total_packet_cam5
        global defected_packet_cam5
        global good_packet_cam5
        global cam5Image
        
        time_on = time.time()
        time_start = str(datetime.now())
        time_start = time_start[:-7]
        global list_defects_cam5
        global result_image_cam5
        
        while True:
            global total_packet_cam5
            global defected_packet_cam5
            global good_packet_cam5
            if show_flag_cam5==True:
                try:
                    start=time.perf_counter()
                    text_img = cv2.imread('./template.jpg')
                    text_img=cv2.resize(text_img, (360, 298))
                    duration_time = time.time() - time_on
                    hrs, rem = divmod(duration_time, 3600)
                    mins, sec = divmod(rem, 60)
                    duration_time = str('{:02}:{:02}:{:02}'.format(int(hrs), int(mins), int(sec)))
                    if total_packet_cam5!=0:
                        rejection_perc = "{:.2f}".format((defected_packet_cam5/total_packet_cam5)*100)
                    else:
                        rejection_perc=0
                    
                    cv2.putText(text_img, 'Start Date and Time:', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), thickness=2)
                    cv2.putText(text_img, time_start[0:16], (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), thickness=2)
                    cv2.putText(text_img, 'Good Packets:', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), thickness=2)
                    cv2.putText(text_img, 'Defected Packets:', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), thickness=2)
                    cv2.putText(text_img, 'Total Packets Scanned:', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0),
                                thickness=2)
                    cv2.putText(text_img, 'Detection Percentage: ', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0),
                                thickness=2)
                    cv2.putText(text_img, str(good_packet_cam5), (280, 70), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), thickness=2)
                    cv2.putText(text_img, str(defected_packet_cam5), (280, 90), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255),
                                thickness=2)
                    cv2.putText(text_img, str(total_packet_cam5), (280, 110), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), thickness=2)
                    cv2.putText(text_img, str(rejection_perc), (280, 130), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), thickness=2)

                    cv2.putText(text_img, str('Duration Time: ' + duration_time), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), thickness=2)
                    cv2.putText(text_img, str('Defects: ' + str(set(list_defects))), (10, 170), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), thickness=2)
                    vis = np.concatenate((result_image_cam5, text_img), axis=0)
                    
                    cam5Image=vis              
                    cam5Image=cv2.resize(cam5Image,(240,360))
                         
                         ##################
                    '''new sender using zmq on 14-4-23'''
                    if total_packet_cam5%5==0:
                        sender_cam5.send_pyobj(cam5Image)
                             ##############
                    cv2.waitKey(1)
                    
                except Exception as e:
                    error_log(e, 'cam5')

    def image_sender(im0,list_detections, brand_name, camera_id, inspection_system_id, inspection_station_id, cam_id):
        
        ######## uncomment below lines to save ng_images on local storage  #############
        
            # try:
            #     img_name=str(datetime.now()).replace(' ', '-').replace('.','-').replace(':','-')+'.jpg'
            #     cv2.imwrite('/mnt/Data/images/'+img_name,im0)
            
            # except Exception as e:
            #     error_log(e)

            retval, buffer = cv2.imencode('.jpg', cv2.resize(im0,(432,358)))
            jpg_as_text = str(base64.b64encode(buffer))[2:-1]
    
            dict_image_result = {
                                    "Image": jpg_as_text,
                                    'Brand':brand_name,
                                    "DateAndTime": str(datetime.strftime(datetime.now() - timedelta(0), '%Y-%m-%d %H:%M:%S.%f')),
                                    "ListDetections":  None if list_detections==[] else list_detections,
                                    "IsDetectionCorrect":None,
                                    "CameraID": camera_id,
                                    "InspectionSystemID": inspection_system_id,
                                    "InspectionStationID": inspection_station_id
                                }

            json_data.append(dict_image_result)
            jsonname= str(datetime.now()).replace(' ', '-').replace('.','-').replace(':','-')

            try:
                ### add path to data path on local storage to place jsons for server #####
                with open(f'/media/itc-jetson-3/Data/data/{cam_id}_data/'+jsonname+'.json','w+') as mongo_data:
                    json.dump(json_data,mongo_data,indent=4)
                    json_data.clear() 
            except Exception as e:
                error_log(e, cam_id)
    # im0,list_detections, brand_name, camera_id, inspection_system_id, inspection_station_id, cam_id
    def data_sender(list_detections, brand_name, camera_id, inspection_system_id, inspection_station_id, cam_id):
        jpg_as_text = " "
  
        dict_image_result = {
                                "Image": jpg_as_text,
                                'Brand':brand_name,
                                "DateAndTime": str(datetime.strftime(datetime.now() - timedelta(0), '%Y-%m-%d %H:%M:%S.%f')),
                                "ListDetections":  None if list_detections==[] else list_detections,
                                "IsDetectionCorrect":None,
                                "CameraID": camera_id,
                                "InspectionSystemID": inspection_system_id,
                                "InspectionStationID": inspection_station_id
                            }
        
        json_data.append(dict_image_result)
        jsonname= str(datetime.now()).replace(' ', '-').replace('.','-').replace(':','-')
        
        try:
            ### add path to data path on local storage to place jsons for server #####
            with open(f'/media/itc-jetson-3/Data/data/{cam_id}_data/'+jsonname+'.json','w+') as mongo_data:
                json.dump(json_data,mongo_data,indent=4)
                json_data.clear() 
        except Exception as e:
            error_log(e, cam_id)


    def object_detection(cam_id,model,image,str, conf_thres=0.35,  iou_thres=0.45, max_det=100,agnostic_nms=False,classes=None,line_thickness=3):
        try:
            global device
            stride, names, pt = model.stride, model.names, model.pt
            img = letterbox(image, 640, stride=stride, auto=pt)[0]
                # Convert
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            im = torch.from_numpy(img).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim            
            pred = model(im, augment=False, visualize=False)
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                img0= image.copy()
                annotator = Annotator(img0)
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()
            return [det,annotator,names]
        except Exception as e:
            error_log(e,cam_id)

    def camera_connect(camera_id,camera_features,cam_id):
        set_camera_feature_flag=False
        ia=neoapi.Cam()
        
        while set_camera_feature_flag==False:
            try:
                ia.Connect(camera_id)
                print(f'{cam_id} Connected.........')
                if ia.IsConnected(): 
                    f = ia.f.PixelFormat
                    fs = neoapi.FeatureStack()
                    fs.Add("ExposureTime", camera_features['ExposureTime'])  # add features to the stack
                    fs.Add("Gain", camera_features["Gain"])
                    fs.Add("Width", camera_features["Width"]) 
                    fs.Add("Height", camera_features["Height"])
                    fs.Add("OffsetX", camera_features["OffsetX"])
                    fs.Add("OffsetY", camera_features["OffsetY"])
                    fs.Add("Gamma",camera_features["Gamma"])
                    ia.WriteFeatureStack(fs)
                    
                    if f.GetEnumValueList().IsReadable("BayerRG8"):  # check if the camera supports the format
                        f.SetString("BayerRG8")
                        f.value = neoapi.PixelFormat_BayerRG8
                    ia.f.TriggerMode.value = neoapi.TriggerMode_Off
                    ia.f.LUTEnable.Set(True)  #line to handle gammas
                    set_camera_feature_flag=True
                    print(f'All {cam_id} features Set...')
                return ia
            except Exception as e:
                error_log(e, cam_id)
                print(f'Setting {cam_id} Features failed retrying... ')
    
    # This logic was implimented to count maximum confindence in perticular defects

    # highest_confidenceC4 = 0.0
    # def update_highest_confidenceC4(current_confidenceC4):
    #     global highest_confidenceC4
    #     if current_confidenceC4 > highest_confidenceC4:
    #         highest_confidenceC4 = current_confidenceC4
    #         print("The new highest cofidence of Cam 4: ", highest_confidenceC4)

    # highest_confidenceC5 = 0.0
    # def update_highest_confidenceC5(current_confidenceC5):
    #     global highest_confidenceC5
    #     if current_confidenceC5 > highest_confidenceC5:
            # highest_confidenceC5 = current_confidenceC5
            # print("The new highest cofidence of Cam 5: ", highest_confidenceC5)
    
    def camera_inference(model,brand_name):
        global opt
        ## for cam4 ##
        # global current_confidenceC4 This logic was implimented to count maximum conf in perticular defects ref.: update_highest_confidenceC4() function
        global total_packet_cam2
        global defected_packet_cam2
        global good_packet_cam2
        global camera_id
        global rejection_trigger_cam4
        global show_flag
        global list_defects
        global result_image
        global camera_features
        start_cam=False
        global check_camera_connection
        global json_data
        global start_rejection_cam4
        if opt.cam4==1:
            ia=camera_connect(camera_id,camera_features,'cam4')

        ## for cam5 ##
        # global current_confidenceC5 #This logic was implimented to count maximum conf in perticular defects ref.: update_highest_confidenceC5() function
        global camera_id_cam5
        global rejection_trigger_cam5
        global camera_features_cam5
        global total_packet_cam5
        global defected_packet_cam5
        global good_packet_cam5
        global check_camera_connection_cam5
        global show_flag_cam5
        global list_defects_cam5
        global result_image_cam5
        global start_rejection_cam5
        start_cam5=False
        if opt.cam5==1:
            ia_cam5=camera_connect(camera_id_cam5,camera_features_cam5,'cam5')

        
        # time_on = time.time()
        time_start = str(datetime.now())
        time_start = time_start[:-7]
        # flag=1
        # empty_flag=False

        while True:
            ng_packet_flag=False
            ng_packet_flag_cam5=False
            if opt.cam4==1:
                try:
                    start=time.perf_counter()
                    
                    if check_camera_connection==False:
                        ia.Disconnect()
                        start_cam=False
                        # print(' in camera disconnect execption')
                        ia=camera_connect(camera_id,camera_features,'cam4') 
                    image = ia.GetImage(20)
                    if image.IsEmpty()==False:
                        image=image.GetNPArray()
                        image=cv2.cvtColor(image,cv2.COLOR_BAYER_RG2RGB)
                        list_defects=[]
                        list_detections=[]
                        if image.shape != (0, 0,1):
                            if start_cam==False:
                                ia.f.TriggerMode.value = neoapi.TriggerMode_On
                                start_cam=True
                            detections=object_detection('cam4',model, image,2)
                            names=detections[2]
                            for *xyxy, conf, cls in reversed(detections[0]):
                                c = int(cls)  # integer class
                                label =  f'{names[c]}'
                                

                                if label.split('_')[0]=='ng' :
                                    

                                    detections[1].box_label(xyxy, f'{names[c]} {conf:.2f}', color=(0,0,255))
                                    
                                    # This logic was implimented to count maximum conf in perticular defects ref.: update_highest_confidenceC4() function

                                    # print(label)
                                    # current_confidenceC4 = conf
                                    # if label == 'ng_wrinkle':
                                    #     update_highest_confidenceC4(current_confidenceC4)

                                    

                                    if label not in list_defects:
                                        list_defects.append(label)
                                        dict_detections={
                                            
                                            'DefectName': label,
                                            'DefectCategory':config_file['defect_category'][label],
                                            'Confidence': f'{conf:.2f}',           
                                        }
                                        list_detections.append(dict_detections)
                                    ng_packet_flag=True
                                    
                                else: #incase of good packet
                                    detections[1].box_label(xyxy, f'{names[c]} {conf:.2f}', color=(0,255,0))
                                    # ng_packet_flag=False
                                    # empty_flag=False
                            im0 = detections[1].result()
                            
                            im0=cv2.resize(im0, (360, 298))
                            result_image=im0
                            # print(f'cam4 cycletime: {time.perf_counter()-start}')
                            if ng_packet_flag:
                                defected_packet_cam2 += 1
                                total_packet_cam2 += 1
                                for label in list_defects:
                                    if label in rejection_list:
                                        rejection_trigger_cam4 = True
                                        print(f"Packet Rejected for defect: {label}")
                                        break
                                

                                send_image=threading.Thread(target=image_sender, args=(im0,list_detections, brand_name, camera_id, inspection_system_id, inspection_station_id, 'cam4',))
                                send_image.start()
                                
                            else:
                                good_packet_cam2 += 1
                                total_packet_cam2 += 1
                                send_data=threading.Thread(target=data_sender,args=(list_detections, brand_name, camera_id, inspection_system_id, inspection_station_id, 'cam4',))
                                send_data.start()
                            

                            show_flag=True
                        else:
                            print('Camera 4 No Trigger')
                            # if os.system(f'ping -c 1 -t 1 {camera_ip}'):
                            #     check_camera_connection=False

                except Exception as e:                
                    error_log(e, 'cam4')
                    if str(e)=='NoAccessException: Device is offline!' or str(e)=='NotConnectedException' :
                        ia.Disconnect()
                        start_cam=False
                        # print(' in camera disconnect execption')
                        ia=camera_connect(camera_id,camera_features,'cam4') 
                                    #### for cam5 #####
            if opt.cam5==1:
                try:
                    if check_camera_connection_cam5==False:
                        ia_cam5.Disconnect()
                        start_cam5=False
                        # print(' in camera disconnect execption')
                        ia_cam5=camera_connect(camera_id_cam5,camera_features_cam5,'cam5') 
                    image = ia_cam5.GetImage(20)
                    if image.IsEmpty()==False:
                        image=image.GetNPArray()
                        image=cv2.cvtColor(image,cv2.COLOR_BAYER_RG2RGB)
                        list_defects_cam5=[]
                        list_detections_cam5=[]
                        if image.shape != (0, 0,1):
                            if start_cam5==False:
                                ia_cam5.f.TriggerMode.value = neoapi.TriggerMode_On
                                start_cam5=True
                            detections=object_detection('cam5',model, image,2)
                            names=detections[2]
                            for *xyxy, conf, cls in reversed(detections[0]):
                                c = int(cls)  # integer class
                                label =  f'{names[c]}'
                                
                                


                                if label.split('_')[0]=='ng' :
                                    detections[1].box_label(xyxy, f'{names[c]} {conf:.2f}', color=(0,0,255))

                                    # This logic was implimented to count maximum conf in perticular defects ref.: update_highest_confidenceC5() function

                                    # print(label)
                                    # current_confidenceC5 = conf
                                    # if label == 'ng_wrinkle':
                                        # update_highest_confidenceC5(current_confidenceC5)


                                    if label not in list_defects_cam5:
                                        list_defects_cam5.append(label)
                                        dict_detections={
                                            
                                            'DefectName': label,
                                            'DefectCategory':config_file['defect_category'][label],
                                            'Confidence': f'{conf:.2f}',           
                                        }
                                        list_detections_cam5.append(dict_detections)
                                    ng_packet_flag_cam5=True
                                    
                                else: #incase of good packet
                                    detections[1].box_label(xyxy, f'{names[c]} {conf:.2f}', color=(0,255,0))
                                    # ng_packet_flag=False
                                    # empty_flag=False
                            im0 = detections[1].result()
                            im0=cv2.resize(im0, (360, 298))
                            result_image_cam5=im0
                            # print(f'cycletime: {time.perf_counter()-start}')
                            if ng_packet_flag_cam5:
                                defected_packet_cam5 += 1
                                total_packet_cam5 += 1
                                for label in list_defects_cam5:
                                    if label in rejection_list:
                                        rejection_trigger_cam5 = True
                                        print(f"Packet Rejected for defect: {label}")
                                        break
                                send_image=threading.Thread(target=image_sender, args=(im0,list_detections_cam5, brand_name, camera_id_cam5, inspection_system_id_cam5, inspection_station_id, 'cam5',))
                                send_image.start()
                                
                            else:
                                good_packet_cam5 += 1
                                total_packet_cam5 += 1
                                send_data=threading.Thread(target=data_sender,args=(list_detections_cam5, brand_name, camera_id_cam5, inspection_system_id_cam5, inspection_station_id, 'cam5',))
                                send_data.start()
                            

                            show_flag_cam5=True
                        else:
                            print('Camera 5 No Trigger')
                            # if os.system(f'ping -c 1 -t 1 {camera_ip}'):
                            #     check_camera_connection=False
                            
                except Exception as e:                
                    error_log(e, 'cam5')
                    if str(e)=='NoAccessException: Device is offline!' or str(e)=='NotConnectedException' :
                        ia_cam5.Disconnect()
                        start_cam=False
                        # print(' in camera disconnect execption')
                        ia_cam5=camera_connect(camera_id_cam5,camera_features_cam5,'cam5')
                        # ia=camera_connect(camera_id,camera_features,'cam4') 

            
            shift_change=['06:00:00','14:00:00','22:00:00']
            if datetime.now().strftime('%H:%M:%S') in shift_change and (total_packet_cam2>15 or total_packet_cam5>15) :   
                total_packet_cam2=0
                defected_packet_cam2=0
                good_packet_cam2=0
            
                total_packet_cam5=0
                defected_packet_cam5=0
                good_packet_cam5=0
            
                
           

    model=run(config_file[camera_no]['xml_path'],config_file[camera_no]['engine_path'])
    show_image_thread=threading.Thread(target=cv_window,)
    show_image_cam5_thread=threading.Thread(target=cv_window_cam5,)


    #Rejection Thread for Cam 4
    start_rejection_cam4 = threading.Thread(target=rejectionC4,)
    start_rejection_cam4.start()

    #Rejection Thread for Cam 5
    start_rejection_cam5 = threading.Thread(target=rejectionC5,)
    start_rejection_cam5.start()
    
    if opt.cam4==1:
        show_image_thread.start()
    if opt.cam5==1:
        show_image_cam5_thread.start()
    camera_inference(model,brand_name)

except Exception as e:    
    print(str(datetime.now())+'error on line '+str(sys.exc_info()[-1].tb_lineno)+' error: '+str(e))
    error_string=str(datetime.now())+' error line no:'+str(sys.exc_info()[-1].tb_lineno)+' error: '+str(e)
    try:
        filename = "Crash.log"
        f = open(filename, "a+")
        f.write('\nFile Name:- Camera-4_5 \n')
        f.write(error_string + " " + "\n")
        f.close()
    except Exception as e:
        print("Unable to write log ", str(e))
