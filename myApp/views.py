from django.http import HttpResponse, JsonResponse
from django.shortcuts import render

from myApp.models import Detector
from django.core.files.storage import FileSystemStorage

from rest_framework.decorators import api_view
from rest_framework.response import Response

# Create your views here.
# @api_view(['POST','GET'])           # means only post method is allowesd .. if we didn.t mention then all allowed
def home(request):
    # return Response({"status": 200 , "message": "this is my message"})
    return render(request,'home.html')

@api_view(['POST'])           # means only post method is allowesd .. if we didn.t mention then all allowed
# @api_view(['POST','GET'])           # means only post method is allowesd .. if we didn.t mention then all allowed
def result(request):
        
    import cv2
    import numpy as np
    import urllib.request
    import os
    import json

    data=dict()
    # data = {"success": False}

    if request.method == 'POST' :
        print('requested s / : ',request.POST.get('file'))

    if request.method == 'POST' and request.POST.get('file') != '' or request.POST.get('file') != None:     #that input field has name='file' there4 
        file = request.FILES['file']
        fss = FileSystemStorage()
        file = fss.save(file.name, file)
        file_url = fss.url(file)
      
        # data["success"] = True

        pathdir =os.getcwd()
        pathdir = pathdir.replace("\\", "/")

        # url = "https://www.thespruce.com/thmb/ZNcCv0_s3l72dvr9z_eDccMGbVg=/1365x1365/smart/filters:no_upscale()/exciting-small-kitchen-ideas-1821197-hero-d00f516e2fbb4dcabb076ee9685e877a.jpg"

        # url_response = urllib.request.urlopen(url)
        # img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        # img = cv2.imdecode(img_array, -1)     

        #for local images
        # path= file_url
        path= pathdir + file_url
        # path = pathdir + '/myApp/static/'+file
        img= cv2.imread(path, cv2.IMREAD_COLOR)

        wht =320    
        confThreshold = 0.1
        nmsThreshold = 0.3     
        showOps=[]

        classesFile = pathdir + '/myApp/static/coco.names'
        # classesFile = 'coco.names'
        classNames = []
        with open(classesFile, 'rt') as f:
            classNames = f.read().rstrip('\n').split('\n')

        modelConfiguration = pathdir + '/myApp/static/yolov3.cfg'
        # modelConfiguration = 'yolov3.cfg'
        modelWeights = pathdir + '/myApp/static/yolov3.weights'
        # modelWeights = 'yolov3.weights'

        net = cv2.dnn.readNetFromDarknet(modelConfiguration , modelWeights)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        def findObjects(outputs, img):
            ht, wt, ct = img.shape
            bbox = []               #bounding box
            classIds = []
            confs = []

            for output in outputs:
                for det in output:
                    scores = det[5:]             
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    if confidence > confThreshold:
                        w,h = int(det[2]*wt) , int(det[3]*ht)                     
                        x,y = int((det[0]*wt) - w/2) , int((det[1]*ht) - h/2)                      
                        bbox.append([x,y,w,h])
                        classIds.append(classId)
                        confs.append(float(confidence))

            indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
            for i in indices:
                box = bbox[i]
                x,y,w,h = box[0], box[1], box[2], box[3]
                cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 2)
                cv2.putText(img, f'{classNames[classIds[i]].upper() } {int(confs[i]*100)}%', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
                if classNames[classIds[i]] not in  showOps:
                    showOps.append(classNames[classIds[i]])
                    print(classNames[classIds[i]],end=" , ")

        blob = cv2.dnn.blobFromImage(img,1/255, (wht,wht), [0,0,0], 1, crop =False)  
        net.setInput(blob)

        layerNames = net.getLayerNames()

        outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]

        outputs = net.forward(outputNames)

        findObjects(outputs, img)

        cv2. namedWindow("Image", cv2.WINDOW_NORMAL)    
        cv2.imshow("Image", img)

        data['output'] = showOps         # setting output to json format

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()     

        # return render(request,'result.html', {'showOps': showOps,'file_url': file_url})
        # return JsonResponse(data)
        return Response({"status": 200 , "message": data})


        # return render(request, 'main/upload.html', {'file_url': file_url})
    # return render(request, 'main/upload.html')

    return Response({"status": 500 , "message": "Unexpected Error"})
        # return JsonResponse(data)

    # return render(request,'result.html')
  

    # params = {'showOps': showOps, 'file': file}

    # return render(request,'result.html', {'file_url': file_url})
    # return render(request,'result.html', params)