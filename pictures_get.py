import cv2
import dlib
import os
import sys
import random

output_dir = './wyj_faces'  #采集图片的输出目录
size = 64

if not os.path.exists(output_dir):  #如果采集图片的输出目录不存在则创建
    os.makedirs(output_dir)

# 改变图片的亮度与对比度
def relight(img, light=1, bias=0):  #定义函数
    w = img.shape[1]  #图片的宽的像素数
    h = img.shape[0]  #图片的高的像素数
    #image = []
    for i in range(0,w):  #一列一列地来改变像素的亮度和对比度
        for j in range(0,h):
            for c in range(3):
                tmp = int(img[j,i,c]*light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j,i,c] = tmp
    return img

#使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()
# 打开摄像头 参数为输入流，可以为摄像头或视频文件
camera = cv2.VideoCapture(0)

index = 1
while True:
    if (index <= 200):  #截取1000张图片
        print('Being processed picture %s' % index)  #输出当前截取图片的进度
        # 从摄像头读取照片
        success, img = camera.read()
        # 转为灰度图片
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 使用detector进行人脸检测
        dets = detector(gray_img, 1)  #detector会返回识别到的人脸的矩形的左下角和右上角的坐标

        for i, d in enumerate(dets):  #enumerate用于遍历括号中的元素及其下标，其中i对应元素下标，d对应元素
            x1 = d.top() if d.top() > 0 else 0  #通过left，right，top，down获取矩形的四个坐标x1，x2，y1，y2
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0

            face = img[x1:y1,x2:y2]  #把图片里含有人脸的矩形截取出来给face
            # 调整图片的对比度与亮度， 对比度与亮度值都取随机数，这样能增加样本的多样性
            face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))

            face = cv2.resize(face, (size,size))  #调整图片大小

            cv2.imshow('Easy & Happy-get picture now', face)  #显示处理后的图像

            cv2.imwrite(output_dir+'/'+str(index)+'.jpg', face)  #将图片保存下来

            index += 1
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
    else:
        print('Finished!')
        break
