'''
将voc格式数据转化成 yolo training 格式
'''
import xml.etree.ElementTree as ET
from os import getcwd
import os

# 扫描数据
sets=[('2012', 'train'), ('2012', 'val')]
# 类型
classes = ["text"]
ROOTPATH = r'C:\Users\hupan\Desktop\Data'



Xml_list_File = 'VOCdevkit/VOC%s/Annotations/%s.xml'
Xml_list_File_abs = os.path.join(ROOTPATH,Xml_list_File)
def convert_annotation(year, image_id, list_file):
    in_file = open(Xml_list_File_abs%(year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


# 当前目录
wd = getcwd()


Train_list_File = 'VOCdevkit/VOC%s/ImageSets/Main/%s.txt'
Train_list_File_abs = os.path.join(ROOTPATH,Train_list_File)

Image_File = 'VOCdevkit/VOC%s/JPEGImages/%s.jpg'
Image_File_abs = os.path.join(ROOTPATH,Image_File)


for year, image_set in sets:


    # 获取train val test信息
    file_name = Train_list_File_abs%(year, image_set)
    print(file_name)
    image_ids = open(file_name).read().strip().split()
    # 导出数据名称
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        list_file.write(Image_File_abs%(year, image_id))
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()

