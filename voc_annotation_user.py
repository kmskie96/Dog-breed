import xml.etree.ElementTree as ET
from os import getcwd
import glob
import re

global classes
def convert_voc_2_darknet(data_folder_name, converted_file_name):
  
    def load_classes():
        with open('%s/classes.txt'%(data_folder_name)) as f:
            aclasses = f.read().splitlines()

        return aclasses

    def convert_annotation(image_id, converted_file):
        tree=ET.parse('%s/Annotations/%s.xml'%(data_folder_name, image_id))
        root = tree.getroot()

        for obj in root.iter('object'):
          difficult = obj.find('difficult').text
          cls = obj.find('name').text
          if cls not in classes or int(difficult)==1:
              continue
          cls_id = classes.index(cls)
          xmlbox = obj.find('bndbox')
          b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
          converted_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

    classes = load_classes()
    print("=========>", '%s/Annotations/*.xml'%(DATA_FOLDER_NAME) ,data_folder_name)
    image_ids = [ re.findall("([\w-]+).xml", f)[0] for f in glob.glob('%s/Annotations/*.xml'%(DATA_FOLDER_NAME))]

    wd = getcwd()
    converted_file = open(converted_file_name, 'w')
    for image_id in image_ids:
      converted_file.write('%s/JPEGImages/%s.jpg'%(data_folder_name, image_id))
      convert_annotation(image_id, converted_file)
      converted_file.write('\n')
    converted_file.close()


DATA_FOLDER_NAME = "yolo_data"
#DATA_FOLDER_NAME = ""
CONVERTED_FILE_NAME = "train_all.txt"

convert_voc_2_darknet(DATA_FOLDER_NAME, CONVERTED_FILE_NAME)
