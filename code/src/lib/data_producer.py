import os
from skimage import io,img_as_float # image process
import numpy as np
import threading

class DataProducer(threading.Thread):
    def __init__(self, name,queue,config):
        threading.Thread.__init__(self, name=name,daemon=True)
        self.data=queue
        self.__fileBlurList=[]
        self.__directoryList=[]
        self.__blurSharpParis=[]
        self.config = config
        self.running = True

    def __traversalDir(self,root):
        for name in os.listdir(root):
          fullPath = os.path.join(root, name)
          if os.path.isdir(fullPath):
            self.__directoryList.append(fullPath)
        for directory in self.__directoryList:
          for parent,dirnames,filenames in os.walk(os.path.join(directory,'blur')):
            for filename in filenames:
              self.__fileBlurList.append(os.path.join(parent,filename))

    def loadDataList(self, path):
        self.__traversalDir(path)
        self.data_length = len(self.__fileBlurList)
        print(f'dataset got:{self.data_length}!')
        return self.data_length

    def __produceAPair(self,index):
        fileFullPath = self.__fileBlurList[index]
        imageBlur = img_as_float(io.imread(fileFullPath))
        imageSharp = img_as_float(io.imread(fileFullPath.replace('/blur','/sharp')))
        patchW = patchH = self.config.trainer.generatorImageSize
        trainImageH = imageBlur.shape[0]
        trainImageW = imageBlur.shape[1]
        rowStart = np.random.randint(0, trainImageH-patchH)
        colStart = np.random.randint(0, trainImageW-patchW)
        blur = imageBlur[rowStart:rowStart+patchH,colStart:colStart+patchW]
        sharp = imageSharp[rowStart:rowStart+patchH,colStart:colStart+patchW]
        self.data.put((blur,sharp),1)#block

    def run(self):
        arr = np.arange(self.data_length)
        while(True):
            #an epoch
            np.random.shuffle(arr)
            for i in range(self.data_length):
                index = arr[i]
                self.__produceAPair(index)