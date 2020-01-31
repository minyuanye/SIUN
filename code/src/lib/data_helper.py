import os
from skimage import io,img_as_float # image process
import numpy as np

class DataHelper:
    def __init__(self):
        self.__fileBlurList=[]
        self.__directoryList=[]
        self.__blurSharpPairs=[]

    def __traversalDir(self,root):
        for name in os.listdir(root):
          fullPath = os.path.join(root, name)
          if os.path.isdir(fullPath):
            self.__directoryList.append(fullPath)
        for directory in self.__directoryList:
          for parent,dirnames,filenames in os.walk(os.path.join(directory,'blur')):
            for filename in filenames:
              self.__fileBlurList.append(os.path.join(parent,filename))

    def load_data(self, path, number):#shuffle
        self.__traversalDir(path)
        if(number>0):
            np.random.shuffle(self.__fileBlurList)
        totalLoaded = 0
        print(f'start loading dataset...')
        for fileFullPath in self.__fileBlurList:
          #imageBlur = io.imread(fileFullPath,as_gray=True)
          #imageSharp = io.imread(fileFullPath.replace('/blur','/sharp'),as_gray=True)
          imageBlur = img_as_float(io.imread(fileFullPath))
          imageSharp = img_as_float(io.imread(fileFullPath.replace('/blur','/sharp')))
          self.__blurSharpPairs.append((imageBlur,imageSharp))
          totalLoaded += 1
          if(totalLoaded == number):#if number < 1, all datas loaded
            break
        print(f'dataset loaded:{totalLoaded}!')

    def getRandomTrainDatas(self,config):
        X_train=[]
        Y_train=[]
        patchW = patchH = config.trainer.generatorImageSize
        for imageBlur,imageSharp in self.__blurSharpPairs:
          trainImageH = imageBlur.shape[0]
          trainImageW = imageBlur.shape[1]
          rowStart = np.random.randint(0, trainImageH-patchH)
          colStart = np.random.randint(0, trainImageW-patchW)
          X_train.append(imageBlur[rowStart:rowStart+patchH,colStart:colStart+patchW])
          Y_train.append(imageSharp[rowStart:rowStart+patchH,colStart:colStart+patchW])
        return X_train,Y_train#(row,col)

    def getTestDatas(self):
        #for imageBlur,imageSharp in self.__blurSharpPairs:
        return self.__fileBlurList

    def getLoadedPairs(self):
        return self.__blurSharpPairs

    def loadDataList(self, path):
        self.__traversalDir(path)
        data_length = len(self.__fileBlurList)
        print(f'dataset got:{data_length}!')
        return data_length

    def getAPair(self,index,config):
        fileFullPath = self.__fileBlurList[index]
        imageBlur = img_as_float(io.imread(fileFullPath))
        imageSharp = img_as_float(io.imread(fileFullPath.replace('/blur','/sharp')))
        patchW = patchH = config.trainer.generatorImageSize
        trainImageH = imageBlur.shape[0]
        trainImageW = imageBlur.shape[1]
        rowStart = np.random.randint(0, trainImageH-patchH)
        colStart = np.random.randint(0, trainImageW-patchW)
        return imageBlur[rowStart:rowStart+patchH,colStart:colStart+patchW],imageSharp[rowStart:rowStart+patchH,colStart:colStart+patchW]