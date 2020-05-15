import os
from src.model.model import DDModel
from src.lib.data_helper import DataHelper
from skimage import io,transform,feature,color,img_as_float
import numpy as np
import math
import time

class Application():
#note that input image must be color, gray image should be expand to 3 channels
#and image size must be even

    def __init__(self,config):
        self.config = config
        self.model = DDModel(config)
        if(config.application.deblurring_result_dir is None):
            config.application.deblurring_result_dir = config.resource.output_dir
        if not os.path.exists(config.application.deblurring_result_dir):
            os.makedirs(config.application.deblurring_result_dir)
        self.__fileBlurList=[]

    def start(self):
        self.application()

    def __tuneSize(self,shape):
        pad = []
        for i in range(2):
            size = shape[i]
            if(size % 256 == 0):
                pad.append(0)
            else:
                n = size // 256 + 1
                pad.append((n*256 - size) // 2)
        return pad

    def __getImage(self,fileFullPath):#self.config.application.deblurring_file_path
        imageBlur = img_as_float(io.imread(fileFullPath))
        #make sure row&col are even
        row = imageBlur.shape[0]
        col = imageBlur.shape[1]
        row = row-1 if row%2==1 else row
        col = col-1 if col%2==1 else col
        imageBlur = imageBlur[0:row,0:col]
        imageOrigin = imageBlur
        pad = self.__tuneSize(imageBlur.shape)
        imageBlur = np.pad(imageBlur,((pad[0],pad[0]),(pad[1],pad[1]),(0,0)),'reflect')
        return imageBlur,imageOrigin

    def __getData(self,root):
        for parent,dirnames,filenames in os.walk(root):
            for filename in filenames:
                self.__fileBlurList.append(os.path.join(parent,filename))
        self.data_length = len(self.__fileBlurList)
        print(f'total data:{self.data_length}!')

    def __deblur(self,imageBlur,imageOrigin):
        pyramid = tuple(transform.pyramid_gaussian(imageBlur, downscale=2, max_layer=self.max_iter, multichannel=True))
        deblurs = []
        for iter in self.iters:
            batch_blur2x = []
            batch_blur1x = []
            runtime = 0;
            for i in range(iter,0,-1):
                if(i == iter):#first iter
                    imageBlur2x = pyramid[i]
                    batch_blur2x.append(imageBlur2x)
                    batch_gen = batch_blur2x
                else:
                    batch_blur2x = batch_blur1x
                    batch_blur1x = []
                imageBlur1x = pyramid[i-1]
                batch_blur1x.append(imageBlur1x)
                data_X1 = np.concatenate((batch_blur2x,batch_gen), axis=3)#6channels
                data_X = {'imageSmall':data_X1,'imageUp':np.array(batch_blur1x)}
                start = time.time()
                batch_gen = self.model.generator.predict(data_X)
                print(f'Runtime @scale {i}:{time.time()-start:4.3f}')
                runtime += time.time()-start;
            print(f'Runtime total @iter {iter}:{runtime:4.3f}')
            deblur = self.__clipOutput(batch_gen[0],imageOrigin.shape)
            deblurs.append(deblur)
        return deblurs

    def application(self):
        if(self.config.application.iter == 0):
            self.iters = [1,2,3,4]
        else:
            self.iters = [self.config.application.iter]
        self.max_iter = max(self.iters)
        deblurring_file_path = self.config.application.deblurring_file_path
        deblurring_dir_path = self.config.application.deblurring_dir_path
        if(deblurring_file_path and os.path.exists(deblurring_file_path)):
            imageBlur,imageOrigin = self.__getImage(deblurring_file_path)
            deblurs = self.__deblur(imageBlur,imageOrigin)
            infos = deblurring_file_path.rsplit('/', 1)
            iter_times = len(deblurs)
            for i in range(iter_times):
                deblur = deblurs[i]
                deblur = (deblur * 255).astype('uint8')
                iter = self.iters[i]
                io.imsave(os.path.join(self.config.application.deblurring_result_dir, 'deblur'+str(iter)+'_'+infos[1]),deblur)
            print(f'file saved')
        elif(deblurring_dir_path and os.path.exists(deblurring_dir_path)):
            self.__getData(deblurring_dir_path)
            index = 0
            for fileFullPath in self.__fileBlurList:
                imageBlur,imageOrigin = self.__getImage(fileFullPath)
                deblurs = self.__deblur(imageBlur,imageOrigin)
                infos = os.path.basename(fileFullPath)
                iter_times = len(deblurs)
                for j in range(iter_times):
                    deblur = deblurs[j]
                    deblur = (deblur * 255).astype('uint8')
                    iter = self.iters[j]
                    io.imsave(os.path.join(self.config.application.deblurring_result_dir, 'deblur'+str(iter)+'_'+infos),deblur)
                index += 1
                print(f'{index}/{self.data_length} done!')
            print(f'all saved')
        else:
            print(f"no deblur file(s)")

    def __clipOutput(self,image,outSize):
        inSize = image.shape
        start = []
        for i in range(2):
            start.append((inSize[i] - outSize[i]) // 2)
        return image[start[0]:start[0]+outSize[0],start[1]:start[1]+outSize[1]]