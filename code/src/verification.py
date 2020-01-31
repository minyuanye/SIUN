from src.model.model import DDModel
from src.lib.data_producer import DataProducer
from src.lib.data_helper import DataHelper
import tensorflow as tf
import keras.backend as K
from keras.optimizers import RMSprop,Adam
from skimage import io,transform,feature,color,img_as_float
import numpy as np
import sys
from keras.utils.training_utils import multi_gpu_model
import queue
import threading
import math

class Verification():
    def __init__(self,config):
        self.config = config
        self.model = DDModel(config)
        self.batch_size = config.trainer.batch_size
        self.learningRate = 1e-6
        self.bestMetric = 0#psnr
        self.bestEpoch = 0
        self.patience = 200
        self.current_size = 0
        self.iters = [3]
        self.iter_length = len(self.iters)
        self.pyramid_blurs = []
        self.pyramid_sharps = []

    def start(self):
        #json_path=self.config.resource.generator_json_path
        #infos = json_path.split('generator')
        #infos = infos[1].split('.')
        #json_info = infos[0]
        #weights_path=self.config.resource.generator_weights_path
        #infos = weights_path.split('generator')
        #infos = infos[1].split('.')
        #weights_info = infos[0]
        #print(f'json/weight:{json_info}/{weights_info}')
        print(f'verification strategy:{self.iters}')
        self.bestMetric = self.__getMetric()#init
        print(f'init metric:{self.bestMetric}')
        self.train()

    def __trainBatch(self):
        batch_blurs2x = []
        batch_blurs1x = []
        batch_sharps1x = []
        n = len(self.pyramid_blurs)
        for i in range(self.max_iter,0,-1):
            if(i == self.max_iter):#first iter
                #generate batch_blurs2x
                for j in range(n):
                    pyramid_blur = self.pyramid_blurs[j]
                    imageBlur2x = pyramid_blur[i]
                    batch_blurs2x.append(imageBlur2x)
                batch_gen = batch_blurs2x
            else:
                #generate batch_blurs2x
                batch_blurs2x = batch_blurs1x
                batch_blurs1x = []
                batch_sharps1x = []
            #generate batch_blurs1x
            for j in range(n):
                pyramid_blur = self.pyramid_blurs[j]
                imageBlur1x = pyramid_blur[i-1]
                batch_blurs1x.append(imageBlur1x)
            #generate batch_sharps1x
            for j in range(n):
                pyramid_sharp = self.pyramid_sharps[j]
                imageSharp1x = pyramid_sharp[i-1]
                batch_sharps1x.append(imageSharp1x)
            #data generate end
            
            #train Generator 2x
            train_X1 = np.concatenate((batch_blurs2x,batch_gen), axis=3)#6channels
            train_X = {'imageSmall':train_X1,'imageUp':np.array(batch_blurs1x)}
            g_loss = self.generator.train_on_batch(train_X,np.array(batch_sharps1x))
            if(i == 1):#last iter
                self.g_loss += g_loss * n
            else:
                batch_gen = self.generator.predict(train_X)
        #train end,reset
        self.current_size = 0
        self.pyramid_blurs = []
        self.pyramid_sharps = []

    def __doInteration(self,blur,sharp,epoch):
        iter_index = epoch%self.iter_length
        self.max_iter = self.iters[iter_index]
        if(self.current_size < self.batch_size):
            self.pyramid_blurs.append(tuple(transform.pyramid_gaussian(blur, downscale=2, max_layer=self.max_iter, multichannel=True)))
            self.pyramid_sharps.append(tuple(transform.pyramid_gaussian(sharp, downscale=2, max_layer=self.max_iter, multichannel=True)))
            self.current_size += 1
        if(self.current_size == self.batch_size):#train a batch
            self.__trainBatch()

    def __compute_psnr(self, x , label , max_diff):
        mse =  np.mean(( x - label ) **2 )
        return 10*math.log10( max_diff**2 / mse )

    def __testBatch(self,pyramid_blurs,batch_sharps):
        n = len(pyramid_blurs)
        psnrs = []
        for iter in self.iters:
            batch_blurs2x = []
            batch_blurs1x = []
            for i in range(iter,0,-1):
                if(i == iter):#first iter
                    #generate batch_blurs2x
                    for j in range(n):
                        pyramid_blur = pyramid_blurs[j]
                        imageBlur2x = pyramid_blur[i]
                        batch_blurs2x.append(imageBlur2x)
                    batch_gen = batch_blurs2x
                else:
                    #generate batch_blurs2x
                    batch_blurs2x = batch_blurs1x
                    batch_blurs1x = []
                #generate batch_blurs1x
                for j in range(n):
                    pyramid_blur = pyramid_blurs[j]
                    imageBlur1x = pyramid_blur[i-1]
                    batch_blurs1x.append(imageBlur1x)
                #data prepare end
                
                #predict 2x
                data_X1 = np.concatenate((batch_blurs2x,batch_gen), axis=3)#6channels
                data_X = {'imageSmall':data_X1,'imageUp':np.array(batch_blurs1x)}
                batch_gen = self.model.generator.predict(data_X)
            #calculate metrics
            batch_psnrs = []
            for i in range(n):
                pImage = batch_gen[i]
                pImage = pImage[24:744]
                psnr = self.__compute_psnr(pImage, batch_sharps[i], 1)
                batch_psnrs.append(psnr)
            psnrs.append(batch_psnrs)
        psnrs = np.array(psnrs)
        best_index = np.argmax(psnrs,axis=0)
        for i in range(n):
            best_psnr = psnrs[best_index[i]][i]
            best_iter = self.iters[best_index[i]]
            self.best_psnrs.append(best_psnr)
            self.best_iters.append(best_iter)

    def __getMetric(self):
        dataHelper = DataHelper()
        dataHelper.loadDataList(self.config.resource.test_directory_path)
        fileBlurList = dataHelper.getTestDatas()
        batch_size = 8
        max_iter = max(self.iters)
        #metrics
        self.best_psnrs = []
        self.best_iters = []
        
        current_size = 0
        pyramid_blurs = []
        batch_sharps = []
        for fileFullPath in fileBlurList:
            blur = img_as_float(io.imread(fileFullPath))
            sharp = img_as_float(io.imread(fileFullPath.replace('/blur','/sharp')))
            if(current_size < batch_size):
                blur = np.pad(blur,((24,24),(0,0),(0,0)),'reflect')#be divided by 256
                pyramid_blurs.append(tuple(transform.pyramid_gaussian(blur, downscale=2, max_layer=max_iter, multichannel=True)))
                batch_sharps.append(sharp)
                current_size += 1
            if(current_size == batch_size):#verify a batch
                self.__testBatch(pyramid_blurs,batch_sharps)
                current_size = 0
                pyramid_blurs = []
                batch_sharps = []
        if(pyramid_blurs):
            self.__testBatch(pyramid_blurs,batch_sharps)
            current_size = 0
            pyramid_blurs = []
            batch_sharps = []
        return np.mean(self.best_psnrs)

    def __verify(self,epoch):
        if(epoch % 50 != 0):
            return False
        metric = self.__getMetric()
        print(f'current metric:{metric}')
        if(metric > self.bestMetric):
            self.bestMetric = metric
            self.bestEpoch = epoch
            self.model.save(self.model.generator,self.config.resource.generator_json_path,self.config.resource.generator_weights_path)
            return False
        elif(epoch - self.bestEpoch < self.patience):
            return False
        else:
            return True
    
    def train(self):
        optimizer = Adam(self.learningRate)
        if(self.config.trainer.gpu_num>1):
            self.generator = multi_gpu_model(self.model.generator, self.config.trainer.gpu_num)
        else:
            self.generator = self.model.generator
        self.generator.compile(loss='mean_absolute_error', optimizer=optimizer)
        print(f'generator:{self.generator.metrics_names}')
        
        image_queue = queue.Queue(maxsize=self.config.trainer.batch_size*4)
        dataProducer = DataProducer('Producer',image_queue,self.config)
        n = dataProducer.loadDataList(self.config.resource.train_directory_path)
        dataProducer.start()
        epoch = 0
        while(True):
            self.g_loss = 0
            for i in range(n):
              imageBlur,imageSharp = image_queue.get(1)#block
              self.__doInteration(imageBlur,imageSharp,epoch)
            if(self.pyramid_blurs):
              #last batch, may smaller than batch_size
              self.__trainBatch()
            #f_g_loss = ["{:.2f}".format(x) for x in self.g_loss]
            self.g_loss = self.g_loss/n
            f_g_loss = "{:.3e}".format(self.g_loss)
            print(f'verification epoch:{epoch},[G loss:{f_g_loss}]')
            epoch += 1
            if(self.__verify(epoch)):
                break