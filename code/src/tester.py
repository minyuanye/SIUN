import os
from src.model.model import DDModel
from src.lib.data_helper import DataHelper
from src.lib.MLVSharpnessMeasure import MLVMeasurement
from skimage import io,transform #reize image
import numpy as np
import pickle
import math

class Tester():
    def __init__(self,config):
        self.config = config
        self.model = DDModel(config)
        self.batch_size = 8
        self.current_size = 0
        self.pyramid_blurs = []
        self.batch_sharps = []
        #metrics
        self.all_psnrs = {}

    def start(self):
        if(self.config.tester.iter == 0):
            self.iters = [1,2,3,4]
        else:
            self.iters = [self.config.application.iter]
        self.max_iter = max(self.iters)
        for iter in self.iters:
            self.all_psnrs[iter] = []
        #json_path=self.config.resource.generator_json_path
        #infos = json_path.split('generator')
        #infos = infos[1].split('.')
        #json_info = infos[0]
        #weights_path=self.config.resource.generator_weights_path
        #infos = weights_path.split('generator')
        #infos = infos[1].split('.')
        #weights_info = infos[0]
        #print(f'json/weight:{json_info}/{weights_info}')
        print(f'test strategy:{self.iters}')
        self.test()

    def __compute_psnr(self, x , label , max_diff):
        mse =  np.mean(( x - label ) **2 )
        return 10*math.log10( max_diff**2 / mse )

    def __doBatchTest(self):
        n = len(self.pyramid_blurs)
        for iter in self.iters:
            batch_blurs2x = []
            batch_blurs1x = []
            for i in range(iter,0,-1):
                if(i == iter):#first iter
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
                #generate batch_blurs1x
                for j in range(n):
                    pyramid_blur = self.pyramid_blurs[j]
                    imageBlur1x = pyramid_blur[i-1]
                    batch_blurs1x.append(imageBlur1x)
                #data prepare end
                
                #predict 2x
                data_X1 = np.concatenate((batch_blurs2x,batch_gen), axis=3)#6channels
                data_X = {'imageSmall':data_X1,'imageUp':np.array(batch_blurs1x)}
                batch_gen = self.model.generator.predict(data_X)
            #calculate metrics
            for i in range(n):
                pImage = batch_gen[i]
                pImage = pImage[24:744]
                psnr = self.__compute_psnr(pImage, self.batch_sharps[i], 1)
                self.all_psnrs[iter].append(psnr)
        #reset
        self.current_size = 0
        self.pyramid_blurs = []
        self.batch_sharps = []

    def __doInteration(self,blur,sharp):
        #self.sharpness.append(self.measure.getScore(blur))
        if(self.current_size < self.batch_size):
            blur = np.pad(blur,((24,24),(0,0),(0,0)),'reflect')#be divided by 256
            self.pyramid_blurs.append(tuple(transform.pyramid_gaussian(blur, downscale=2, max_layer=self.max_iter, multichannel=True)))
            self.batch_sharps.append(sharp)
            self.current_size += 1
        if(self.current_size == self.batch_size):#train a batch
            self.__doBatchTest()

    def test(self):
        dataHelper = DataHelper()
        dataHelper.load_data(self.config.resource.test_directory_path,0)
        
        blurSharpParis = dataHelper.getLoadedPairs()
        for imageBlur,imageSharp in blurSharpParis:
            self.__doInteration(imageBlur,imageSharp)
        if(self.pyramid_blurs):
            self.__doBatchTest()
        
        #analyse results
        psnrs = []
        for iter in self.iters:
            psnrs.append(self.all_psnrs[iter])
        psnrs = np.array(psnrs)
        psnrs_by_iter = np.mean(psnrs,axis=1)
        for i in range(len(psnrs_by_iter)):
            print(f'PSNR:{psnrs_by_iter[i]}@{self.iters[i]}')
        best_psnrs = np.amax(psnrs,axis=0)
        path=os.path.join(self.config.resource.output_dir, "psnrs.pkl")
        with open(path, 'wb') as pfile:
          pickle.dump(best_psnrs, pfile, protocol=pickle.HIGHEST_PROTOCOL)
        best_iters_index = np.argmax(psnrs,axis=0)
        iters = np.array(self.iters)
        best_iters = iters[best_iters_index]
        path=os.path.join(self.config.resource.output_dir, "iters.pkl")
        with open(path, 'wb') as pfile:
          pickle.dump(best_iters, pfile, protocol=pickle.HIGHEST_PROTOCOL)
        calculate_data_n = len(best_psnrs)
        #path=os.path.join(self.config.resource.output_dir, "sharpness.pkl")
        #with open(path, 'wb') as pfile:
        #  pickle.dump(self.sharpness, pfile, protocol=pickle.HIGHEST_PROTOCOL)
        calculate_data_n = len(best_psnrs)
        print(f'{calculate_data_n}/{len(blurSharpParis)} done! Average PSNRs(Best):{np.mean(best_psnrs)}')