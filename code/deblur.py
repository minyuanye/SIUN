import os
import sys
import argparse
from src.config import Config
from src.lib.tf_util import set_session_config

_PATH_ = os.path.dirname(os.path.dirname(__file__))

if _PATH_ not in sys.path:
    sys.path.append(_PATH_)



def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train the model", action="store_true", default=False)
    parser.add_argument("--test", help="test the model", action="store_true", default=False)
    parser.add_argument("--apply", help="use the model", action="store_true", default=False)
    parser.add_argument("--verify", help="verify the model", action="store_true", default=False)
    parser.add_argument("--gpu", help="test device list", default="0")
    parser.add_argument("--file-path", help="file path of the input image")
    parser.add_argument("--dir-path", help="dir path of the input images")
    parser.add_argument("--result-dir", help="deblur result dir of the input images")
    parser.add_argument("--iter", help="iter times", default=0, type=int)
    return parser.parse_args()
    
if __name__ == "__main__":
    args = getArgs()
    config = Config()
    config.resource.create_directories()
    if(args.file_path):
        config.application.deblurring_file_path = args.file_path
    if(args.dir_path):
        config.application.deblurring_dir_path = args.dir_path
    if(args.iter):
        config.application.iter = args.iter
    if(args.result_dir):
        config.application.deblurring_result_dir = args.result_dir
    set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True, device_list=args.gpu)
    gpus = args.gpu.split(",")
    config.trainer.gpu_num = len(gpus)
    if(args.train):
        #trainer
        from src.trainer import Trainer
        Trainer(config).start()
    elif(args.test):
        #tester
        from src.tester import Tester
        Tester(config).start()
    elif(args.apply):
        #application
        from src.application import Application
        Application(config).start()
    elif(args.verify):
        #verification
        from src.verification import Verification
        Verification(config).start()
    else:
        #info
        from src.model.model import DDModel
        model = DDModel(config)
        model.generator.summary(line_length=150)


