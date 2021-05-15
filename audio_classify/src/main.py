import argparse
import json

from pre_process import PreProcess
from mlp_classifier import MlpClassifier
from pre_process_youtube import PreProcessYoutube


class Main:

    def run(self):
        """[read the arguments passed to check if is to train model, to run preprocess or run both]
        """        
        config_json = self._open_config()
        my_parser = argparse.ArgumentParser(description='Model to classify if is speech or music')
        my_parser.add_argument('-m','--model', required = False, action='store_true')
        my_parser.add_argument('-d','--data',  required = False, action='store_true')
        my_parser.add_argument('-y','--youtube',  required = False, action='store_true')
        args = my_parser.parse_args()
        if(args.data):
            preProcess = PreProcess(config_json)
            preProcess.run()
        if(args.model):
            mlpClassifier = MlpClassifier(config_json)
            mlpClassifier.run()
        if(args.youtube):
            preProcessYoutube = PreProcessYoutube(config_json)
            preProcessYoutube.run()

    def _open_config(self):
        """[Open json file with all parameters to train model and preprocess]
        """        
        with open('config.json') as json_file:
            return json.load(json_file)


if __name__ == "__main__":
    main = Main()
    main.run()
