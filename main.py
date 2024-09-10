import models
import utils

import argparse

models_dict = {
    'convit': models.ConViT,
    'convmixer': models.convmixer_1024_20_ks9_p14,
    'densenet': models.DenseNet,
    'vit': models.ViT,
    'vim': models.Vim,
    'vimhybrid': models.VimRes,
}

def pipeline (model_name, mode, *args):
    if model_name not in models_dict:
        raise ValueError(f"Model '{model_name}' not recognized. Available options: {list(models_dict.keys())}")
    
    model_class = models_dict[model_name]

    if mode == 'train':
        utils.train_model(model_class, *args)
    elif mode == 'test':
        utils.test_model(model_class, *args)
    else:
        raise ValueError("Mode must be `train` or `test`")

def main():
    parser = argparse.ArgumentParser(description='Train or test a model.')
    parser.add_argument('model', type=str, choices=models_dict.keys(), help='The model to use.')
    parser.add_argument('mode', type=str, choices=['train', 'test'], help='Whether to train or test the model.')

    args = parser.parse_args()

    pipeline(args.model, args.mode)

if __name__ == '__main__':
    main()