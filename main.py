import os
import toml
import argparse
from   pprint import pprint

import torch
from   torch.backends import cudnn
from   torch.utils.data import DataLoader

import utils
from   utils import CONFIG
from   tester import Tester
from   dataloader.image_file import ImageFileTrain, ImageFileTest
from   dataloader.data_generator import DataGenerator

def main():

    # Train or Test
    if CONFIG.phase.lower() == "train":
        raise RuntimeError("Training Code Will be Avaliable after Paper is Accepted. \
            If You Don't Want to Use FP16, You Can Just Try the Training Code of GCA-Matting.")

    elif CONFIG.phase.lower() == "test":
        CONFIG.log.logging_path += "_test"
        if CONFIG.test.alpha_path is not None:
            utils.make_dir(CONFIG.test.alpha_path)
        utils.make_dir(CONFIG.log.logging_path)

        # Create a logger
        logger = utils.get_logger(CONFIG.log.logging_path,
                                  logging_level=CONFIG.log.logging_level)

        test_image_file = ImageFileTest(alpha_dir=CONFIG.test.alpha,
                                        merged_dir=CONFIG.test.merged,
                                        trimap_dir=CONFIG.test.trimap)
        test_dataset = DataGenerator(test_image_file, phase='test', test_scale=CONFIG.test.scale)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=CONFIG.test.batch_size,
                                     shuffle=False,
                                     num_workers=CONFIG.data.workers,
                                     drop_last=False)

        tester = Tester(test_dataloader=test_dataloader)
        tester.test()

    else:
        raise NotImplementedError("Unknown Phase: {}".format(CONFIG.phase))


if __name__ == '__main__':
    print('Torch Version: ', torch.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--config', type=str, default='config/empty.toml')
    parser.add_argument('--local_rank', type=int, default=0)

    # Parse configuration
    args = parser.parse_args()
    with open(args.config) as f:
        utils.load_config(toml.load(f))

    # Check if toml config file is loaded
    if CONFIG.is_default:
        raise ValueError("No .toml config loaded.")

    CONFIG.phase = args.phase
    CONFIG.log.logging_path = os.path.join(CONFIG.log.logging_path, CONFIG.version)
    CONFIG.log.tensorboard_path = os.path.join(CONFIG.log.tensorboard_path, CONFIG.version)
    CONFIG.log.checkpoint_path = os.path.join(CONFIG.log.checkpoint_path, CONFIG.version)
    if CONFIG.test.alpha_path is not None:
        CONFIG.test.alpha_path = os.path.join(CONFIG.test.alpha_path, CONFIG.version)
    if args.local_rank == 0:
        print('CONFIG: ')
        pprint(CONFIG)
    CONFIG.local_rank = args.local_rank

    # Train or Test
    main()