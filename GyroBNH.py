import hydra
from omegaconf import DictConfig

from RieNets.hnns.train_kfold import train_kfold

class Args:
    """ a Struct Class  """
    pass
args=Args()
args.config_name='GyroBNH.yaml'


@hydra.main(config_path='./conf/', config_name=args.config_name, version_base='1.1')
def main(cfg: DictConfig):
    train_kfold(cfg,args)

if __name__ == '__main__':
    main()