import hydra


@hydra.main(config_path='../conf/', config_name='defaults.yaml')
def main(cfg):
    print(cfg)

if __name__ == '__main__':
    main()
