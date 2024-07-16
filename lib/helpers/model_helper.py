from lib.models.gupnet import GUPNet


def build_model(cfg,mean_size):
    if cfg['model']['type'] == 'gupnet':
        return GUPNet(backbone=cfg['model']['backbone'], neck=cfg['model']['neck'], mean_size=mean_size, cfg= cfg, downsample=cfg['model']['downsample'])
    else:
        raise NotImplementedError("%s model is not supported" % cfg['type'])


def build_teacher_model(cfg,mean_size):
    if cfg['teacher_model']['type'] == 'gupnet':
        return GUPNet(backbone=cfg['teacher_model']['backbone'], neck=cfg['teacher_model']['neck'], mean_size=mean_size, cfg= cfg, downsample=cfg['teacher_model']['downsample'])
    else:
        raise NotImplementedError("%s model is not supported" % cfg['type'])
