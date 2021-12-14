from ..flyingthings3d import FlyingThings3D, DataConf


class FlyingThings3DTrain(FlyingThings3D):

    def __init__(self, root=None, aug_fcts=None, to_torch=False, verbose=True):

        split = 'train'
        type = 'flow'
        sample_confs = self._get_sample_confs()

        super().__init__(sample_confs=sample_confs, split=split, type=type, root=root,
                         aug_fcts=aug_fcts, to_torch=to_torch, verbose=verbose)

    def _get_sample_confs(self):
        sample_conf = {
            'images': [DataConf('frames_cleanpass', 'left', 0), DataConf('frames_cleanpass', 'left', 1)],
            'gt_flow': DataConf('flows_into_future', 'left', 0),
        }

        # TODO: add more e.g. backward flow

        return [sample_conf]


class FlyingThings3DTest(FlyingThings3D):

    def __init__(self, root=None, aug_fcts=None, to_torch=False, verbose=True):

        split = 'test'
        type = 'flow'
        sample_confs = self._get_sample_confs()

        super().__init__(sample_confs=sample_confs, split=split, type=type, root=root,
                         aug_fcts=aug_fcts, to_torch=to_torch, verbose=verbose)

    def _get_sample_confs(self):
        sample_conf = {
            'images': [DataConf('frames_cleanpass', 'left', 0), DataConf('frames_cleanpass', 'left', 1)],
            'gt_flow': DataConf('flows_into_future', 'left', 0),
        }

        return [sample_conf]
