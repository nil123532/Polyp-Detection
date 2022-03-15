import copy, os, sys

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

from paddle import fluid

from ppdet.utils.eval_utils import parse_fetches
from ppdet.core.workspace import load_config, merge_config, create
from ppdet.utils.check import check_gpu, check_version, check_config, enable_static_mode
from ppdet.data.reader import create_reader
import ppdet.utils.checkpoint as checkpoint

import cv2
import numpy as np
import time

class Compose(object):
    def __init__(self, transforms, ctx=None):
        self.transforms = transforms
        self.ctx = ctx

    def __call__(self, data):
        ctx = self.ctx if self.ctx else {}
        for f in self.transforms:
            try:
                data = f(data, ctx)
            except Exception as e:
                #stack_info = traceback.format_exc()
                #logger.warn("fail to map op [{}] with error: {} and stack:\n{}".
                #            format(f, e, str(stack_info)))
                raise e
        return data

class ObjectDetector():
    def __init__(self, threshold=0.5, config="configs/ppyolo_2x_polyp.yml", opt={ "use_gpu": True, "weights": "weights/60000.pdparams" }):
        self.threshold = threshold

        enable_static_mode()

        cfg = load_config(config)

        merge_config(opt)
        check_config(cfg)
        # check if set use_gpu=True in paddlepaddle cpu version
        check_gpu(cfg.use_gpu)
        # check if paddlepaddle version is satisfied
        check_version()

        main_arch = cfg.architecture

        dataset = cfg.TestReader['dataset']

        place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
        self.exe = fluid.Executor(place)

        model = create(main_arch)

        startup_prog = fluid.Program()
        infer_prog = fluid.Program()
        with fluid.program_guard(infer_prog, startup_prog):
            with fluid.unique_name.guard():
                inputs_def = cfg['TestReader']['inputs_def']
                inputs_def['iterable'] = True
                feed_vars, loader = model.build_inputs(**inputs_def)
                test_fetches = model.test(feed_vars)
        self.infer_prog = infer_prog.clone(True)

        #reader = create_reader(cfg.TestReader, devices_num=1)
        #loader.set_sample_list_generator(reader, place)

        _fields = copy.deepcopy(inputs_def[
                                    'fields']) if inputs_def else None
        self._sample_transforms = Compose(cfg.TestReader["sample_transforms"],
                                     {'fields': _fields})

        self.exe.run(startup_prog)
        if cfg.weights:
            checkpoint.load_params(self.exe, self.infer_prog, cfg.weights)

        assert cfg.metric in ['COCO', 'VOC', 'OID', 'WIDERFACE'], \
            "unknown metric type {}".format(cfg.metric)
        extra_keys = []
        if cfg['metric'] in ['COCO', 'OID']:
            extra_keys = ['im_info', 'im_id', 'im_shape']
        if cfg['metric'] == 'VOC' or cfg['metric'] == 'WIDERFACE':
            extra_keys = ['im_id', 'im_shape']
        self.keys, self.values, _ = parse_fetches(test_fetches, self.infer_prog, extra_keys)

        self.is_bbox_normalized = False
        if hasattr(model, 'is_bbox_normalized') and \
                callable(model.is_bbox_normalized):
            self.is_bbox_normalized = model.is_bbox_normalized()

        #self.observations = []

    def apply_model(self, image=None):
        #start = time.time()
        #image = cv2.imread('/home/david/Downloads/CVC-ClinicDB/Images_png/1.png')

        image = np.array(image)
        # Convert RGB to BGR
        #image = image[:, :, ::-1].copy()

        #end1 = time.time()

        height, width, channel = image.shape
        #width, height = image.size

        #_, enc_image = cv2.imencode(".bmp", image)

        image2 = self._sample_transforms({"im_id": [0], "image": image, "im_size": [[height, width]]})

        #end2 = time.time()

        outs = self.exe.run(self.infer_prog,
                       feed={"im_id": [0], "image": [image2["image"]], "im_size": [[height, width]]},
                       fetch_list=self.values,
                       return_numpy=False)

        #end3 = time.time()


        #res = {
        #    k: (np.array(v), v.recursive_sequence_lengths())
        #    for k, v in zip(self.keys, outs)
        #}

        #end4 = time.time()

        #bboxes = []


        #print("res:")
        #print(np.array(outs[0]))



        #for a in res['bbox'][0]:
        #    clsid, score, xmin, ymin, xmax, ymax = a.tolist()

        #    clsid = int(clsid)
        #    xmin = int(xmin)
        #    ymin = int(ymin)
        #    xmax = int(xmax)
        #    ymax = int(ymax)

        #    if score >= self.threshold:
        #        bboxes.append({ "class": clsid, "score": score, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax })

        #end5 = time.time()

        #print("1: " + str(end1 - start))
        #print("2: " + str(end2 - start))
        #print("3: " + str(end3 - start))
        #print("4: " + str(end4 - start))
        #print("5: " + str(end5 - start))

        #self.observations.append(bboxes)

        return np.array(outs[0])