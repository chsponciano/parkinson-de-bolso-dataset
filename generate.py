# !curl -LO https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.2/mask_rcnn_coco.h5

import numpy as np
import skimage.io
import pixellib
import cv2
import os
from pixellib.instance import instance_segmentation


_current_path = os.getcwd()
_untreated_data = _current_path + '\\untreated\\'
_parkinson_data = _current_path + '\\parkinson\\'
_dimension = (320, 240) 

_instance_seg = instance_segmentation()
_instance_seg.load_model('mask_rcnn_coco.h5')
_target_classes = _instance_seg.select_target_classes(person=True)

for _code in range(1, 10):
    _current = _untreated_data + f'dp_{_code}.mp4'
    print(f'=== Processing: {_current}... ===')

    _capture = cv2.VideoCapture(_current)
    _index = 0

    if not _capture.isOpened(): 
        print("=== Error opening video stream or file ===")

    while _capture.isOpened():
        _continue, _frame = _capture.read()
        if _continue:            
            cv2.imwrite('temp.png', _frame) 
            _mask, _ = _instance_seg.segmentImage('temp.png', segment_target_classes=_target_classes, extract_segmented_objects=True)
            _mask = _mask['masks']
            _mask = _mask.astype(int)

            for i in range(_mask.shape[2]):
                for j in range(_frame.shape[2]):
                    _frame[:,:,j] = _frame[:,:,j] * _mask[:,:,i]
                _, _frame = cv2.threshold(_frame, 0, 255, cv2.THRESH_BINARY)

            _output = cv2.resize(_frame, _dimension, interpolation=cv2.INTER_AREA) 

            _index += 1
            _filename = f'img_{_code}_{_index}.png'
            _path = f'{_parkinson_data}\\{_code}\\{_filename}'
            cv2.imwrite(_path, _output) 
            print(f'Save: {_path}')

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            _capture.release()
            cv2.destroyAllWindows()
            break