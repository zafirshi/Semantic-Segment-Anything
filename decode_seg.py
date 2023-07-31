import json

import numpy as np
import pycocotools.mask as maskutil


def decode_mask_from_json(json_file: str):
    with open(json_file) as json_file:
        data = json.load(json_file)
        # contain 3 keys
        # - 'annotations': (list of dict) All the sub-areas come from SSA
        # - 'semantic_mask': (dict) The semantic merged sub-areas' masks from SSA
        # - 'segformer' : (dict) The semantic segmentation mask from Segformer
        mask_list = []
        for i in data['semantic_mask'].keys():
            if i in selected_id2label:
                mask = maskutil.decode(data['semantic_mask'][i])
                # TODO: add weight to each mask
                mask_list.append(mask)
        mask_array = np.array(np.stack(mask_list, axis=-1), dtype=np.uint8)
        # mask = maskutil.decode(data['semantic_mask']['1'])
        # mask2 = maskutil.decode(data['segformer']['0'])

    return mask_array


if __name__ == '__main__':
    # file_path = '/media/zafirshi/software/Code/Semantic-Segment-Anything/data/pitts250k/train/queries_compare/@0584468.67@4477301.78@17@T@040.44207@-080.00399@005007@00@@@@@@pitch1_yaw1@_semantic.json'
    file_path = 'data/msls/train/queries_compare/@0241821.78@9862147.68@37@M@-01.24616@0036.67991@Oz2hnuwJn1QUBN95Y8SGZQ@@@@@@20180527@day_forward_nairobi@_semantic.json'
    decode_mask_from_json(file_path)
