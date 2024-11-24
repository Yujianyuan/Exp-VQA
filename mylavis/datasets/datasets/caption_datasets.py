"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from mylavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
import json
import torch

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


class CaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
        
        self.mask_dict = json.load(open('/home/yuanyujian/large_model/blipvqa/mask_file/gft_mask_196.json','r'))

    # def __getitem__(self, index):

    #     # TODO this assumes image input, not general enough
    #     ann = self.annotation[index]

    #     image_path = os.path.join(self.vis_root, ann["image"])
    #     image = Image.open(image_path).convert("RGB")

    #     image = self.vis_processor(image)
    #     '''if "caption" in ann.keys(): #for OPT
    #         caption = self.text_processor(ann["caption"])

    #         return {
    #             "image": image,
    #             "text_input": caption,
    #             "image_id": self.img_ids[ann["image_id"]],
    #         }
    #     else: #for FlanT5
    #         txt_input = self.text_processor(ann["text_input"])
    #         txt_output = self.text_processor(ann["text_output"])

    #         return {
    #             "image": image,
    #             "text_input": txt_input,
    #             "text_output": txt_output,
    #             "image_id": self.img_ids[ann["image_id"]],
    #         }'''
    #     text_input = self.text_processor(ann["text_input"])
    #     text_output = self.text_processor(ann["text_output"])
    #     #empty_str = ''
    #     return {
    #         "image": image,
    #         "text_input": text_input,
    #         "text_output": text_output,
    #         "image_id": self.img_ids[ann["image_id"]],
    #     }
    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        text_input = self.text_processor(ann["text_input"])
        text_output = self.text_processor(ann["text_output"])
        area = ann["area"]
        mask = self.mask_dict[ann["image"]][area]
        mask = torch.tensor(mask).to(torch.float32)
        #empty_str = ''
        return {
            "image": image,
            "mask": mask,
            "text_input": text_input,
            "text_output": text_output,
            "image_id": self.img_ids[ann["image_id"]],
        }



class CaptionEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        return {
            "image": image,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }
