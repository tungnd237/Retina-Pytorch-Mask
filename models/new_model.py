import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from torchvision.ops import roi_align
from collections import OrderedDict

from models.net import MobileNetV1 as MobileNetV1
from models.net import FPN as FPN
from models.net import SSH as SSH

class ClassHead(nn.Module):
    def __init__(self, in_channels: int = 512, num_anchors: int = 3) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


class MaskHead(nn.Module):
    def __init__(self, in_channels: int = 512, num_anchors: int = 3, num_properties: int = 1) -> None:
        super().__init__()
        self.num_properties = num_properties
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * num_properties, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, self.num_properties)


class BboxHead(nn.Module):
    def __init__(self, in_channels: int = 512, num_anchors: int = 3):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, in_channels: int = 512, num_anchors: int = 3):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):
    def __init__(self, cfg = None, phase = 'train', num_properties: int = 1):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace,self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load("./weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, in_channels=out_channels)
        self.MaskHead = self._make_mask_head(
            fpn_num=3, in_channels=out_channels, num_properties=num_properties
        )
        self.BboxHead = self._make_bbox_head(fpn_num=3, in_channels=out_channels)
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, in_channels=out_channels)

    def _make_class_head(self, fpn_num: int = 3, in_channels: int = 64, anchor_num: int = 2) -> nn.ModuleList:
        classhead = nn.ModuleList()
        for _ in range(fpn_num):
            classhead.append(ClassHead(in_channels, anchor_num))
        return classhead

    def _make_mask_head(self,
        fpn_num: int = 3, in_channels: int = 64, anchor_num: int = 2, num_properties: int = 1
    ) -> nn.ModuleList:
        classhead = nn.ModuleList()
        for _ in range(fpn_num):
            classhead.append(MaskHead(in_channels, anchor_num, num_properties))
        return classhead

    def _make_bbox_head(self, fpn_num: int = 3, in_channels: int = 64, anchor_num: int = 2) -> nn.ModuleList:
        bboxhead = nn.ModuleList()
        for _ in range(fpn_num):
            bboxhead.append(BboxHead(in_channels, anchor_num))
        return bboxhead
    
    def _make_landmark_head(self, fpn_num: int = 3, in_channels: int = 64, anchor_num: int = 2) -> nn.ModuleList:
        landmarkhead = nn.ModuleList()
        for _ in range(fpn_num):
            landmarkhead.append(LandmarkHead(in_channels, anchor_num))
        return landmarkhead

    def convert_bbox_format_and_scale(self, bbox_regressions, feature_maps, input_tensor):
        """
        Adjust bounding boxes based on dynamic input and feature map sizes.
        
        Parameters:
        - bbox_regressions: Tensor of bounding box predictions.
        - feature_maps: List of feature map tensors from the model.
        - input_tensor: The original input tensor to the model.
        
        Returns:
        - A tensor of scaled and formatted bounding boxes.
        """
        batch_size = input_tensor.shape[0]
        
        # Assuming all feature maps have the same size, take the size from the first one
        feature_map_size = feature_maps[0].shape[-2:]  # [H, W]
        
        # Input image size
        input_size = input_tensor.shape[-2:]  # [H, W]
        
        # Calculate the scale factor for both height and width
        scale_factor_h = feature_map_size[0] / input_size[0]
        scale_factor_w = feature_map_size[1] / input_size[1]
        
        # Apply scaling to bbox regressions (assuming format [x1, y1, x2, y2])
        scaled_bboxes = bbox_regressions.clone()  # Clone to avoid modifying the original tensor
        scaled_bboxes[..., [0, 2]] *= scale_factor_w  # Scale x coordinates
        scaled_bboxes[..., [1, 3]] *= scale_factor_h  # Scale y coordinates
        
        # Generate batch indices for each bounding box
        batch_indices = torch.arange(batch_size, device=bbox_regressions.device).view(-1, 1).expand(-1, bbox_regressions.shape[1]).reshape(-1, 1)
        
        # Reshape and concatenate to format [batch_index, x1, y1, x2, y2]
        scaled_and_formatted_bboxes = torch.cat((batch_indices, scaled_bboxes.reshape(-1, 4)), dim=1)
        
        return scaled_and_formatted_bboxes

    def forward(self,inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        roi_boxes = self.convert_bbox_format_and_scale(bbox_regressions, features, inputs)

        # Perform ROI Align for each feature map
        roi_aligned_feature1 = roi_align(feature1, roi_boxes, output_size=(7, 7))
        roi_aligned_feature2 = roi_align(feature2, roi_boxes, output_size=(7, 7))
        roi_aligned_feature3 = roi_align(feature3, roi_boxes, output_size=(7, 7))
        features = [roi_aligned_feature1, roi_aligned_feature2, roi_aligned_feature3]
        
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications_mask = torch.cat(
            [self.MaskHead[i](feature) for i, feature in enumerate(features)], dim=1
        )
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == "train":
            output = (bbox_regressions, classifications, ldm_regressions, classifications_mask)
        else:
            output = (
                bbox_regressions,
                F.softmax(classifications, dim=-1),
                ldm_regressions,
                classifications_mask,
            )
        return output