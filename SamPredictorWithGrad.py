import segment_anything as sa
import torch
from segment_anything import SamPredictor
from segment_anything.modeling import Sam
from typing import Any, Dict, List, Optional, Tuple

class SamPredictor_WithGrad(SamPredictor):

    #Overrides original predict_torch method but without the no_grad call. Keeps all other attributes
    # and methods of the original SamPredictor class the same
    def predict_torch(
            self,
            point_coords: Optional[torch.Tensor],
            point_labels: Optional[torch.Tensor],
            boxes: Optional[torch.Tensor] = None,
            mask_input: Optional[torch.Tensor] = None,
            multimask_output: bool = True,
            return_logits: bool = False,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        # if not self.is_image_set:
        #    raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        test = self.model

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )


        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks