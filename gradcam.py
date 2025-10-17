import cv2
import numpy as np

class GradCAM():
    """
    Class to implement the GradCam function with it's necessary Pytorch hooks.

    Attributes
    ----------
    model : detectron2 GeneralizedRCNN Model
        A model using the detectron2 API for inferencing
    layer_name : str
        name of the convolutional layer to perform GradCAM with
    """

    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.activations = None
        self.gradient = None
        self.model.eval()
        self.activations_grads = []
        self._register_hook()
        self.model.zero_grad()

    def _get_activations_hook(self, module, input, output):
        self.activations = output

    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.model.named_modules():
            print(name + " " + str(self.target_layer_name))
            if name == self.target_layer_name:
                self.activations_grads.append(module.register_forward_hook(self._get_activations_hook))
                self.activations_grads.append(module.register_backward_hook(self._get_grads_hook))
                return True
        print(f"Layer {self.target_layer_name} not found in Model!")
        # assert False

    def _release_activations_grads(self):
      for handle in self.activations_grads:
            handle.remove()
    
    def _postprocess_cam(self, raw_cam, img_width, img_height):
        cam_orig = np.sum(raw_cam, axis=0)  # [H,W]
        cam_orig = np.maximum(cam_orig, 0)  # ReLU
        cam_orig -= np.min(cam_orig)
        cam_orig /= np.max(cam_orig)
        cam = cv2.resize(cam_orig, (img_width, img_height))
        return cam, cam_orig
    
    def _forward_backward_pass(self, inputs, target_instance):
        self.model.zero_grad()
        output = self.model.forward(inputs)[0]
        instances = inputs[0]['instances']
        gt_classes = instances.gt_classes.cpu().data.numpy()
        gt_boxes = instances.gt_boxes.tensor.cpu().data.numpy()
        max_idx = [-1 for i in range(len(gt_classes))]
        output_instancs = output['instances']
        pred_resuts = {}
        pre_labels = []
        sort_labels = np.sort(gt_classes)
        for idx, gt_label in enumerate(zip(gt_classes)):
            pre = 0
            if gt_label in pre_labels:
                pre = max_idx[idx - 1]+1
            for idx_pred, (pred_score, pred_label) in enumerate( zip(output_instancs.scores.detach().cpu().numpy(), \
                                            output_instancs.pred_classes.cpu().numpy() ) ):
                if idx_pred < pre:
                    continue

                if gt_label != pred_label:
                    continue
                if max_idx[idx] == -1:
                    max_idx[idx] = idx_pred
                elif pred_score > output_instancs.scores[max_idx[idx]]:
                    max_idx[idx] = idx_pred
            pre_labels.append(gt_label)
        score = None
        for idx, idx_pred in enumerate(max_idx):
            if idx_pred == -1:
                continue
            score = output['instances'].scores[idx_pred] if score is None else score + output['instances'].scores[idx_pred] 
            if idx not in pred_resuts.keys():
                pred_resuts[idx] = []
            pred_resuts[idx] = (idx_pred, output_instancs.pred_boxes[idx_pred])
        if score is None and len(output['instances'].scores) > 0:
            target_instance =  np.argmax(output['instances'].scores.cpu().data.numpy(), axis=-1)
            assert len(output['instances']) >= target_instance, f"Only {len(output['instances'])} objects found but you request object number {target_instance}"
            score = output['instances'].scores[target_instance]
        if score is not None:
            score.backward()
        return output, pred_resuts
    

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._release_activations_grads()

    def __call__(self, inputs, target_instance):
        """
        Calls the GradCAM++ instance

        Parameters
        ----------
        inputs : dict
            The input in the standard detectron2 model input format
            https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-input-format

        target_instance : int, optional
            The target category index. If `None` the highest scoring class will be selected

        Returns
        -------
        cam : np.array()
          Gradient weighted class activation map
        output : list
          list of Instance objects representing the detectron2 model output
        """

        output, results = self._forward_backward_pass(inputs, target_instance)
        activations = self.activations[0].cpu().data.numpy()  # [C,H,W]

        if self.gradient is not None and 0:

            gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
            weight = np.mean(gradient, axis=(1, 2))  # [C]
            
            #time activation(C,H,W) with weights (C,)
            cam = activations * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        else:
            cam = activations

        cam, cam_orig = self._postprocess_cam(cam, inputs[0]["image"].shape[-2], inputs[0]["image"].shape[-1])

        return cam, cam_orig, output, results

class GradCamPlusPlus(GradCAM):
    """
    Subclass to implement the GradCam++ function with it's necessary PyTorch hooks.
    ...

    Attributes
    ----------
    model : detectron2 GeneralizedRCNN Model
        A model using the detectron2 API for inferencing
    target_layer_name : str
        name of the convolutional layer to perform GradCAM++ with

    """
    def __init__(self, model, target_layer_name):
        super().__init__(model, target_layer_name)

    def __call__(self, inputs, target_instance):
        """
        Calls the GradCAM++ instance

        Parameters
        ----------
        inputs : dict
            The input in the standard detectron2 model input format
            https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-input-format

        target_instance : int, optional
            The target category index. If `None` the highest scoring class will be selected

        Returns
        -------
        cam : np.array()
          Gradient weighted class activation map
        output : list
          list of Instance objects representing the detectron2 model output
        """

        output = self._forward_backward_pass(inputs, target_instance)
        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        activations = self.activations[0].cpu().data.numpy()  # [C,H,W]

        #from https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/grad_cam_plusplus.py
        grads_power_2 = gradient**2
        grads_power_3 = grads_power_2 * gradient
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = np.sum(activations, axis=(1, 2))
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 +
                               sum_activations[:, None, None] * grads_power_3 + eps)
        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(gradient != 0, aij, 0)

        weights = np.maximum(gradient, 0) * aij
        weight = np.sum(weights, axis=(1, 2))

        cam = activations * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        # cam, cam_orig = self._postprocess_cam(cam, inputs[0]["width"], inputs[0]["height"])
        cam, cam_orig = self._postprocess_cam(cam, inputs[0]['image'].shape[-1], inputs[0]['image'].shape[-2])

        return cam, cam_orig, output