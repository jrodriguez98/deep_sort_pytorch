import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging
import PIL

from .model_sq import Net

class Extractor(object):
    def __init__(self, model_path, num_classes=34, apply_pad=False, use_cuda=True):
        self.net = Net(num_classes=num_classes, reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (128, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.apply_pad = apply_pad

    def _resize_and_pad(self, image, size):
        old_size = image.shape[:2]
        ratio = float(size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        image = PIL.Image.fromarray(image)
        im = image.resize(new_size, PIL.Image.ANTIALIAS)

        new_im = PIL.Image.new("RGB", (size, size))
        new_im.paste(im, ((size - new_size[0]) // 2,
                          (size - new_size[1]) // 2))

        return new_im

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        if self.apply_pad:
            images_pre = [self.norm(self._resize_and_pad(im, self.size[0])).unsqueeze(0) for im in im_crops]
        else:
            images_pre = [self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops]

        im_batch = torch.cat(images_pre, dim=0).float()

        return im_batch


    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:,:,(2,1,0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)

