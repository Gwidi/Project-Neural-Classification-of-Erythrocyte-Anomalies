import numpy as np

class AutoCrop:
    """Automatically crop black borders from image"""
    def __init__(self, threshold=10):
        self.threshold = threshold
    
    def __call__(self, img):
        img_array = np.array(img)
        
        if len(img_array.shape) == 3:
            mask = img_array.sum(axis=2) > self.threshold
        else:
            mask = img_array > self.threshold
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            return img.crop((cmin, rmin, cmax + 1, rmax + 1))
        return img