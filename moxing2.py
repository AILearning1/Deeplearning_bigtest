import numpy as np
from PIL import Image
from sklearn.datasets import fetch_openml  # for fetching if needed
from typing import Tuple, Optional

# =====================
# Data type
# =====================
DTYPE = np.float32

# ------------------
# Helper functions for im2col
# ------------------
def get_im2col_indices(x_shape: Tuple[int,int,int,int], fh:int, fw:int, pad:int, stride:int):
    N,C,H,W = x_shape
    out_h = (H + 2*pad - fh)//stride + 1
    out_w = (W + 2*pad - fw)//stride + 1
    i0 = np.repeat(np.arange(fh), fw)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_h), out_w)
    j0 = np.tile(np.arange(fw), fh*C)
    j1 = stride * np.tile(np.arange(out_w), out_h)
    i = i0.reshape(-1,1) + i1.reshape(1,-1)
    j = j0.reshape(-1,1) + j1.reshape(1,-1)
    k = np.repeat(np.arange(C), fh*fw).reshape(-1,1)
    return k.astype(int), i.astype(int), j.astype(int)

def im2col(x: np.ndarray, fh:int, fw:int, pad:int, stride:int) -> np.ndarray:
    x_p = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant')
    k,i,j = get_im2col_indices(x.shape, fh, fw, pad, stride)
    cols = x_p[:, k, i, j]
    C = x.shape[1]
    return cols.transpose(1,2,0).reshape(fh*fw*C, -1)

# ------------------
# Layer definitions
# ------------------
class Conv2D:
    def __init__(self, in_c, out_c, k, stride=1, pad=0):
        self.stride, self.pad = stride, pad
        self.W = np.zeros((out_c, in_c, k, k), dtype=DTYPE)
        self.b = np.zeros(out_c, dtype=DTYPE)
    def forward(self, x: np.ndarray) -> np.ndarray:
        N,C,H,W = x.shape
        cols = im2col(x, self.W.shape[2], self.W.shape[3], self.pad, self.stride)
        W_col = self.W.reshape(self.W.shape[0], -1)
        out = W_col @ cols + self.b.reshape(-1,1)
        H_out = (H + 2*self.pad - self.W.shape[2]) // self.stride + 1
        W_out = (W + 2*self.pad - self.W.shape[3]) // self.stride + 1
        return out.reshape(self.W.shape[0], H_out, W_out, N).transpose(3,0,1,2)

class MaxPool2D:
    def __init__(self, k, stride=None):
        self.k = k; self.stride = stride or k
    def forward(self, x: np.ndarray) -> np.ndarray:
        N,C,H,W = x.shape
        x_r = x.reshape(N*C,1,H,W)
        cols = im2col(x_r, self.k, self.k, 0, self.stride)
        out = cols.max(axis=0)
        H_out = (H-self.k)//self.stride+1; W_out=(W-self.k)//self.stride+1
        return out.reshape(H_out,W_out,N,C).transpose(2,3,0,1)

class ReLU:
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

class Linear:
    def __init__(self, in_f, out_f):
        self.W = np.zeros((in_f, out_f), dtype=DTYPE)
        self.b = np.zeros(out_f, dtype=DTYPE)
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + self.b

class Dropout:
    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        return x

# ------------------
# Manual AlexNet
# ------------------
class ManualAlexNet:
    def __init__(self, num_classes=10):
        self.features = [
            Conv2D(3,64,11,4,2), ReLU(), MaxPool2D(3,2),
            Conv2D(64,192,5,1,2), ReLU(), MaxPool2D(3,2),
            Conv2D(192,384,3,1,1), ReLU(),
            Conv2D(384,256,3,1,1), ReLU(),
            Conv2D(256,256,3,1,1), ReLU(), MaxPool2D(3,2)
        ]
        self.classifier = [
            Dropout(), Linear(256*6*6,4096), ReLU(),
            Dropout(), Linear(4096,4096), ReLU(),
            Linear(4096,num_classes)
        ]
    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.features:
            x = layer.forward(x)
        x = x.reshape(x.shape[0], -1)
        for layer in self.classifier:
            x = layer.forward(x)
        return x

# ------------------
# Load pretrained weights (keys autodetected)
# ------------------
def load_model_weights(model: ManualAlexNet, path: str = 'manual_alexnet_model.npz'):
    weights = np.load(path)
    # inspect saved keys: layer{idx}_W, layer{idx}_b
    layer_map = model.features + model.classifier
    for key in weights.files:
        if key.endswith('_W'):
            idx = int(key.split('_')[0][5:])  # parse layer index
            layer = layer_map[idx]
            layer.W = weights[key]
            layer.b = weights[f'layer{idx}_b']

# ------------------
# Preprocess single image, arbitrary resolution
# ------------------
def preprocess_image(path: str) -> np.ndarray:
    im = Image.open(path).convert('L')
    im = im.resize((224,224), Image.BILINEAR)
    arr = np.array(im, dtype=DTYPE) / 255.0
    arr = (arr - 0.5) / 0.5
    x = np.stack([arr,arr,arr], axis=0)
    return x[np.newaxis, ...]  # (1,3,224,224)

# ------------------
# Main inference
# ------------------
if __name__ == '__main__':
    # Initialize and load weights
    model = ManualAlexNet()
    load_model_weights(model, 'manual_alexnet_model_numpy_metrics_1.npz')

    # Preprocess your JPG (e.g., 1136x1136)
    img_np = preprocess_image('test3.jpg')  # replace with your file

    # Forward pass
    scores = model.forward(img_np)
    pred = int(np.argmax(scores, axis=1)[0])
    print(f'Predicted digit: {pred}')

#测试模型