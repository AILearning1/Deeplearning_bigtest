import numpy as np
from sklearn.datasets import fetch_openml
from PIL import Image
from typing import Tuple

# =====================
# Data utils: custom Dataset & DataLoader
# =====================
class MNISTDataset:
    def __init__(self, train: bool = True):
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        data = mnist['data'].astype(np.uint8)
        targets = mnist['target'].astype(int)
        if train:
            self.X_raw = data[:60000]
            self.y = targets[:60000]
        else:
            self.X_raw = data[60000:]
            self.y = targets[60000:]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        img28 = self.X_raw[idx]
        label = self.y[idx]
        im = Image.fromarray(img28.reshape(28,28), 'L')
        im = im.resize((224,224), Image.BILINEAR)
        arr = np.array(im, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        x = np.stack([arr, arr, arr], axis=0)
        return x, label

class DataLoader:
    def __init__(self, dataset: MNISTDataset, batch_size: int = 64, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        for start in range(0, len(self.indices), self.batch_size):
            batch_idx = self.indices[start:start+self.batch_size]
            xs, ys = zip(*(self.dataset[i] for i in batch_idx))
            yield np.stack(xs, axis=0), np.array(ys, dtype=np.int64)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

# =====================
# Convolution helpers
# =====================
DTYPE = np.float32

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
    return k, i, j

def im2col(x: np.ndarray, fh:int, fw:int, pad:int, stride:int) -> np.ndarray:
    N,C,H,W = x.shape
    x_p = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant')
    k,i,j = get_im2col_indices((N,C,H,W), fh, fw, pad, stride)
    cols = x_p[:, k, i, j]
    return cols.transpose(1,2,0).reshape(fh*fw*C, -1)

def col2im(cols: np.ndarray, x_shape: Tuple[int,int,int,int], fh:int, fw:int, pad:int, stride:int) -> np.ndarray:
    N,C,H,W = x_shape
    H_p = H + 2*pad + stride - 1
    W_p = W + 2*pad + stride - 1
    x_p = np.zeros((N,C,H_p,W_p), dtype=DTYPE)
    k,i,j = get_im2col_indices(x_shape, fh, fw, pad, stride)
    cols_r = cols.reshape(C*fh*fw, -1, N).transpose(2,0,1)
    np.add.at(x_p, (slice(None), k, i, j), cols_r)
    if pad==0:
        return x_p[:,:,:H,:W]
    return x_p[:,:,pad:pad+H,pad:pad+W]

# =====================
# Layers
# =====================
class Conv2D:
    def __init__(self, in_c, out_c, k, stride=1, pad=0):
        self.stride, self.pad = stride, pad
        scale = np.sqrt(2/(in_c*k*k))
        self.W = np.random.randn(out_c, in_c, k, k).astype(DTYPE) * scale
        self.b = np.zeros(out_c, dtype=DTYPE)
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        self.cols = im2col(x, self.W.shape[2], self.W.shape[3], self.pad, self.stride)
        W_col = self.W.reshape(self.W.shape[0], -1)
        out = W_col @ self.cols + self.b.reshape(-1,1)
        N,C,H,W = self.x_shape
        H_out = (H + 2*self.pad - self.W.shape[2])//self.stride + 1
        W_out = (W + 2*self.pad - self.W.shape[3])//self.stride + 1
        return out.reshape(self.W.shape[0], H_out, W_out, N).transpose(3,0,1,2)
    def backward(self, dout: np.ndarray, lr: float) -> np.ndarray:
        F = self.W.shape[0]
        dout_r = dout.transpose(1,2,3,0).reshape(F, -1)
        dW = dout_r @ self.cols.T; dW = dW.reshape(self.W.shape)
        db = dout_r.sum(axis=1)
        W_col = self.W.reshape(F, -1)
        dcols = W_col.T @ dout_r
        dx = col2im(dcols, self.x_shape, self.W.shape[2], self.W.shape[3], self.pad, self.stride)
        np.clip(dW, -5,5, out=dW); np.clip(db,-5,5,out=db)
        self.W -= lr * dW; self.b -= lr * db
        return dx

class MaxPool2D:
    def __init__(self, k, stride=None):
        self.k = k; self.stride = stride or k
        self.argmax = None; self.x_shape = None
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        N,C,H,W = x.shape; x_r = x.reshape(N*C,1,H,W)
        cols = im2col(x_r, self.k, self.k, 0, self.stride)
        self.argmax = np.argmax(cols, axis=0)
        out = cols[self.argmax, np.arange(self.argmax.size)]
        H_out = (H-self.k)//self.stride+1; W_out=(W-self.k)//self.stride+1
        return out.reshape(H_out,W_out,N,C).transpose(2,3,0,1)
    def backward(self, dout: np.ndarray, lr: float) -> np.ndarray:
        dout_flat = dout.transpose(2,3,0,1).ravel()
        dcols = np.zeros((self.k*self.k, dout_flat.size), dtype=DTYPE)
        dcols[self.argmax, np.arange(self.argmax.size)] = dout_flat
        H_out, W_out = dout.shape[2], dout.shape[3]
        H_in = (H_out-1)*self.stride + self.k; W_in = (W_out-1)*self.stride + self.k
        dx_r = col2im(dcols, (self.x_shape[0]*self.x_shape[1],1,H_in,W_in), self.k, self.k, 0, self.stride)
        return dx_r.reshape(self.x_shape)

class ReLU:
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0; return x * self.mask
    def backward(self, dout: np.ndarray, lr: float) -> np.ndarray:
        return dout * self.mask

class Linear:
    def __init__(self, in_f, out_f):
        scale = np.sqrt(2/in_f)
        self.W = np.random.randn(in_f,out_f).astype(DTYPE) * scale
        self.b = np.zeros(out_f, dtype=DTYPE)
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x; return x @ self.W + self.b
    def backward(self, dout: np.ndarray, lr: float) -> np.ndarray:
        dx = dout @ self.W.T; dW = self.x.T @ dout; db = dout.sum(axis=0)
        np.clip(dW,-5,5,out=dW); np.clip(db,-5,5,out=db)
        self.W -= lr*dW; self.b -= lr*db; return dx

class Dropout:
    def __init__(self,p=0.5): self.p=p; self.mask=None
    def forward(self, x, training=True):
        if training: self.mask=(np.random.rand(*x.shape)<self.p)/self.p; return x*self.mask
        return x
    def backward(self, dout, lr): return dout*self.mask

class ManualAlexNet:
    def __init__(self, num_classes=10):
        self.features=[Conv2D(3,64,11,4,2),ReLU(),MaxPool2D(3,2),Conv2D(64,192,5,1,2),ReLU(),MaxPool2D(3,2),Conv2D(192,384,3,1,1),ReLU(),Conv2D(384,256,3,1,1),ReLU(),Conv2D(256,256,3,1,1),ReLU(),MaxPool2D(3,2)]
        self.classifier=[Dropout(0.5),Linear(256*6*6,4096),ReLU(),Dropout(0.5),Linear(4096,4096),ReLU(),Linear(4096,num_classes)]
    def forward(self, x, training=True):
        for layer in self.features: x=layer.forward(x) if not isinstance(layer,Dropout) else layer.forward(x,training)
        x=x.reshape(x.shape[0],-1)
        for layer in self.classifier: x=layer.forward(x) if not isinstance(layer,Dropout) else layer.forward(x,training)
        return x
    def backward(self, dout, lr):
        for layer in reversed(self.classifier): dout=layer.backward(dout,lr)
        dout=dout.reshape(dout.shape[0],256,6,6)
        for layer in reversed(self.features): dout=layer.backward(dout,lr)
        return dout

    def predict(self, x):
        # 预测函数
        scores = self.forward(x, training=False)
        return np.argmax(scores, axis=1)

    def save_params(self, filename):
        # 参数保存
        params = {}
        for idx, layer in enumerate(self.features + self.classifier):
            if hasattr(layer, 'W'):
                params[f'layer{idx}_W'] = layer.W
            if hasattr(layer, 'b'):
                params[f'layer{idx}_b'] = layer.b
        np.savez(filename, **params)

    def load_params(self, filename):
        # 参数加载
        params = np.load(filename)
        layers = self.features + self.classifier
        for idx in range(len(layers)):
            if f'layer{idx}_W' in params:
                layers[idx].W = params[f'layer{idx}_W']
            if f'layer{idx}_b' in params:
                layers[idx].b = params[f'layer{idx}_b']

# =====================
# Loss
# =====================
def softmax_loss(scores: np.ndarray, y: np.ndarray):
    exp_s=np.exp(scores-np.max(scores,axis=1,keepdims=True))
    probs=exp_s/np.sum(exp_s,axis=1,keepdims=True)
    N=scores.shape[0]
    loss=-np.log(probs[np.arange(N),y]+1e-12).mean()
    ds=probs.copy(); ds[np.arange(N),y]-=1; ds/=N
    return loss, ds

# =====================
# Training & Evaluation with metrics
# =====================
if __name__=='__main__':
    train_ds=MNISTDataset(train=True)
    test_ds=MNISTDataset(train=False)
    train_loader=DataLoader(train_ds,batch_size=64,shuffle=True)
    test_loader=DataLoader(test_ds,batch_size=64,shuffle=False)

    model=ManualAlexNet()
    lr=1e-3
    epochs=5

    for ep in range(epochs):
        total_loss=0.0
        for imgs,labels in train_loader:
            scores=model.forward(imgs,training=True)
            loss,ds=softmax_loss(scores,labels)
            total_loss+=loss
            model.backward(ds,lr)
        print(f"Epoch {ep+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

        # Collect all preds and labels
        preds_list, labels_list = [], []
        for imgs, labels in test_loader:
            scores = model.forward(imgs, training=False)
            preds = np.argmax(scores, axis=1)
            preds_list.append(preds)
            labels_list.append(labels)
        y_pred = np.concatenate(preds_list)
        y_true = np.concatenate(labels_list)

        # Accuracy
        accuracy = np.mean(y_pred == y_true)*100
        # Macro Recall
        num_cls = y_true.max()+1
        recalls = []
        for cls in range(num_cls):
            tp = np.sum((y_pred==cls)&(y_true==cls))
            fn = np.sum((y_pred!=cls)&(y_true==cls))
            recalls.append(tp/(tp+fn+1e-12))
        recall_macro = np.mean(recalls)*100
        # RMSE
        rmse = np.sqrt(np.mean((y_pred-y_true)**2))

        print(f"    Accuracy: {accuracy:.2f}%  Recall (macro): {recall_macro:.2f}%  RMSE: {rmse:.4f}")

    # Save model
    params = {}
    for idx, layer in enumerate(model.features+model.classifier):
        if hasattr(layer,'W'):
            params[f'layer{idx}_W'] = layer.W
        if hasattr(layer,'b'):
            params[f'layer{idx}_b'] = layer.b
    np.savez('manual_alexnet_model_numpy_metrics_1.npz', **params)
    print('Model saved with metrics!')

