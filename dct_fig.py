import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
from PIL import Image


class DCT:
    def __init__(self, N):
        self.N = N  # データ数．
        # 2次元離散コサイン変換の基底ベクトルをあらかじめ作っておく
        self.phi_1d = np.array([self.phi(i) for i in range(self.N)])

        self.phi_2d = np.zeros((N, N, N, N))
        for i in range(N):
            for j in range(N):
                phi_i, phi_j = np.meshgrid(self.phi_1d[i], self.phi_1d[j])
                self.phi_2d[i, j] = phi_i * phi_j

    # 2次元離散コサイン変換
    def dct2(self, data):
        return np.sum(self.phi_2d.reshape(N * N, N * N) * data.reshape(N * N), axis=1).reshape(N, N)

    # 2次元離散コサイン逆変換
    def idct2(self, c, number):
        c[number:, number:] = 0
        return np.sum((c.reshape(N, N, 1) * self.phi_2d.reshape(N, N, N * N)).reshape(N * N, N * N), axis=0)\
            .reshape(N, N)

    # 基底関数
    def phi(self, k):
        if k == 0:
            return np.ones(self.N) / np.sqrt(self.N)
        else:
            return np.sqrt(2.0 / self.N) * np.cos((k * np.pi * (np.arange(self.N) * 2 + 1) / (2 * self.N)))

    def image_loading(self, image):
        # 画像情報の確認
        print(image)
        print(image.shape)

        c = dct.dct2(image)  # 2次元離散コサイン変換
        # print(c)
        y = dct.idct2(c, number)  # 2次元離散コサイン逆変換
        # print(y)

        return c, y


def show_image(img, c, y):
    # 元の画像
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="Greys")
    plt.title("original")
    plt.xticks([])
    # コサイン変換
    # plt.subplot(1, 3, 2)
    # plt.imshow(c, cmap="Greys")
    # plt.title("cosine transform")
    # plt.xticks([])
    # 復元した画像
    plt.subplot(1, 3, 3)
    plt.imshow(y, cmap="Greys")
    plt.title("restored")
    plt.xticks([])
    plt.show()


if __name__ == "__main__":
    # 画像のピクセル数を代入
    N = 50
    # 0に丸め込む範囲の指定
    number = 15
    # クラスの読み込み
    dct = DCT(N)

    # サンプル画像を作る
    test_image = randint(0, 2, (50, 50))

    # 画像読み込み
    image = np.array(Image.open('./image/image3.jpg'))
    # グレースケール変換
    im_gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    c, y = dct.image_loading(im_gray)

    # 画像表示
    show_image(im_gray, c, y)
