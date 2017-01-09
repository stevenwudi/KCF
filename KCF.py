"""
This is a python reimplementation of the open source tracker in
High-Speed Tracking with Kernelized Correlation Filters
Joao F. Henriques, Rui Caseiro, Pedro Martins, and Jorge Batista, tPAMI 2015

modified by Di Wu
stevenwudi@gmail.com
"""
import numpy as np
from scipy.misc import imresize


class KCFTracker:
    def __init__(self, feature_type='raw', sub_feature_type='', debug=False, gt_type='rect', load_model=False):
        """
        object_example is an image showing the object to track
        feature_type:
            "raw pixels":
            "hog":
            "CNN":
        """
        # parameters according to the paper --
        self.padding = 2.2  # extra area surrounding the target
        self.lambda_value = 1e-4  # regularization
        self.spatial_bandwidth_sigma_factor = 1 / float(16)
        self.feature_type = feature_type
        self.patch_size = []
        self.output_sigma = []
        self.cos_window = []
        self.pos = []
        self.x = []
        self.alphaf = []
        self.xf = []
        self.yf = []
        self.im_crop = []
        self.response = []
        self.target_out = []
        self.target_sz = []
        self.vert_delta = 0
        self.horiz_delta = 0
        # OBT dataset need extra definition
        self.sub_feature_type = sub_feature_type
        self.name = 'KCF' + feature_type
        if self.sub_feature_type:
            self.name += '_'+sub_feature_type
        self.fps = -1
        self.type = gt_type
        self.res = []
        self.im_sz = []
        self.debug = debug # a flag indicating to plot the intermediate figures
        self.first_patch_sz = []
        self.first_target_sz = []
        self.currentScaleFactor = 1
        self.load_model = load_model

        # following is set according to Table 2:
        if self.feature_type == 'raw':
            self.adaptation_rate = 0.075  # linear interpolation factor for adaptation
            self.feature_bandwidth_sigma = 0.2
            self.cell_size = 1
        elif self.feature_type == 'vgg':
            if self.feature_type == 'vgg':
                from keras.applications.vgg19 import VGG19
                from keras.models import Model
                self.base_model = VGG19(include_top=False, weights='imagenet')
                self.extract_model = Model(input=self.base_model.input, output=self.base_model.get_layer('block2_conv2').output)
            self.cell_size = 2
            self.feature_bandwidth_sigma = 1
            self.adaptation_rate = 0.01
            if self.sub_feature_type == 'grabcut':
                self.grabcut_mask_path = './figures/grabcut_masks/'

    def train(self, im, init_rect):
        """
        :param im: image should be of 3 dimension: M*N*C
        :param pos: the centre position of the target
        :param target_sz: target size
        """
        self.pos = [init_rect[1]+init_rect[3]/2., init_rect[0]+init_rect[2]/2.]
        self.res.append(init_rect)
        # Duh OBT is the reverse
        self.target_sz = np.asarray(init_rect[2:])
        self.target_sz = self.target_sz[::-1]
        self.first_target_sz = self.target_sz  # because we might introduce the scale changes in the detection
        # desired padded input, proportional to input target size
        self.patch_size = np.floor(self.target_sz * (1 + self.padding))
        self.first_patch_sz = np.array(self.patch_size).astype(int)  # because we might introduce the scale changes in the detection
        # desired output (gaussian shaped), bandwidth proportional to target size
        self.output_sigma = np.sqrt(np.prod(self.target_sz)) * self.spatial_bandwidth_sigma_factor

        grid_y = np.arange(np.floor(self.patch_size[0] / self.cell_size)) - np.floor(
            self.patch_size[0] / (2 * self.cell_size))
        grid_x = np.arange(np.floor(self.patch_size[1] / self.cell_size)) - np.floor(
            self.patch_size[1] / (2 * self.cell_size))
        rs, cs = np.meshgrid(grid_x, grid_y)
        self.y = np.exp(-0.5 / self.output_sigma ** 2 * (rs ** 2 + cs ** 2))
        self.yf = self.fft2(self.y)
        # store pre-computed cosine window
        self.cos_window = np.outer(np.hanning(self.yf.shape[0]), np.hanning(self.yf.shape[1]))

        # extract and pre-process subwindow
        if self.feature_type == 'raw':
            im = im /255.
            self.im_sz = im.shape
        elif self.feature_type == 'vgg':
            self.im_sz = im.shape[1:]

        self.im_crop = self.get_subwindow(im, self.pos, self.patch_size)

        self.x = self.get_features()
        import cv2

        cv2.imshow('2', self.im_crop)
        cv2.waitKey(1000)

        self.xf = self.fft2(self.x)

        k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, self.xf, self.x)
        self.alphaf = np.divide(self.yf, self.fft2(k) + self.lambda_value)
        self.response = np.real(np.fft.ifft2(np.multiply(self.alphaf, self.fft2(k))))

    def detect(self, im):
        """
        Note: we assume the target does not change in scale, hence there is no target size
        :param im: image should be of 3 dimension: M*N*C
        :return:
        """
        # extract and pre-process subwindow
        if self.feature_type == 'raw':
            im = im/255.

        # Quote from BMVC2014paper: Danelljan:
        # "In visual tracking scenarios, the scale difference between two frames is typically smaller compared to the
        # translation. Therefore, we first apply the translation filter hf given a new frame, afterwards the scale
        # filter hs is applied at the new target location.
        self.im_crop = self.get_subwindow(im, self.pos, self.patch_size)
        z = self.get_features()
        zf = self.fft2(z)

        k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, self.xf, self.x, zf, z)
        kf = self.fft2(k)
        self.response = np.real(np.fft.ifft2(np.multiply(self.alphaf, kf)))

        if self.feature_type == 'raw' or self.feature_type == 'vgg':
            # target location is at the maximum response. We must take into account the fact that, if
            # the target doesn't move, the peak will appear at the top-left corner, not at the centre
            # (this is discussed in the paper Fig. 6). The response map wrap around cyclically.
            v_centre, h_centre = np.unravel_index(self.response.argmax(), self.response.shape)
            self.vert_delta, self.horiz_delta = [v_centre - self.response.shape[0] / 2,
                                                 h_centre - self.response.shape[1] / 2]
            self.pos = self.pos + np.dot(self.cell_size, [self.vert_delta, self.horiz_delta])

        ##################################################################################
        # we need to train the tracker again here, it's almost the replicate of train
        ##################################################################################
        self.im_crop = self.get_subwindow(im, self.pos, self.patch_size)
        x_new = self.get_features()
        xf_new = self.fft2(x_new)

        k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, xf_new, x_new)
        kf = self.fft2(k)
        alphaf_new = np.divide(self.yf, kf + self.lambda_value)
        self.x = (1 - self.adaptation_rate) * self.x + self.adaptation_rate * x_new
        self.xf = (1 - self.adaptation_rate) * self.xf + self.adaptation_rate * xf_new
        self.alphaf = (1 - self.adaptation_rate) * self.alphaf + self.adaptation_rate * alphaf_new

        # we also require the bounding box to be within the image boundary
        self.res = [min(self.im_sz[1] - self.target_sz[1], max(0, self.pos[1] - self.target_sz[1] / 2.)),
                         min(self.im_sz[0] - self.target_sz[0], max(0, self.pos[0] - self.target_sz[0] / 2.)),
                         self.target_sz[1], self.target_sz[0]]

        return self.res

    def dense_gauss_kernel(self, sigma, xf, x, zf=None, z=None, feature_correlation=None):
        """
        Gaussian Kernel with dense sampling.
        Evaluates a gaussian kernel with bandwidth SIGMA for all displacements
        between input images X and Y, which must both be MxN. They must also
        be periodic (ie., pre-processed with a cosine window). The result is
        an MxN map of responses.

        If X and Y are the same, ommit the third parameter to re-use some
        values, which is faster.
        :param sigma: feature bandwidth sigma
        :param x:
        :param y: if y is None, then we calculate the auto-correlation
        :return:
        """
        N = xf.shape[0]*xf.shape[1]
        xx = np.dot(x.flatten().transpose(), x.flatten())  # squared norm of x

        if zf is None:
            # auto-correlation of x
            zf = xf
            zz = xx
        else:
            zz = np.dot(z.flatten().transpose(), z.flatten())  # squared norm of y

        xyf = np.multiply(zf, np.conj(xf))
        if self.feature_type == 'raw':
            if len(xyf.shape) == 3:
                xyf_ifft = np.fft.ifft2(np.sum(xyf, axis=2))
            elif len(xyf.shape) == 2:
                xyf_ifft = np.fft.ifft2(xyf)
            # elif len(xyf.shape) == 4:
            #     xyf_ifft = np.fft.ifft2(np.sum(xyf, axis=3))
        elif self.feature_type == 'vgg':
            if feature_correlation is None:
                xyf_ifft = np.fft.ifft2(np.sum(xyf, axis=2))
            else:
                xyf_ifft = np.fft.ifft2(np.sum(np.multiply(xyf, self.feature_correlation[None, None, :]), axis=2))

        row_shift, col_shift = np.floor(np.array(xyf_ifft.shape) / 2).astype(int)
        xy_complex = np.roll(xyf_ifft, row_shift, axis=0)
        xy_complex = np.roll(xy_complex, col_shift, axis=1)
        c = np.real(xy_complex)
        d = np.real(xx) + np.real(zz) - 2 * c
        k = np.exp(-1. / sigma**2 * np.maximum(0, d) / N)

        return k

    def get_subwindow(self, im, pos, sz):
        """
        Obtain sub-window from image, with replication-padding.
        Returns sub-window of image IM centered at POS ([y, x] coordinates),
        with size SZ ([height, width]). If any pixels are outside of the image,
        they will replicate the values at the borders.

        The subwindow is also normalized to range -0.5 .. 0.5, and the given
        cosine window COS_WINDOW is applied
        (though this part could be omitted to make the function more general).
        """

        if np.isscalar(sz):  # square sub-window
            sz = [sz, sz]

        ys = np.floor(pos[0]) + np.arange(sz[0], dtype=int) - np.floor(sz[0] / 2)
        xs = np.floor(pos[1]) + np.arange(sz[1], dtype=int) - np.floor(sz[1] / 2)

        ys = ys.astype(int)
        xs = xs.astype(int)

        # check for out-of-bounds coordinates and set them to the values at the borders
        ys[ys < 0] = 0
        ys[ys >= self.im_sz[0]] = self.im_sz[0] - 1

        xs[xs < 0] = 0
        xs[xs >= self.im_sz[1]] = self.im_sz[1] - 1

        # extract image

        if self.feature_type == 'raw':
            out = im[np.ix_(ys, xs)]
            # introduce scaling, here, we need them to be the same size
            if np.all(self.first_patch_sz == out.shape[:2]):
                return out
            else:
                out = imresize(out, self.first_patch_sz)
                return out / 255.
        elif self.feature_type == 'vgg':
            c = np.array(range(3))
            out = im[np.ix_(c, ys, xs)]
            return out

    def fft2(self, x):
        """
        FFT transform of the first 2 dimension
        :param x: M*N*C the first two dimensions are used for Fast Fourier Transform
        :return:  M*N*C the FFT2 of the first two dimension
        """
        return np.fft.fft2(x, axes=(0, 1))

    def get_features(self):
        """
        :param im: input image
        :return:
        """
        if self.feature_type == 'raw':
            #using only grayscale:
            if len(self.im_crop.shape) == 3:
                if self.sub_feature_type == 'gray':
                    img_gray = np.mean(self.im_crop, axis=2)
                    img_gray = img_gray - img_gray.mean()
                    features = np.multiply(img_gray, self.cos_window)
                else:
                    img_colour = self.im_crop - self.im_crop.mean()
                    features = np.multiply(img_colour, self.cos_window[:, :, None])
                return features

        elif self.feature_type == 'vgg' or self.feature_type == 'resnet50':
            from keras.applications.vgg19 import preprocess_input
            x = np.expand_dims(self.im_crop.copy(), axis=0)
            x = preprocess_input(x)
            features = self.extract_model.predict(x)
            features = np.squeeze(features)
            features = (features.transpose(1, 2, 0) - features.min()) / (features.max() - features.min())
            features = np.multiply(features, self.cos_window[:, :, None])
            return features
        else:
            assert 'Non implemented!'


