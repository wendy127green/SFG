import mxnet as mx
import numpy as np
from mxnet import nd, gluon, autograd
import pdb
from .MaskFlownet import *
from .config import Reader
from .layer import Reconstruction2D, Reconstruction2DSmooth
import copy
import skimage.io
import os
import pandas as pd
import random
import string
import matplotlib.pyplot as plt
import cv2

def build_network(name):
	return eval(name)

def get_coords(img):
	shape = img.shape
	range_x = nd.arange(shape[2], ctx = img.context).reshape(shape = (1, 1, -1, 1)).tile(reps = (shape[0], 1, 1, shape[3]))
	range_y = nd.arange(shape[3], ctx = img.context).reshape(shape = (1, 1, 1, -1)).tile(reps = (shape[0], 1, shape[2], 1))
	return nd.concat(range_x, range_y, dim = 1)


class PipelineFlownet:
	_lr = None

	def __init__(self, ctx, config):
		self.ctx = ctx
		self.network = build_network(getattr(config.network, 'class').get('MaskFlownet'))(config=config)
		self.network.hybridize()
		self.network.collect_params().initialize(init=mx.initializer.MSRAPrelu(slope=0.1), ctx=self.ctx)
		self.trainer = gluon.Trainer(self.network.collect_params(), 'adam', {'learning_rate': 1e-4})
		self.strides = self.network.strides or [64, 32, 16, 8, 4]

		self.scale = self.strides[-1]
		self.upsampler = Upsample(self.scale)
		self.upsampler_mask = Upsample(self.scale)

		self.epeloss = EpeLoss()
		self.epeloss.hybridize()
		self.epeloss_with_mask = EpeLossWithMask()
		self.epeloss_with_mask.hybridize()

		## start:gl add loss function
		self.raw_weight = 1
		self.raw_loss_op = CorrelationLoss()
		self.raw_loss_op.hybridize()

		self.regularization_op = RegularizatonLoss()
		self.regularization_op.hybridize()
		self.reg_weight = config.optimizer.regularization.get(0)
		self.boundary_loss_op = BoundaryLoss()
		self.boundary_loss_op.hybridize()
		self.boundary_weight = config.optimizer.boundary.get(0)
		## end: gl add loss function

		multiscale_weights = config.network.mw.get([.005, .01, .02, .08, .32])
		if len(multiscale_weights) != 5:
			multiscale_weights = [.005, .01, .02, .08, .32]
		self.multiscale_epe = MultiscaleEpe(
			scales = self.strides, weights = multiscale_weights, match = 'upsampling',
			eps = 1e-8, q = config.optimizer.q.get(None))
		self.multiscale_epe.hybridize()

		self.reconstruction = Reconstruction2DSmooth(3)
		self.reconstruction.hybridize()

		self.lr_schedule = config.optimizer.learning_rate.value

	def save(self, prefix):
		self.network.save_parameters(prefix + '.params')
		self.trainer.save_states(prefix + '.states')

	def load(self, checkpoint):
		self.network.load_parameters(checkpoint, ctx=self.ctx)

	def load_head(self, checkpoint):
		self.network.load_head(checkpoint, ctx=self.ctx)

	def fix_head(self):
		self.network.fix_head()

	def set_learning_rate(self, steps):
		i = 0
		while i < len(self.lr_schedule) and steps > self.lr_schedule[i][0]:
			i += 1
		try:
			lr = self.lr_schedule[i][1]
		except IndexError:
			return False
		self.trainer.set_learning_rate(lr)
		self._lr = lr
		return True	

	@property
	def lr(self):
		return self._lr

	def loss(self, pred, occ_masks, labels, masks):
		loss = self.multiscale_epe(labels, masks, *pred)
		return loss
	
	def centralize(self, img1, img2):
		rgb_mean = nd.concat(img1, img2, dim = 2).mean(axis = (2, 3)).reshape((-2, 1, 1))
		return img1 - rgb_mean, img2 - rgb_mean, rgb_mean

	def train_batch(self, raw_weight, img1, img2, sift1, sift2, color_aug, aug):
		losses = []
		reg_losses = []
		raw_losses = []
		dist_losses = []
		batch_size = img1.shape[0]
		img1, img2, sift1, sift2 = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2, sift1, sift2))
		hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
		with autograd.record():
			for img1s, img2s,sift1s, sift2s in zip(img1, img2,sift1, sift2):
				img1s, img2s = img1s / 255.0, img2s / 255.0
				img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
				pred, occ_masks, warpeds = self.network(img1s, img2s)

				shape = img1s.shape
				flow = self.upsampler(pred[-1])
				if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
					flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))

				warp = self.reconstruction(sift2s, flow)
				flows = []
				flows.append(flow)
				
				# raw loss calculation
				raw_loss = self.raw_loss_op(sift1s, warp)
				reg_loss = self.regularization_op(flow) + self.boundary_loss_op(flow) * self.boundary_weight

				loss = raw_loss * raw_weight + reg_loss * self.reg_weight
				losses.append(loss)
				reg_losses.append(reg_loss)
				raw_losses.append(raw_loss)
				# dist_losses.append(dist_loss)


		for loss in losses:
			loss.backward()
		self.trainer.step(batch_size)
		return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses])), "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses]))}

	def landmark_dist(self, lmk1, lmk2, flows):
		if np.shape(lmk2)[0] > 0:
			flow_len = len(flows)
			shape = nd.array(flows[0].shape[2: 4], ctx=flows[0].context)

			lmk_mask = (1 - nd.prod(lmk1 == lmk1[0][199][0] * lmk1[0][199][1], axis=-1)) * (
						1 - nd.prod(lmk2 == lmk2[0][199][0] * lmk2[0][199][1], axis=-1)) > 0.5
			for flow in flows:
				batch_lmk = lmk1 / (nd.reshape(shape, (1, 1, 2)) - 1) * 2 - 1
				batch_lmk = batch_lmk.transpose((0, 2, 1)).expand_dims(axis=3)
				warped_lmk = lmk1 + nd.BilinearSampler(flow, batch_lmk.flip(axis=1)).squeeze(axis=-1).transpose(
					(0, 2, 1))
				lmk1 = warped_lmk
			lmk_dist = nd.mean(nd.sqrt(nd.sum(nd.square(warped_lmk - lmk2), axis=-1) * lmk_mask + 1e-5), axis=-1)
			lmk_dist = lmk_dist/(np.sum(lmk_mask, axis=1)+1e-5)*200
			lmk_dist = lmk_dist*(np.sum(lmk_mask, axis=1)!=0)
			return lmk_dist / (shape[0]*1.414), warped_lmk, lmk2
		else:
			return 0, [], []

	def landmark_dist_v(self, lmk1, lmk2, flows):
		lmknew = np.zeros((np.shape(lmk1)[0], np.shape(lmk1)[1], np.shape(lmk1)[2]))
		lmk2new = np.zeros((np.shape(lmk2)[0], np.shape(lmk2)[1], np.shape(lmk2)[2]))
		if np.shape(lmk2)[0] > 0:
			flow_len = len(flows)
			shape = nd.array(flows[0].shape[2: 4], ctx=flows[0].context)
			lmk_dist_all = nd.ones((np.shape(lmk1)[0],), ctx=flows[0].context)
			lmk_dist_all2 = []
			for k in range(0, np.shape(lmk1)[0]):

				lmk1n = lmk1[k]
				lmk1n = lmk1n.reshape(1, np.shape(lmk1n)[0], np.shape(lmk1n)[1])
				lmk2n = lmk2[k]
				lmk2n = lmk2n.reshape(1, np.shape(lmk2n)[0], np.shape(lmk2n)[1])
				lmk_mask = (1 - (lmk1n[0, :, 0] * lmk1n[0, :, 1] == lmk1n[0][199][0] * lmk1n[0][199][1])) * (1 - (lmk2n[0, :, 0] * lmk2n[0, :, 1] == lmk2n[0][199][0] * lmk2n[0][199][1])) > 0.5
				mask_num = np.sum(lmk_mask)
				mask_num = int(mask_num.asnumpy())
				lmk1n = lmk1n[:, :mask_num, :]
				lmk2n = lmk2n[:, :mask_num, :]
				for flow in flows:
					flow = flow[k]
					flow = flow.reshape(1, np.shape(flow)[0], np.shape(flow)[1], np.shape(flow)[2])
					batch_lmk = lmk1n / (nd.reshape(shape, (1, 1, 2)) - 1) * 2 - 1
					batch_lmk = batch_lmk.transpose((0, 2, 1)).expand_dims(axis=3)
					warped_lmk = lmk1n + nd.BilinearSampler(flow, batch_lmk.flip(axis=1)).squeeze(axis=-1).transpose((0, 2, 1))
				# start: median rTRE
				lmk_dist = nd.sqrt(nd.sum(nd.square(warped_lmk - lmk2n), axis=-1) + 1e-5)
				lmk_dist_numpy = []
				for m in range(0, np.shape(lmk_dist)[1]):
					lmk_dist_numpy.append(lmk_dist[0, m].asnumpy())
				if np.shape(lmk_dist)[1] % 2 == 0:
					med = lmk_dist_numpy.index(np.median(lmk_dist_numpy[1:]))
				else:
					med = lmk_dist_numpy.index(np.median(lmk_dist_numpy))
				lmk_dist_median = lmk_dist[0,med]
				lmk_dist_all[k]=lmk_dist_median.asnumpy()
				lmk2new[k, :mask_num, :] = lmk2n.asnumpy()
				lmknew[k, :mask_num, :] = warped_lmk.asnumpy()

			return lmk_dist_all / (shape[0]*1.414), lmknew, lmk2new
		else:
			return 0, [], []

	def validate(self, data, batch_size):
		results = []
		raws = []
		dist_mean = []
		size = len(data)
		bs = batch_size * len(self.ctx)
		bs = len(self.ctx)
		output_cnt = 0
		for j in range(0, size, bs):
			batch_data = data[j: j + bs]
			ctx = self.ctx[: min(len(batch_data), len(self.ctx))]
			nd_data = [gluon.utils.split_and_load([record[i] for record in batch_data], ctx, even_split=False) for i in range(len(batch_data[0]))]
			for img1, img2, lmk1, lmk2 in zip(*nd_data):
				img1, img2 = img1 / 255.0, img2 / 255.0
				img1, img2, rgb_mean = self.centralize(img1, img2)
				# _, flows, warpeds = self.predict_batch_mx(img1, img2)
				pred, occ_masks, warpeds = self.network(img1, img2)
				shape = img1.shape
				flow = self.upsampler(pred[-1])
				if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
					flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
				warp = self.reconstruction(img2, flow)
				flows = []
				flows.append(flow)

				raw = self.raw_loss_op(img1, warp)
				raws.append(raw.mean())
				dist_loss_mean, warped_lmk, lmk2new = self.landmark_dist(lmk1, lmk2, flows)
				# print(dist_loss_mean)
				dist_mean.append(dist_loss_mean)
				batchnum = 0

		rawmean = []
		for raw in raws:
			raw = raw.asnumpy()
			rawmean.append(raw)
		distmean = []
		for distm in dist_mean:
			distm = distm.asnumpy()
			distmean.append(distm)
		results_median = []

		return np.mean(rawmean), np.mean(distmean), np.median(distmean), np.mean(distmean)#np.median(results_median)


	def predict(self, img1, img2, batch_size, resize = None):
		''' predict the whole dataset
		'''
		size = len(img1)
		bs = batch_size
		for j in range(0, size, bs):
			batch_img1 = img1[j: j + bs]
			batch_img2 = img2[j: j + bs]

			batch_img1 = np.transpose(np.stack(batch_img1, axis=0), (0, 3, 1, 2))
			batch_img2 = np.transpose(np.stack(batch_img2, axis=0), (0, 3, 1, 2))

			batch_flow = []
			batch_occ_mask = []
			batch_warped = []

			ctx = self.ctx[ : min(len(batch_img1), len(self.ctx))]
			nd_img1, nd_img2 = map(lambda x : gluon.utils.split_and_load(x, ctx, even_split = False), (batch_img1, batch_img2))
			for img1s, img2s in zip(nd_img1, nd_img2):
				img1s, img2s = img1s / 255.0, img2s / 255.0
				flow, occ_mask, warped, _ = self.do_batch(img1s, img2s, resize = resize)
				batch_flow.append(flow)
				batch_occ_mask.append(occ_mask)
				batch_warped.append(warped)
			flow = np.concatenate([x.asnumpy() for x in batch_flow])
			occ_mask = np.concatenate([x.asnumpy() for x in batch_occ_mask])
			warped = np.concatenate([x.asnumpy() for x in batch_warped])
			
			flow = np.transpose(flow, (0, 2, 3, 1))
			flow = np.flip(flow, axis = -1)
			occ_mask = np.transpose(occ_mask, (0, 2, 3, 1))
			warped = np.transpose(warped, (0, 2, 3, 1))
			for k in range(len(flow)):
				yield flow[k], occ_mask[k], warped[k]
