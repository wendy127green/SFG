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

	def train_batch(self, dist_weight, img1, img2, lmk1s, lmk2s, sift1, sift2, color_aug, aug):
		losses = []
		reg_losses = []
		raw_losses = []
		dist_losses = []
		batch_size = img1.shape[0]
		img1, img2, lmk1s, lmk2s, sift1, sift2 = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2, lmk1s, lmk2s, sift1, sift2))
		hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
		with autograd.record():
			for img1s, img2s, lmk1, lmk2, sift1s, sift2s in zip(img1, img2, lmk1s, lmk2s, sift1, sift2):
				img1s, img2s = img1s / 255.0, img2s / 255.0
				#img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
				# img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
				img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
				pred, occ_masks, warpeds = self.network(img1s, img2s) # this warpeds is not mean the warped image
				## start: gl add warped image obtain and raw_loss calculation
				shape = img1s.shape
				flow = self.upsampler(pred[-1])
				if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
					flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
				# np.shape(img1s)=(8,3,256,256), np.shape(flow)=(8,2,256,256)
				warp = self.reconstruction(sift2s, flow)
				# warp = self.reconstruction(img2s, flow)
				flows = []
				flows.append(flow)
				dist_loss, warped_lmk, lmk2new = self.landmark_dist(lmk1, lmk2, flows)
				# raw loss calculation
				raw_loss = self.raw_loss_op(sift1s, warp)
				reg_loss = self.regularization_op(flow) + self.boundary_loss_op(flow) * self.boundary_weight # gl, in this program, although cascaded, len(flow)=1
				self.reg_weight = 1 #0.2
				# dist_weight = 200#200#200#0#50 # 10#1 #50 #100 #200
				loss = raw_loss * self.raw_weight + reg_loss * self.reg_weight + dist_loss*dist_weight # gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
				#loss = dist_loss * dist_weight  # gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
				losses.append(loss)
				reg_losses.append(reg_loss)
				raw_losses.append(raw_loss)
				dist_losses.append(dist_loss)
				## end: gl add warped image obtain and raw_loss calculation
				# # start: gl train result visuatlization
				# batchnum = 0
				# for img1s, img2s, warped, lmk1s, lmk2s, warped_lmk, lmk2new in zip(img1s.asnumpy(), img2s.asnumpy(), warp.asnumpy(), lmk1.asnumpy(), lmk2.asnumpy(), warped_lmk, lmk2new):
				# 	output_prefix = os.path.join(r".\visualization\train", str(hsh)+str(batchnum))
				# 	skimage.io.imsave(output_prefix + "_1moving.png", np.clip(img2s.transpose((1, 2, 0)).squeeze() * 255, 0, 255).astype(np.uint8))
				# 	skimage.io.imsave(output_prefix + "_2warped.png", np.clip(warped.transpose((1, 2, 0)).squeeze() * 255, 0, 255).astype(np.uint8))
				# 	skimage.io.imsave(output_prefix + "_3fixed.png",  np.clip(img1s.transpose((1, 2, 0)).squeeze() * 255, 0, 255).astype(np.uint8))
				# 	lmk1 = lmk1s.T
				# 	warped_lmk = warped_lmk.T
				# 	lmk2new = lmk2new.T
				# 	lmk1name = output_prefix + "_1moving.csv"
				# 	csv1 = pd.DataFrame({'x': lmk2new[0], 'y': lmk2new[1]})
				# 	csv1.to_csv(lmk1name)
				# 	lmk2name = output_prefix + "_2warped.csv"
				# 	csv2 = pd.DataFrame({'x': warped_lmk[0], 'y': warped_lmk[1]})
				# 	csv2.to_csv(lmk2name)
				# 	lmk3name = output_prefix + "_3fixed.csv"
				# 	csv3 = pd.DataFrame({'x': lmk1[0], 'y': lmk1[1]})
				# 	csv3.to_csv(lmk3name)
				#
				# 	## start: present key points with image
				# 	output_prefix_present = os.path.join(r".\visualization\train_present", str(hsh)+str(batchnum))
				# 	movingimg = np.clip(img2s.transpose((1, 2, 0)).squeeze() * 255, 0, 255).astype(np.uint8)
				# 	warpedimg = np.clip(warped.transpose((1, 2, 0)).squeeze() * 255, 0, 255).astype(np.uint8)
				# 	fixedimg = np.clip(img1s.transpose((1, 2, 0)).squeeze() * 255, 0, 255).astype(np.uint8)
				# 	lmk1 = lmk1.T
				# 	warped_lmk = warped_lmk.T
				# 	lmk2new = lmk2new.T
				# 	plt.imshow(movingimg)
				# 	locsl = np.shape(lmk2new)[0]
				# 	for i in range(locsl):
				# 		plt.plot(int(lmk2new[i][1]), int(lmk2new[i][0]), 'r*')
				# 	plt.savefig(output_prefix_present + "_1moving.jpg")
				# 	plt.close('all')
				#
				# 	plt.imshow(warpedimg)
				# 	locsl = np.shape(warped_lmk)[0]
				# 	for i in range(locsl):
				# 		plt.plot(int(warped_lmk[i][1]), int(warped_lmk[i][0]), 'r*')
				# 	for i in range(locsl):
				# 		plt.plot(int(lmk2new[i][1]), int(lmk2new[i][0]), 'b*')
				# 	plt.savefig(output_prefix_present + "_2warped.jpg")
				# 	plt.close('all')
				#
				# 	plt.imshow(fixedimg)
				# 	locsl = np.shape(lmk1)[0]
				# 	for i in range(locsl):
				# 		plt.plot(int(lmk1[i][1]), int(lmk1[i][0]), 'r*')
				# 	plt.savefig(output_prefix_present + "_3fixed.jpg")
				# 	plt.close('all')
				# 	## end: present key points with image
				# 	batchnum = batchnum+1
				# ## end: gl train result visualization

		for loss in losses:
			loss.backward()
		self.trainer.step(batch_size)
		# print('loss:', np.mean(np.concatenate([loss.asnumpy() for loss in losses])))
		return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses])), "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses])), "dist loss": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses]))}

	def landmark_dist(self, lmk1, lmk2, flows):
		if np.shape(lmk2)[0] > 0:
			flow_len = len(flows)
			shape = nd.array(flows[0].shape[2: 4], ctx=flows[0].context)
			# old lmk_mask is when lmk1 and lmk2 are all not 0, 不知为何，会出现非零的补充。所以改为和200项最后一项相同的就不要
			lmk_mask = (1 - nd.prod(lmk1 == lmk1[0][199][0] * lmk1[0][199][1], axis=-1)) * (
						1 - nd.prod(lmk2 == lmk2[0][199][0] * lmk2[0][199][1], axis=-1)) > 0.5
			for flow in flows:
				batch_lmk = lmk1 / (nd.reshape(shape, (1, 1, 2)) - 1) * 2 - 1
				batch_lmk = batch_lmk.transpose((0, 2, 1)).expand_dims(axis=3)
				warped_lmk = lmk1 + nd.BilinearSampler(flow, batch_lmk.flip(axis=1)).squeeze(axis=-1).transpose(
					(0, 2, 1))
				lmk1 = warped_lmk
			lmk_dist = nd.mean(nd.sqrt(nd.sum(nd.square(warped_lmk - lmk2), axis=-1) * lmk_mask + 1e-5), axis=-1)
			lmk_dist = lmk_dist/(np.sum(lmk_mask, axis=1)+1e-5)*200 # 消除当kp数目为0的时候的影响
			lmk_dist = lmk_dist*(np.sum(lmk_mask, axis=1)!=0) # 消除当kp数目为0的时候的影响
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
				# old lmk_mask is when lmk1 and lmk2 are all not 0, 不知为何，会出现非零的补充。所以改为和200项最后一项相同的就不要
				lmk1n = lmk1[k]
				lmk1n = lmk1n.reshape(1, np.shape(lmk1n)[0], np.shape(lmk1n)[1])
				lmk2n = lmk2[k]
				lmk2n = lmk2n.reshape(1, np.shape(lmk2n)[0], np.shape(lmk2n)[1])
				lmk_mask = (1 - (lmk1n[0, :, 0] * lmk1n[0, :, 1] == lmk1n[0][199][0] * lmk1n[0][199][1])) * (1 - (lmk2n[0, :, 0] * lmk2n[0, :, 1] == lmk2n[0][199][0] * lmk2n[0][199][1])) > 0.5
				mask_num = np.sum(lmk_mask)  # gl resuse lmk_mask
				mask_num = int(mask_num.asnumpy())  # gl resuse lmk_mask
				lmk1n = lmk1n[:, :mask_num, :]  # gl resuse lmk_mask
				lmk2n = lmk2n[:, :mask_num, :]  # gl resuse lmk_mask
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
				# end:median rTRE
				# # start: mean rTRE
				# lmk_dist = nd.mean(nd.sqrt(nd.sum(nd.square(warped_lmk - lmk2n), axis=-1) + 1e-5), axis=-1)
				# lmk_dist_all[k] = lmk_dist.asnumpy()
				# # end: mean rTRE
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
				# try:
				# 	dist_loss, warped_lmk, lmk2new = self.landmark_dist_v(lmk1, lmk2, flows)
				# except:
				# 	img1 = img1[0]
				# 	img1 = np.clip(img1.transpose((1, 2, 0)).squeeze() * 255, 0, 255).astype(np.uint8)
				# 	img2 = img2[0]
				# 	img2 = np.clip(img2.transpose((1, 2, 0)).squeeze() * 255, 0, 255).astype(np.uint8)
				# 	print('img1', img1)
				# 	print('img2', img2)
				# 	print('flow', flow)
				# 	img1 = img1.asnumpy()
				# 	img2 = img2.asnumpy()
				# 	plt.figure()
				# 	plt.subplot(121)
				# 	plt.imshow(img1)
				# 	plt.subplot(122)
				# 	plt.imshow(img2)
				# 	plt.show()
				# 	pdb.set_trace()
				# 	cv2.imshow('img1', img1)
				# 	cv2.waitKey(0)
				# 	pdb.set_trace()
				# results.append(dist_loss)
				## warped = self.reconstruction(img2, flow)
				raw = self.raw_loss_op(img1, warp)
				raws.append(raw.mean())
				dist_loss_mean, warped_lmk, lmk2new = self.landmark_dist(lmk1, lmk2, flows)
				# print(dist_loss_mean)
				dist_mean.append(dist_loss_mean)
				batchnum = 0
				# ## start: gl, visualization
				# for img1, img2, warped, lmk1, lmk2, warped_lmk, lmk2new in zip(img1.asnumpy(), img2.asnumpy(), warp.asnumpy(), lmk1.asnumpy(), lmk2.asnumpy(), warped_lmk, lmk2new):
				# 	output_prefix = os.path.join(r"./visualization/test", str(output_cnt))
				# 	skimage.io.imsave(output_prefix + "_1moving.png", np.clip(img2.transpose((1, 2, 0)).squeeze() * 255, 0, 255).astype(np.uint8))
				# 	skimage.io.imsave(output_prefix + "_2warped.png", np.clip(warped.transpose((1, 2, 0)).squeeze() * 255, 0, 255).astype(np.uint8))
				# 	skimage.io.imsave(output_prefix + "_3fixed.png", np.clip(img1.transpose((1, 2, 0)).squeeze() * 255, 0, 255).astype(np.uint8))
				# 	lmk1 = lmk1.T
				# 	lmk2 = lmk2.T
				# 	warped_lmk = warped_lmk.T
				# 	lmk2new = lmk2new.T
				# 	lmk_mask = (1 - (lmk1[0, :] * lmk1[1, :] == lmk1[0][199] * lmk1[1][199])) > 0.5
				# 	mask_num = np.sum(lmk_mask)
				# 	lmk2 = lmk2[:,:mask_num]
				# 	lmk1name = output_prefix + "_1moving.csv"
				# 	csv1 = pd.DataFrame({'x': lmk2[0], 'y': lmk2[1]})
				# 	csv1.to_csv(lmk1name)
				# 	lmk2name = output_prefix + "_2warped.csv"
				# 	warped_lmk = warped_lmk.asnumpy()
				# 	warped_lmk = warped_lmk[:,:mask_num]
				# 	csv2 = pd.DataFrame({'x': warped_lmk[0], 'y': warped_lmk[1]})
				# 	csv2.to_csv(lmk2name)
				# 	lmk3name = output_prefix + "_3fixed.csv"
				# 	lmk1 = lmk1[:, :mask_num]
				# 	csv3 = pd.DataFrame({'x': lmk1[0], 'y': lmk1[1]})
				# 	csv3.to_csv(lmk3name)
				#
				# 	lmk2new = lmk2new.asnumpy()
				# 	lmk2new = lmk2new[:,:mask_num]
				# 	## start: present key points with image
				# 	output_prefix_present = os.path.join(r"./visualization/test_present", str(output_cnt))
				# 	movingimg = np.clip(img2.transpose((1, 2, 0)).squeeze() * 255, 0, 255).astype(np.uint8)
				# 	warpedimg = np.clip(warped.transpose((1, 2, 0)).squeeze() * 255, 0, 255).astype(np.uint8)
				# 	fixedimg = np.clip(img1.transpose((1, 2, 0)).squeeze() * 255, 0, 255).astype(np.uint8)
				# 	lmk1 = lmk1.T
				# 	warped_lmk = warped_lmk.T
				# 	lmk2new = lmk2new.T
				# 	plt.imshow(movingimg)
				# 	locsl = np.shape(lmk2new)[0]
				# 	for i in range(locsl):
				# 		plt.plot(int(lmk2new[i][1]), int(lmk2new[i][0]), 'r*')
				# 	plt.savefig(output_prefix_present + "_1moving.jpg")
				# 	plt.close('all')
				#
				# 	plt.imshow(warpedimg)
				# 	locsl = np.shape(warped_lmk)[0]
				# 	for i in range(locsl):
				# 		plt.plot(int(warped_lmk[i][1]), int(warped_lmk[i][0]), 'r*')
				# 	for i in range(locsl):
				# 		plt.plot(int(lmk2new[i][1]), int(lmk2new[i][0]), 'b*')
				# 	plt.savefig(output_prefix_present + "_2warped.jpg")
				# 	plt.close('all')
				#
				# 	plt.imshow(fixedimg)
				# 	locsl = np.shape(lmk1)[0]
				# 	for i in range(locsl):
				# 		plt.plot(int(lmk1[i][1]), int(lmk1[i][0]), 'r*')
				# 	plt.savefig(output_prefix_present + "_3fixed.jpg")
				# 	plt.close('all')
				# 	## end: present key points with image
				# 	batchnum = batchnum+1
				# 	output_cnt += 1
				# ## end: gl, visualization
		rawmean = []
		for raw in raws:
			raw = raw.asnumpy()
			rawmean.append(raw)
		distmean = []
		for distm in dist_mean:
			distm = distm.asnumpy()
			distmean.append(distm)
		results_median = []
		# for result in results:
		# 	result = result.asnumpy()
		# 	for lmk_dist in result:
		# 		if np.shape(lmk_dist[lmk_dist >= 0])[0] > 0:
		# 			results_median.append(np.median(lmk_dist[lmk_dist >= 0]))
		#print('validation results_median', results_median)
		return np.mean(rawmean), np.mean(distmean), np.median(distmean), np.mean(distmean)#np.median(results_median)

	# def validate(self, img1, img2, label, mask=None, batch_size=1, resize=None, return_type='epe'):
	# 	''' validate the whole dataset
	# 	'''
	# 	np_epes = []
	# 	size = len(img1)
	# 	bs = batch_size
	# 	if mask is None:
	# 		mask = [np.full(shape=(1, 1, 1), fill_value=255, dtype=np.uint8)] * size
	# 	for j in range(0, size, bs):
	# 		batch_img1 = img1[j: j + bs]
	# 		batch_img2 = img2[j: j + bs]
	# 		batch_label = label[j: j + bs]
	# 		batch_mask = mask[j: j + bs]
	#
	# 		batch_img1 = np.transpose(np.stack(batch_img1, axis=0), (0, 3, 1, 2))
	# 		batch_img2 = np.transpose(np.stack(batch_img2, axis=0), (0, 3, 1, 2))
	# 		batch_label = np.transpose(np.stack(batch_label, axis=0), (0, 3, 1, 2))
	# 		batch_mask = np.transpose(np.stack(batch_mask, axis=0), (0, 3, 1, 2))
	#
	# 		def Norm(x):
	# 			return nd.sqrt(nd.sum(nd.square(x), axis=1, keepdims=True))
	#
	# 		batch_epe = []
	# 		ctx = self.ctx[: min(len(batch_img1), len(self.ctx))]
	# 		nd_img1, nd_img2, nd_label, nd_mask = map(lambda x: gluon.utils.split_and_load(x, ctx, even_split=False),
	# 												  (batch_img1, batch_img2, batch_label, batch_mask))
	# 		for img1s, img2s, labels, masks in zip(nd_img1, nd_img2, nd_label, nd_mask):
	# 			img1s, img2s, labels, masks = img1s / 255.0, img2s / 255.0, labels.astype("float32",
	# 																					  copy=False), masks / 255.0
	# 			labels = labels.flip(axis=1)
	# 			flows, _, _, epe = self.do_batch(img1s, img2s, labels, masks, resize=resize)
	#
	# 			# calculate the metric for kitti dataset evaluation
	# 			if return_type is not 'epe':
	# 				eps = 1e-8
	# 				epe = ((Norm(flows - labels) > 3) * (
	# 							(Norm(flows - labels) / (Norm(labels) + eps)) > 0.05) * masks).sum(axis=0,
	# 																							   exclude=True) / masks.sum(
	# 					axis=0, exclude=True)
	#
	# 			batch_epe.append(epe)
	# 		np_epes.append(np.concatenate([epe.asnumpy() for epe in batch_epe]))
	#
	# 	return np.mean(np.concatenate(np_epes, axis=0), axis=0)

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
