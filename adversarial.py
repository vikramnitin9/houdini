from helper import *

class Houdini(torch.autograd.Function):

	@staticmethod
	def forward(ctx, input, labels, task_loss, ignore_index=255):

		Y_pred			= input
		Y 				= labels.unsqueeze(1)

		max_preds, _    = Y_pred.max(axis=1)
		mask            = (Y != ignore_index)
		Y               = torch.where(mask, Y, torch.zeros_like(Y).to(Y.device))
		true_preds      = torch.gather(Y_pred, 1, Y).squeeze(1)

		normal_dist     = torch.distributions.Normal(0.0, 1.0)
		probs           = 1.0 - normal_dist.cdf(true_preds - max_preds)
		loss            = torch.sum(probs * task_loss * mask.squeeze(1)) / torch.sum(mask.float())

		ctx.save_for_backward(Y_pred, Y, mask, task_loss)
		return loss

	@staticmethod
	def backward(ctx, grad_output):

		Y_pred, Y, mask, task_loss = ctx.saved_tensors

		C = 1./math.sqrt(2 * math.pi)

		max_preds, max_inds = Y_pred.max(axis=1)
		Y               	= torch.where(mask, Y, torch.zeros_like(Y).to(Y.device))
		true_preds      	= torch.gather(Y_pred, 1, Y).squeeze(1)

		temp 				= C * torch.exp(-1.0 * (torch.abs(true_preds - max_preds) ** 2) / 2.0) * task_loss * mask.squeeze(1)
		grad_input 			= torch.zeros_like(Y_pred).to(Y_pred.device)

		grad_input.scatter_(1, max_inds.unsqueeze(1), temp.unsqueeze(1))
		grad_input.scatter_(1, Y, -1.0 * temp.unsqueeze(1))

		grad_input 			/= torch.sum(mask.float())

		return (grad_input, None, None, None)


class Adversarial(object):

	def __init__(self, loss_fn):
		self.loss_fn = loss_fn

	def pgd_perturb(self, X, Y, model, k=40, a=0.01, epsilon=0.3, target='untargeted', rand_start=True):

		X_pgd = X.clone()

		if rand_start:
			X_pgd = X_pgd + (2 * epsilon * torch.rand(*X_pgd.shape).to(X_pgd.device) - epsilon)
			X_pgd = torch.clamp(X_pgd, 0, 1) # ensure valid pixel range

		houdini = Houdini.apply

		for i in range(k):

			X_pgd.requires_grad = True
			Y_pred = model.forward(X_pgd)

			if target == 'untargeted':
				loss = self.loss_fn(Y_pred, Y)
			if isinstance(target, numbers.Number):
				target = target * torch.ones_like(Y)
			if isinstance(target, list) or isinstance(target, np.ndarray):
				target = torch.LongTensor(target)
			if isinstance(target, torch.Tensor):
				assert target.shape == Y.shape, "PGD target shape mismatch"
				loss = -1.0 * self.loss_fn(Y_pred, target)

			# loss = loss.mean()
			preds   	= torch.argmax(Y_pred, dim=1)
			task_loss	= torch.mean((preds == Y).float())

			loss = houdini(Y_pred, Y, task_loss)

			loss.backward()

			grad_X = X_pgd.grad

			with torch.no_grad():
				X_pgd = X_pgd + a * torch.sign(grad_X)

				X_pgd = torch.where(X_pgd > X + epsilon, X + epsilon, X_pgd)
				X_pgd = torch.where(X_pgd < X - epsilon, X - epsilon, X_pgd)
				X_pgd = torch.clamp(X_pgd, 0, 1) # ensure valid pixel range

			X_pgd = X_pgd.detach()

		return X_pgd


	def loo_bypass(self, X, Y, model, k=40, a=0.01, epsilon=0.3, target='untargeted', rand_start=True):

		X_orig = X.clone()

		if rand_start:
			X = X + epsilon * torch.rand(*X.shape).to(X.device)
			X = torch.clamp(X, 0, 1) # ensure valid pixel range

		for i in range(k):
			X_ = X.clone()
			X_.requires_grad = True
			Y_pred = model.forward(X_)

			if target == 'untargeted':
				loss = self.loss_fn(Y_pred, Y)
			if isinstance(target, numbers.Number):
				target = target * torch.ones_like(Y)
			if isinstance(target, list) or isinstance(target, np.ndarray):
				target = torch.LongTensor(target)
			if isinstance(target, torch.Tensor):
				assert target.shape == Y.shape, "PGD target shape mismatch"
				loss = -1.0 * self.loss_fn(Y_pred, target)

				# # Code for soft targets
				# true_preds	= torch.gather(Y_pred,  1, Y.unsqueeze(1))
				# tar_preds 	= torch.gather(Y_pred, 	1, target.unsqueeze(1))
				# Y_adv 		= Y_pred.clone().detach()
				# Y_adv 		= torch.scatter(Y_adv, 1, target.unsqueeze(1), true_preds)
				# Y_adv 		= torch.scatter(Y_adv, 1, Y.unsqueeze(1), tar_preds)

				# loss = -1.0 * soft_ce(Y_pred, Y_adv)

			loo_loss = 1000 * self.sample_loo_loss(X_, model, 10).mean()

			loss += loo_loss

			loss.backward()

			grad_X = X_.grad

			X = X + a * torch.sign(grad_X)

			X = torch.where(X > X_orig + epsilon, X_orig + epsilon, X)
			X = torch.where(X < X_orig - epsilon, X_orig - epsilon, X)
			X = torch.clamp(X, 0, 1) # ensure valid pixel range

		return X, Y

	def sample_loo_loss(self, X, model, num_sample):
		Y_pred 	= F.softmax(model.forward(X), dim=1)

		max_preds, inds = Y_pred.max(axis=1)

		diffs = torch.zeros(X.shape[0]).to(X.device)

		for _ in range(num_sample):
			X_ = X.clone()

			ind_i 		= torch.randint(0, X_.shape[2], size=(X_.shape[0],)).to(X.device)
			ind_j 		= torch.randint(0, X_.shape[3], size=(X_.shape[0],)).to(X.device)
			comb_ind 	= ind_i * X.shape[3] + ind_j

			orig_shape 	= X_.shape
			X_ 			= X_.view(X_.shape[0], X_.shape[1], -1)
			comb_ind	= comb_ind.view(-1, 1, 1).repeat(1, X_.shape[1], 1)
			X_ 			= torch.scatter(X_, 2, comb_ind, 0.0)
			X_ 			= X_.view(orig_shape)

			Y_new 		= F.softmax(model.forward(X_), dim=1)
			diffs 		+= max_preds - torch.gather(Y_new, 1, inds.unsqueeze(1)).squeeze(1)

		diffs /= num_sample

		return diffs

	def loo_get_iqr(self, X, model):
		Y_pred 	= F.softmax(model.forward(X), dim=1)

		max_preds, inds = Y_pred.max(axis=1)

		diffs = torch.zeros(X.shape[0], X.shape[2], X.shape[3])

		for i in range(X.shape[2]):
			for j in range(X.shape[3]):
				X_ = X.clone()
				X_[:, :, i, j] = 0.0

				with torch.no_grad():
					Y_new = F.softmax(model.forward(X_), dim=1)
					diffs[:, i, j] = max_preds - torch.gather(Y_new, 1, inds.unsqueeze(1)).squeeze(1)

				del X_

		diffs 		= diffs.reshape(diffs.shape[0], -1).cpu().numpy()
		q75, q25 	= np.percentile(diffs, [75 ,25], axis=1)
		iqr 		= q75 - q25

		return iqr

	def loo_train_step(self, X, adv_labels, threshold, optim, model):
		iqr 	= self.loo_get_iqr(X, model)
		iqr 	= torch.Tensor(iqr)
		pred 	= torch.sigmoid(iqr - threshold)
		loss 	= F.binary_cross_entropy(pred, adv_labels)

		loss.backward()
		optim.step()

		print("Loss : {:.4}".format(loss.item()))

		adv_pred = (iqr > threshold)

		return adv_pred

	def loo_detect(self, X, model, threshold):
		iqr 		= self.loo_get_iqr(X, model)
		adv_pred 	= (iqr > threshold)

		return adv_pred

	def noise_detect(self, X, model, adv_labels, threshold):
		sigma 		= 0.3
		noise 		= (sigma ** 2) * torch.randn_like(X)
		X_perturb 	= X.clone() + noise

		Y 			= F.softmax(model.forward(X), dim=1)
		Y_perturb 	= F.softmax(model.forward(X_perturb), dim=1)

		norms 		= torch.sum(torch.abs(Y - Y_perturb), axis=1)

		import pdb; pdb.set_trace()
