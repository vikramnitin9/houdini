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

	def __init__(self, loss_fn, pixel_min=[0.0, 0.0, 0.0], pixel_max=[1.0, 1.0, 1.0]):
		self.loss_fn 	= loss_fn
		self.pixel_min 	= np.array(pixel_min).reshape(1, -1, 1, 1)
		self.pixel_max 	= np.array(pixel_max).reshape(1, -1, 1, 1)

	def valid_pixel_range(self, X):
		# TODO
		return X

	def pgd_perturb(self, X, Y, model, k=40, a=2, epsilon=4, target='untargeted', rand_start=True):

		X_pgd = X.clone()

		if rand_start:
			X_pgd = X_pgd + (2 * epsilon * torch.rand(*X_pgd.shape).to(X_pgd.device) - epsilon)
			X_pgd = self.valid_pixel_range(X_pgd)

		for i in range(k):

			print(i)

			X_pgd.requires_grad = True
			Y_pred = model.forward(X_pgd)

			if target == 'untargeted':
				loss = self.loss_fn(Y_pred, Y)
			if isinstance(target, np.ndarray):
				target = torch.LongTensor(target)
			if isinstance(target, torch.Tensor):
				assert target.shape == Y.shape, "PGD target shape mismatch"
				loss = -1.0 * self.loss_fn(Y_pred, target)

			import pdb; pdb.set_trace()
			loss.backward()

			grad_X = X_pgd.grad

			with torch.no_grad():
				X_pgd = X_pgd + a * torch.sign(grad_X)

				X_pgd = torch.where(X_pgd > X + epsilon, X + epsilon, X_pgd)
				X_pgd = torch.where(X_pgd < X - epsilon, X - epsilon, X_pgd)
				X_pgd = self.valid_pixel_range(X_pgd)

			X_pgd = X_pgd.detach()

		return X_pgd
