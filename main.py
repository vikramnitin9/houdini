from helper import *
from adversarial import Adversarial
from dilation_model import Dilation10

class Main(object):

	def get_batches(self, data, batch_size, split='train', shuffle=None):

		if shuffle is None:
			if self.p.debug: 	shuffle = False
			else:				shuffle = True

		dataset = data[split]
		loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=self.p.num_workers)

		for batch in loader:
			yield batch

	def process_batch(self, batch):
		X, Y    = batch
		X       = X.float().to(self.device)
		Y       = Y.long(). to(self.device)

		return (X, Y)

	def load_data(self):
		# Image size 1024x2048
		self.margin 		= 186
		self.mean 			= (73.16, 82.91, 72.39)
		self.num_classes 	= 19

		class_remapping = {}
		self.palette = []
		count = 0
		for c in datasets.Cityscapes.classes:
			if c.ignore_in_eval is True:
				class_remapping[c.id] = 255
			else:
				class_remapping[c.id] = count
				self.palette.append(c.color)
				count += 1
		assert count == self.num_classes

		data_transform 		= transforms.Compose([
									lambda x : torch.from_numpy(np.array(x, np.float32, copy=False)),
									lambda x : x.permute(2, 0, 1),
									# lambda x : torch.flip(x, (0,)), # RGB to BGR expected by the model
									transforms.Normalize(self.mean, (1.0, 1.0, 1.0)),
									# lambda x : nn.ReflectionPad2d(self.margin)(x.unsqueeze(0)).squeeze(0)
								])
		target_transform 	= transforms.Compose([
									lambda x : torch.from_numpy(np.array(x, np.int32, copy=False)).unsqueeze(0),
									RemapClasses(class_remapping)
								])
		data = {}

		data['train'] 	= datasets.Cityscapes(root='data/cityscape_dataset_large', split='train', 	target_type='semantic', transform=data_transform, target_transform=target_transform)
		data['val'] 	= datasets.Cityscapes(root='data/cityscape_dataset_large', split='val', 	target_type='semantic', transform=data_transform, target_transform=target_transform)
		data['test']	= datasets.Cityscapes(root='data/cityscape_dataset_large', split='test', 	target_type='semantic', transform=data_transform, target_transform=target_transform)

		return data

	def add_model(self):
		model = Dilation10().to(self.device)
		return model

	def __init__(self, args):

		self.p = args

		pprint(vars(self.p))
		self.logger = get_logger(self.p.name, self.p.log_dir)
		self.logger.info(vars(self.p))

		if self.p.gpu != '-1' and torch.cuda.is_available():
			self.device = torch.device('cuda')
			torch.cuda.set_rng_state(torch.cuda.get_rng_state())
			torch.backends.cudnn.deterministic = True
		else:
			self.device = torch.device('cpu')

		self.data = self.load_data()
		self.model = self.add_model()

		self.load_model('pretrained/dilation10_cityscapes_RGB.pth')

	def save_model(self, model, path):
		torch.save(model.state_dict(), path)

	def load_model(self, path):
		self.model.load_state_dict(torch.load(path))

	def predict(self, model, data, split):
		model.eval()
		ious 	= []
		hist 	= np.zeros((self.num_classes, self.num_classes))
		pad  	= nn.ReflectionPad2d(self.margin)

		for step, batch in enumerate(self.get_batches(data, split=split, batch_size=self.p.batch_size)):
			X, Y 	= self.process_batch(batch)

			print("Step {}".format(step))

			with torch.no_grad():
				Y_pred 				= model.forward(pad(X))
				class_prediction  	= torch.argmax(Y_pred, dim=1)
				if self.p.debug:
					self.visualize_seg(class_prediction[0], 'segmentation.png')
					self.visualize_seg(Y[0, 0], 'ground_truth.png')
					self.save_as_img(X[0], 'input.png')
					exit(0)
				class_prediction    = class_prediction.cpu().detach().numpy().flatten()
				target_seg          = Y.cpu().detach().numpy().flatten()
				hist 				+= fast_hist(class_prediction, target_seg, self.num_classes)

		ious = per_class_iu(hist) * 100
		mIoU = np.nanmean(ious)

		print("mIoU : {:.4}".format(mIoU))

		return mIoU

	def eval(self, split='train'):
		val_iou		= self.predict(self.model, self.data, 'val')
		test_iou    = self.predict(self.model, self.data, 'test')

		self.logger.info('[Evaluation]: Valid IoU: {:.4}, Test IoU : {:5}'.format(val_iou, test_iou))

	def save_as_img(self, x, fname):
		if isinstance(x, torch.Tensor): x = np.array(x.detach().cpu())
		x 	+= np.array(self.mean).reshape(3, 1, 1)
		x 	= np.uint8(x.transpose(1, 2, 0))
		x 	= x[self.margin : -self.margin, self.margin : -self.margin]
		im 	= Image.fromarray(x)
		im.save(fname)

	def visualize_seg(self, x, fname):
		if isinstance(x, torch.Tensor): x = np.array(x.detach().cpu(), np.uint8)
		x[x == 255] = 0
		x 	= np.array(self.palette, np.uint8)[x.ravel()].reshape(x.shape[0], x.shape[1], 3)
		im 	= Image.fromarray(x)
		im.save(fname)

	def pgd_attack(self, split='val', approx=False):
		self.model.eval()
		losses, accs, targ_accs = [], [], []

		old_loss, old_acc = self.predict(self.model, self.loss_fn, self.data, split)
		self.logger.info("Running PGD attack on {} set".format(split))

		target_class = 'untargeted'

		adv = Adversarial(self.loss_fn)

		new_X = []
		new_Y = []

		for step, batch in enumerate(self.get_batches(self.data, split=split, batch_size=self.p.batch_size)):

			X, Y 	= self.process_batch(batch)
			X_adv 	= adv.pgd_perturb(X, Y, self.model, k=self.p.k, a=self.p.a, epsilon=self.p.pgd_eps, target=target_class, rand_start=True)

			with torch.no_grad():
				Y_pred  = self.model.forward(X_adv)

			loss 		= self.loss_fn(Y_pred, Y).mean()
			acc     	= self.get_acc(Y_pred, Y)
			if target_class != 'untargeted':
				targ_acc	= self.get_acc(Y_pred, target_class * torch.ones_like(Y))
			else:
				targ_acc 	= torch.tensor(-1.0).to(self.device)

			new_X.append(X_adv.cpu().numpy())
			new_Y.append(Y.cpu().numpy())

			self.logger.info("PGD batch : {}, Accuracy : {:.4}, Target Success : {:.4}".format(step, acc, targ_acc))

			losses.append(loss.item())
			accs.append(acc.item())
			targ_accs.append(targ_acc.item())

			if approx and step == 10: break

		self.logger.info("Before attack : Accuracy {}".format(old_acc))
		self.logger.info("After attack : Accuracy {}, Target Success : {:.4}". format(np.mean(accs), np.mean(targ_accs)))

		new_X = np.concatenate(new_X, axis=0)
		new_Y = np.concatenate(new_Y, axis=0)

		return new_X, new_Y


if __name__== "__main__":

	name_hash = 'test_' + str(uuid.uuid4())[:8]

	parser = argparse.ArgumentParser(description='Houdini attack on Semantic Segmentation Models')

	parser.add_argument('-name',		dest="name",		default=name_hash,              help='name')
	parser.add_argument('-model',		dest="model",		default='res18',type=str,     	help='Model to train')
	parser.add_argument('-batch',		dest="batch_size",	default=4,		type=int,		help='batch_size')
	parser.add_argument('-seed',		dest="seed",		default=42,		type=int,		help='seed')
	parser.add_argument('-gpu',			dest="gpu",			default='0',	type=str,		help='gpu')
	parser.add_argument('-num_workers',	dest="num_workers",	default=8,		type=int,		help='num_workers')
	parser.add_argument('-logdir',		dest="log_dir",		default='log',	type=str,		help='log_dir')
	parser.add_argument('-debug',		dest="debug",		action='store_true',			help='debug')
	# PGD params
	parser.add_argument('-pgd',			dest="pgd",			action='store_true',			help='PGD?')
	parser.add_argument('-k', 			dest="k", 			default=10,		type=int,		help='PGD k')
	parser.add_argument('-a', 			dest="a", 			default=2./255,	type=float,		help='PGD a')
	parser.add_argument('-pgd_eps',		dest="pgd_eps",		default=8./255,	type=float,		help='PGD epsilon')

	args = parser.parse_args()

	args.name = args.name + '_' + time.strftime("%d-%m-%Y") + '_' + time.strftime("%H:%M:%S")

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	set_gpu(args.gpu)

	main = Main(args)

	main.eval()
