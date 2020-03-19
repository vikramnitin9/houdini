from helper import *

class Dilation10(nn.Module):

	def __init__(self):

		super(Dilation10, self).__init__()

		self.conv1_1	= nn.Conv2d(in_channels=3,		out_channels=64,	kernel_size=3, 	dilation=1)
		self.conv1_2	= nn.Conv2d(in_channels=64,  	out_channels=64,  	kernel_size=3, 	dilation=1)
		self.pool1 		= nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv2_1	= nn.Conv2d(in_channels=64,  	out_channels=128, 	kernel_size=3, 	dilation=1)
		self.conv2_2	= nn.Conv2d(in_channels=128, 	out_channels=128, 	kernel_size=3, 	dilation=1)
		self.pool2 		= nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv3_1	= nn.Conv2d(in_channels=128,  	out_channels=256, 	kernel_size=3, 	dilation=1)
		self.conv3_2	= nn.Conv2d(in_channels=256, 	out_channels=256, 	kernel_size=3, 	dilation=1)
		self.conv3_3	= nn.Conv2d(in_channels=256, 	out_channels=256, 	kernel_size=3, 	dilation=1)
		self.pool3 		= nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv4_1	= nn.Conv2d(in_channels=256,  	out_channels=512, 	kernel_size=3, 	dilation=1)
		self.conv4_2	= nn.Conv2d(in_channels=512, 	out_channels=512, 	kernel_size=3, 	dilation=1)
		self.conv4_3	= nn.Conv2d(in_channels=512, 	out_channels=512, 	kernel_size=3, 	dilation=1)

		self.conv5_1	= nn.Conv2d(in_channels=512,  	out_channels=512, 	kernel_size=3, 	dilation=2)
		self.conv5_2	= nn.Conv2d(in_channels=512, 	out_channels=512, 	kernel_size=3, 	dilation=2)
		self.conv5_3	= nn.Conv2d(in_channels=512, 	out_channels=512, 	kernel_size=3, 	dilation=2)

		self.fc6		= nn.Conv2d(in_channels=512, 	out_channels=4096, 	kernel_size=7, 	dilation=4)
		self.fc7		= nn.Conv2d(in_channels=4096, 	out_channels=4096, 	kernel_size=1, 	dilation=1)

		self.final 		= nn.Conv2d(in_channels=4096, 	out_channels=19, 	kernel_size=1, 	dilation=1)

		self.ctx_conv1_1 = nn.Conv2d(in_channels=19, 	out_channels=19, 	kernel_size=3, 	dilation=1,	padding=1)
		self.ctx_conv1_2 = nn.Conv2d(in_channels=19, 	out_channels=19, 	kernel_size=3, 	dilation=1,	padding=1)

		self.ctx_conv2_1 = nn.Conv2d(in_channels=19, 	out_channels=19, 	kernel_size=3, 	dilation=2,	padding=2)
		self.ctx_conv3_1 = nn.Conv2d(in_channels=19, 	out_channels=19, 	kernel_size=3, 	dilation=4,	padding=4)
		self.ctx_conv4_1 = nn.Conv2d(in_channels=19, 	out_channels=19, 	kernel_size=3, 	dilation=8,	padding=8)
		self.ctx_conv5_1 = nn.Conv2d(in_channels=19, 	out_channels=19, 	kernel_size=3, 	dilation=16,padding=16)
		self.ctx_conv6_1 = nn.Conv2d(in_channels=19, 	out_channels=19, 	kernel_size=3, 	dilation=32,padding=32)
		self.ctx_conv7_1 = nn.Conv2d(in_channels=19, 	out_channels=19, 	kernel_size=3, 	dilation=64,padding=64)

		self.ctx_fc1 	= nn.Conv2d(in_channels=19, 	out_channels=19, 	kernel_size=3,	dilation=1, padding=1)
		self.ctx_final 	= nn.Conv2d(in_channels=19, 	out_channels=19, 	kernel_size=1, 	dilation=1)

		self.ctx_upsample = nn.ConvTranspose2d(in_channels=19, out_channels=19, kernel_size=16, stride=8, padding=4, groups=19, bias=False)

	def forward(self, x):
		x = F.relu(self.conv1_1(x))
		x = F.relu(self.conv1_2(x))
		x = self.pool1(x)

		x = F.relu(self.conv2_1(x))
		x = F.relu(self.conv2_2(x))
		x = self.pool2(x)

		x = F.relu(self.conv3_1(x))
		x = F.relu(self.conv3_2(x))
		x = F.relu(self.conv3_3(x))
		x = self.pool3(x)

		x = F.relu(self.conv4_1(x))
		x = F.relu(self.conv4_2(x))
		x = F.relu(self.conv4_3(x))

		x = F.relu(self.conv5_1(x))
		x = F.relu(self.conv5_2(x))
		x = F.relu(self.conv5_3(x))

		x = F.dropout(F.relu(self.fc6(x)), p=0.5)
		x = F.dropout(F.relu(self.fc7(x)), p=0.5)

		x = self.final(x)

		x = F.relu(self.ctx_conv1_1(x))
		x = F.relu(self.ctx_conv1_2(x))

		x = F.relu(self.ctx_conv2_1(x))
		x = F.relu(self.ctx_conv3_1(x))
		x = F.relu(self.ctx_conv4_1(x))
		x = F.relu(self.ctx_conv5_1(x))
		x = F.relu(self.ctx_conv6_1(x))
		x = F.relu(self.ctx_conv7_1(x))

		x = F.relu(self.ctx_fc1(x))
		x = self.ctx_final(x)

		x = F.softmax(self.ctx_upsample(x), dim=1)

		return x