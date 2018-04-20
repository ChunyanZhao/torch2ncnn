
require 'torch'
require 'nn'
require('cunn')
require('cudnn')
require 'image'
require 'xlua'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   --std = { 0.229, 0.224, 0.225 },
   std = {0.003921568, 0.003921568, 0.003921568},
}

local torchNet = torch.load('./resnet-34.t7')
torchNet:evaluate()
--print(torchNet)
local img = image.load('./dog224.png')
print(img)
for i=1,3 do
 img[i]:add(-meanstd.mean[i])
 --img[i]:div(meanstd.std[i])
end
img = img:view(1,3,224,224):cuda()
output = torchNet:forward(img)
max,idx = torch.max(output:view(1000,1),1)
print(max)
print(idx)
