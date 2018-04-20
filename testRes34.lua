require('nn')
require('cunn')
require('cudnn')

local NcnnNet = require('NcnnNet')

torch.setdefaulttensortype('torch.FloatTensor')

local torchNet = torch.load('./resnet-34.t7')

local ncnnParamFile = 'res34.param'
local ncnnBinFile = 'res34.bin'

--print(torchNet)


local ncnnNet = NcnnNet()

ncnnNet:setInput(1,3, 224, 224)

-- (1)-(2)-(3)-(4)
local name = '1'
local bottoms = {'data'}
local tlayer = torchNet:get(1)
bottoms = ncnnNet:addLayer(name, bottoms, tlayer)
tlayer = torchNet:get(2)
bottoms = ncnnNet:addLayer(name, bottoms, tlayer)
tlayer = torchNet:get(3)
bottoms = ncnnNet:addLayer(name, bottoms, tlayer)
tlayer = torchNet:get(4)
bottoms = ncnnNet:addLayer(name, bottoms, tlayer)
splitIdx = 0
function resModule(name, bottoms, tlayer,flags)
    local cuTlayer = tlayer:get(1)
    local cuBottoms = {}
    local cuName = ''

    local splitBlobs = ncnnNet:addSplit(splitIdx, bottoms, 2)
    splitIdx  = splitIdx + 1

    cuName = name..'_branch2a'
    cuBottoms[1] = splitBlobs[2]
    cuBottoms = ncnnNet:addLayer(cuName, cuBottoms, cuTlayer:get(1):get(1))
    cuBottoms = ncnnNet:addLayer(cuName, cuBottoms, cuTlayer:get(1):get(2))
    cuName = name..'_branch2'
    cuBottoms = ncnnNet:addLayer(cuName, cuBottoms, cuTlayer:get(1):get(3))
    cuName = name..'_branch2b'
    cuBottoms = ncnnNet:addLayer(name, cuBottoms, cuTlayer:get(1):get(4))
    cuBottoms = ncnnNet:addLayer(name, cuBottoms, cuTlayer:get(1):get(5))
    if(flags) then
        cuName = name..'_branch1'
        brachBottom = {splitBlobs[1]}
        brachBottom = ncnnNet:addLayer(cuName, brachBottom, cuTlayer:get(2))
        cuBottoms = {brachBottom[1], cuBottoms[1]}
        cuBottoms = ncnnNet:addEltwise(name, cuBottoms)
    else
        cuBottoms = {splitBlobs[1], cuBottoms[1]}
        cuBottoms = ncnnNet:addEltwise(name, cuBottoms)
    end
    cuName = name
    cuBottoms = ncnnNet:addLayer(cuName, cuBottoms, tlayer:get(3))
    return cuBottoms 
end

tlayer = torchNet:get(5)
for i = 1,3 do
    name = 'res5'..i
    bottoms = resModule(name, bottoms, tlayer:get(i))
end

tlayer = torchNet:get(6)
    name = 'res61'
    bottoms = resModule(name, bottoms, tlayer:get(1),1)
for i = 2,4 do
    name = 'res6'..i
    bottoms = resModule(name, bottoms, tlayer:get(i))
end

tlayer = torchNet:get(7)
    name = 'res71'
    bottoms = resModule(name, bottoms, tlayer:get(1),1)
for i = 2,6 do
    name = 'res7'..i
    bottoms = resModule(name, bottoms, tlayer:get(i))
end

tlayer = torchNet:get(8)
    name = 'res81'
    bottoms = resModule(name, bottoms, tlayer:get(1),1)
for i = 2,3 do
    name = 'res8'..i
    bottoms = resModule(name, bottoms, tlayer:get(i))
end

tlayer = torchNet:get(9)
name = '2'
bottoms = ncnnNet:addLayer(name, bottoms, tlayer)

tlayer = torchNet:get(10)
name = '2'
bottoms = ncnnNet:addLayer(name, bottoms, tlayer)

tlayer = torchNet:get(11)
name = 'prob'
bottoms = ncnnNet:addLayer(name, bottoms, tlayer)

ncnnNet:toNcnnParam(ncnnParamFile)
ncnnNet:toNcnnBin(ncnnBinFile)





