local M = {}
local NcnnNet = torch.class('NcnnNet',M)

local Layer = require('Layer')
local Blob = require('Blob')

function NcnnNet:__init()
    self.magic = 0  
    self.layers = {}
    self.layer_index = {}
    self.blobs = {}
    self.layernames = {}
end

--[[
    input 和 output 为 botton 和 top 的 size（#）， 不需要特地设置
]]--

function NcnnNet:setInput(batch_size, c, h, w)
    local layer = Layer()
    layer.type = 'Input'
    layer.name = 'data'
   
    layer.tops[1] = 'data'

    if batch_size > 1 then
        layer.params[1] = batch_size
    else 
        layer.params[1] = -233
    end

    
    layer.params[2] = c
    layer.params[3] = h
    layer.params[4] = w

    self.layers[layer.name] = layer
    self.layernames[#self.layernames + 1] = layer.name
    self.blobs[#self.blobs + 1] = layer.name
    return layer.tops
end

function NcnnNet:addConv(name, bottoms, tlayer)
    local layer = Layer()
    layer.type = 'Convolution'
    layer.name = 'conv_'..name

    layer.tops[1] = layer.name
    layer.bottoms = bottoms

    local weight = tlayer.weight:float()
    local bias = tlayer.bias:float()
    layer.params[1] = tlayer.nOutputPlane
    layer.params[2] = tlayer.kH
    layer.params[3] = 1
    layer.params[4] = tlayer.dH
    layer.params[5] = tlayer.padH


    layer.weights[1] = weight:clone()
    if(bias ~= nil)then
        layer.params[6] = 1
        layer.weights[2] = bias:clone()
    else
        layer.params[6] = 0
    end
    layer.params[7] = weight:nElement()
    self.layers[layer.name] = layer
    self.layernames[#self.layernames + 1] = layer.name
    self.blobs[#self.blobs + 1] = layer.name
end

function NcnnNet:addBN(name, bottoms, tlayer)
    local layer = Layer()
    layer.name = 'bn_'..name
    layer.type = 'BatchNorm'

    layer.tops[1] = layer.name
    layer.bottoms = bottoms 

    local mean = tlayer.running_mean:float()
    local var = tlayer.running_var:float()
    layer.params[1] = mean:nElement() 
    layer.weights[1] = torch.Tensor(mean:size()):fill(1.0)
    layer.weights[2] = mean:clone()
    layer.weights[3] = var:clone()
    layer.weights[4] = torch.Tensor(mean:size()):fill(0.0)
    self.layers[layer.name] = layer
    self.layernames[#self.layernames + 1] = layer.name
    self.blobs[#self.blobs + 1] = layer.name
end

function NcnnNet:addScale(name, bottoms, tlayer)
    local layer = Layer()
    layer.type = 'Scale'
    layer.name = 'scale_'..name

    layer.tops[1] = layer.name
    layer.bottoms[1] = bottoms

    local weights = tlayer.weight:float()
    local bias = tlayer.bias:float()
    layer.params[1] = weights:nElement()
    layer.params[2] = 1
    layer.weights[1] = weights:clone()
    if(bias ~= nil) then
        layer.weights[2] = bias:clone()
    end
    self.layers[layer.name] = layer
    self.layernames[#self.layernames + 1] = layer.name
    self.blobs[#self.blobs + 1] = layer.name
end

function NcnnNet:addReLU(name, bottoms, tlayer)
    local layer = Layer()
    layer.type = 'ReLU'
    layer.name = 'relu_'..name

    layer.tops[1] = layer.name
    layer.bottoms = bottoms

    layer.params[1] = 0.0
    self.layers[layer.name] = layer
    self.layernames[#self.layernames + 1] = layer.name
    self.blobs[#self.blobs + 1] = layer.name
end

function NcnnNet:addConcat(name, bottoms, axis)
    local layer = Layer()
    layer.type = 'Concat'
    layer.name = name

    layer.tops[1] = layer.name
    layer.bottoms = bottoms

    layer.params[1] = axis
    self.layers[layer.name] = layer
    self.layernames[#self.layernames + 1] = layer.name
    self.blobs[#self.blobs + 1] = layer.name
    return layer.tops
end

function NcnnNet:addEltwise(name, bottoms)
    local layer = Layer()
    layer.type = 'Eltwise'
    layer.name = name

    layer.tops[1] = layer.name
    layer.bottoms = bottoms

    layer.params[1] = 1
    layer.params[-23301] = 0

    self.layers[layer.name] = layer
    self.layernames[#self.layernames + 1] = layer.name
    self.blobs[#self.blobs + 1] = layer.name
    return layer.tops
end

function NcnnNet:addPool(name, bottoms, tlayer)
        local layer = Layer()
    layer.type = 'Pooling'
    layer.name = 'pool_'..name

    layer.tops[1] = layer.name
    layer.bottoms = bottoms

    if(string.match(torch.type(tlayer), 'MaxPooling')) then
        layer.params[1] = 0
    elseif(string.match(torch.type(tlayer), 'AveragePooling')) then
        layer.params[1] = 1
    else
        print(" ##ERROR## unspported pooling layer:" .. mtype)
        assert(false)        
    end
    layer.params[2] = tlayer.kH
    layer.params[3] = tlayer.dH
    layer.params[4] = 0
    layer.params[5] = 0
    layer.params[6] = 0

    self.layers[layer.name] = layer
    self.layernames[#self.layernames + 1] = layer.name
    self.blobs[#self.blobs + 1] = layer.name
end

-- TODO check weight filler
function NcnnNet:addDeconvolution(name, bottoms, tlayer)
    local layer = Layer()
    layer.type = 'Deconvolution'
    layer.name = 'upsample_'..name

    layer.tops[1] = layer.name
    layer.bottoms = bottoms

    local num_output = tlayer.outputSize[1]
    local h_input = tlayer.inputSize[2]
    local w_input = tlayer.inputSize[3]
    local scale_factor = tlayer.scale_factor
    local kernel_size = 2*scale_factor - scale_factor%2 
    layer.params[1] = num_output
    layer.params[2] = kernel_size
    layer.params[3] = 1
    layer.params[4] = scale_factor
    layer.params[5] = torch.ceil((scale_factor - 1)/2.0)
    layer.params[6] = 0 -- bias_term
    layer.params[7] = num_output * kernel_size * kernel_size-- weight_data_size

    local weight = torch.tensor(num_output, kernel_size, kernel_size)
    local f = torch.ceil(kernel_size/2.0)
    local c = (kernel_size - 1)(2.0*f)
    for i = 1, num_output do
        for y = 1, kernel_size do
            for x = 1, kernel_size do
                weight[i][y][x] = (1-torch.abs(x/f - c))*(1-torch.abs(y/f - c))
            end
        end
    end

    layer.weights = weight:clone()
    self.layers[layer.name] = layer
    self.layernames[#self.layernames + 1] = layer.name
    self.blobs[#self.blobs + 1] = layer.name
end


function NcnnNet:addSplit(index, bottom, topsize)
    local layer = Layer()
    layer.type = 'Split'
    layer.name = 'splitncnn_'..index 

    layer.bottoms = bottom

    for i = 1, topsize do
        layer.tops[i] = bottom[1].. 'splitncnn_'..(i-1)
        self.blobs[#self.blobs + 1] = bottom[1].. 'splitncnn_'..(i-1)
    end

    self.layers[layer.name] = layer
    self.layernames[#self.layernames + 1] = layer.name
    return layer.tops
end

function NcnnNet:addFlatten(name,bottom)
    local layer = Layer()
    layer.type = 'Flatten'
    layer.name =  'faltten_'..name
    layer.tops[1] = layer.name
    layer.bottoms = bottom

    self.layers[layer.name] = layer
    self.layernames[#self.layernames + 1] = layer.name
    self.blobs[#self.blobs + 1] = layer.name
end

function NcnnNet:addInnerProduct(name, bottom, tlayer)
    local layer = Layer()
    layer.type = 'InnerProduct'
    layer.name =  name
    layer.tops[1] = layer.name
    layer.bottoms = bottom

    layer.params[1] = tlayer.weight:size(1)

    if(bias ~= nil)then
        layer.params[2] = 1
        layer.weights[2] = tlayer.bias:clone()
    else
        layer.params[2] = 0
    end
    layer.params[3] = tlayer.weight:nElement()
    layer.weights[1] = tlayer.weight:clone()
print(layer.weights[1]:size())
    self.layers[layer.name] = layer
    self.layernames[#self.layernames + 1] = layer.name
    self.blobs[#self.blobs + 1] = layer.name
end


function NcnnNet:addBinActive(name, bottoms, tlayer)
    local layer = Layer()
    layer.type = 'BinaryActive'
    layer.name = 'bactive_'..name

    layer.tops[1] = layer.name
    layer.bottoms = bottoms

    self.layers[layer.name] = layer
    self.layernames[#self.layernames + 1] = layer.name
    self.blobs[#self.blobs + 1] = layer.name
end

function NcnnNet:test(name,bottom)
    print('name'..name)
    print(bottom)
end

function NcnnNet:addLayer(name, bottoms, tlayer)
    local mtype = torch.type(tlayer)
    if(string.match(mtype, 'Convolution')) then
        self:addConv(name, bottoms, tlayer)
    elseif(string.match(mtype, 'BatchNorm'))then
        self:addBN(name, bottoms, tlayer)
        if( tlayer.affine == true ) then
            self:addScale(name, 'bn_'..name, tlayer)
        end
    elseif(string.match(mtype, 'ReLU')) then
        self:addReLU(name, bottoms, tlayer)
    elseif(string.match(mtype, 'Pooling')) then
        self:addPool(name, bottoms, tlayer)
    elseif(string.match(mtype, 'Upsampling')) then
        self:addDeconvolution(name, bottoms, tlayer)
    elseif(string.match(mtype, 'View')) then
        self:addFlatten(name, bottoms)
    elseif(string.match(mtype, 'Linear')) then
        self:addInnerProduct(name, bottoms, tlayer)
    elseif(string.match(mtype, 'Binary')) then
        self:addBinActive(name, bottoms, tlayer)
    elseif(string.match(mtype, 'bnn.SpatialConvolution')) then
        self:addBinConv(name, bottoms, tlayer)
    else 
        print(" ##ERROR## unspported layer:" .. mtype)
        assert(false)
    end
    local lastLayer = self.layers[self.layernames[#self.layernames]]
    return lastLayer.tops
end

function NcnnNet:toNcnnParam(paramFile)
    local file = torch.DiskFile(paramFile, 'w')
    file:writeString('7767517\r\n')
    file:writeString(#self.layernames..' '..#self.blobs..'\r\n')
    for i = 1, #self.layernames do
        local layer = self.layers[self.layernames[i]]
        file:writeString(layer:toString())
        file:writeString('\r\n')
    end
end

function NcnnNet:toNcnnBin(binFile)
    local file = torch.DiskFile(binFile, 'w'):binary()

    for i = 1, #self.layernames do
        local layer = self.layers[self.layernames[i]]
        if(string.match(layer.type, 'Convolution') or string.match(layer.type, 'InnerProduct') ) then
            file:writeChar(0)
            file:writeChar(0)
            file:writeChar(0)
            file:writeChar(0)
        end

        for j = 1, #layer.weights do
            if(string.match(layer.type, 'InnerProduct')) then
            end
            for m = 1, layer.weights[j]:nElement() do
                file:writeFloat(layer.weights[j]:storage()[m])
            end
         end
    end
end

return M.NcnnNet


-- torch API 
--[[
    torch math.md
]]--
