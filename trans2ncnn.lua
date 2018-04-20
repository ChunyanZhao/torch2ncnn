require('nn')
require('cunn')
require('cudnn')

local LayerParameter = require('LayerParameter')

--local torchNet = torch.load('./resnet-34.t7')


function data(inputs)
    layer = LayerParameter(ncnn_type)

    input_shape = inputs:size()
print('input_shape:')
print(input_shape)
    for dim = 1,4 do
        if dim -1 < input_shape:size() then
            size = input_shape[dim]
print('size: '..size)
        else
            size = -233
        end
        layer.param[#layer.param + 1] = size

    end
print(layer.param)
end

function Slice(torch_layer)

end

function Split(torch_layer)

end

function permute(torch_layer)

end

function flatten(torch_layer)

end

function inner_product(torch_layer)
    layer = LayerParameter(ncnn_type)

    num_output = torch_layer.nOutputPlane
    blobs_weight = torch_layer.weight:float()
    bias = torch_layer.bias:float()

    layer.param[#layer.param + 1] = num_output

    if(bias ~= nil) then
        layer.param[#layer.param + 1] = true
        layer.param[#layer.param + 1] = blobs_weight.size(1)
        layer.weights[#layer.weights + 1] = blobs_weight
        layer.weights[#layer.weights + 1] = bias
    else
        layer.param[#layer.param + 1] = false
        layer.param[#layer.param + 1] = blobs_weight.size(1)
        layer.weights[#layer.weights + 1] = blobs_weight   
    end
print(layer.param)
end

function concat(torch_layer)

end

function spatial_convolution(torch_layer)

end

function binary_convolution(torch_layer)

end

function FillBilinear(torch_layer)

end

function UpsampleBilinear(torch_layer)

end

function CopyPoolingParamater(torch_layer)

end

function MaxPooling(torch_layer)

end

function AvgPooling(torch_layer)

end

function dropput(torch_layer)

end

function elu(torch_layer)

end

function binary_active(torch_layer)

end

function ReLU(torch_layer)

end

function leaky_ReLU(torch_layer)

end

function PReLU(torch_layer)

end

function MulConst(torch_layer)

end

function AddConst(torch_layer)

end

function softmax(torch_layer)

end

function eltwise(torch_layer)

end

function eltwise_max(torch_layer)

end

function negate(torch_layer)

end

function batchnorm(torch_layer)

end

function ty(ncnn_type)

end

function toBatchNorm(tm)
    layer_bn = LayerParameter('BatchNorm')
    layer_scale = LayerParameter('Scale')
    if ( tm.affine == true) then
        --assert(type(layerName) == 'table')
        --assert(#layerName == 2)
        local weights = tm.weight:float()
        local bias = tm.bias:float()
        local mean = tm.running_mean:float()
        local var = tm.running_var:float()
        layer_bn.param[#layer_bn.param + 1] = mean:nElement()
        layer_bn.weights[#layer_bn.weights+1] = mean:size()
        layer_bn.weights[#layer_bn.weights+1] = mean
        layer_bn.param[#layer_bn.param + 1] = mean:nElement()
        layer_bn.weights[#layer_bn.weights+1] = var

        layer_scale.param[#layer_scale.param + 1] = weights:nElement()
        layer_scale.param[#layer_scale.param + 1] = true
        layer_scale.weights[#layer_scale.weights+1] = weights
        layer_scale.weights[#layer_scale.weights+1] = bias
        --C.writeCaffeBNLayer(caffeNet[0], layerName[1], mean, var);
        --C.writeCaffeScaleLayer(caffeNet[0], layerName[2], weights, bias);
    else
        --assert(type(layerName) == 'string')
        local mean = tm.running_mean:float()
        local var = tm.running_var:float()
        layer_bn.param[#layer_bn.param + 1] = mean:nElement()
        layer_bn.weights[#layer_bn.weights+1] = mean:size()
        layer_bn.weights[#layer_bn.weights+1] = mean
        layer_bn.param[#layer_bn.param + 1] = mean:nElement()
        layer_bn.weights[#layer_bn.weights+1] = var
        --C.writeCaffeBNLayer(caffeNet[0], layerName[0], mean, var);
    end

    return {layer_bn, layer_scale}
end

function toConv(tm)
    layer = LayerParameter('Convolution')

    layer.param[#layer.param + 1] = tm.nOutputPlane
    layer.param[#layer.param + 1] = tm.kW
    layer.param[#layer.param + 1] = 1
    layer.param[#layer.param + 1] = tm.dW
    layer.param[#layer.param + 1] = tm.padW
 
    local weights = tm.weight:float()  
    layer.weights[#layer.weights+1] = weight
 
    if(tm.bias ~= nil) then
        layer.param[#layer.param + 1] = true
        layer.param[#layer.param + 1] = weight:nElement()
        layer.weights[#layer.weights+1] = bias
    else
        layer.param[#layer.param + 1] = false
        layer.param[#layer.param + 1] = weight:nElement()
    end
end

function build_convert(tlayer)
    local layer = nil

    local mtype = torch.type(tlayer)
    if ( mtype == 'nn.Linear' ) then
        layer = toinner_product(tlayer)
    elseif ( mtype == 'nn.BatchNormalization' or mtype == 'nn.SpatialBatchNormalization' ) then
        layer = toBatchNorm(tlayer)
    elseif ( string.match(mtype, 'SpatialConvolution')) then
        layer = toConv(tlayer)
    elseif ( string.match(mtype, 'BinaryConvolution')) then
        layer = toBinConv(tlayer)
    elseif ( string.match(mtype, 'AveragePooling')) then
        layer = toAve(tlayer)
    elseif ( string.match(mtype, 'ReLU')) then
        layer = toReLU(tlayer)
    elseif ( string.match(mtype, 'MaxPooling')) then
        layer = toMax(tlayer)
    elseif ( string.match(mtype, 'MaxPooling')) then
        layer = toMax(tlayer)
    else
        print(" ##ERROR## unspported layer:" .. mtype)
        assert(false)
    end

    return layer
end


x = torch.Tensor(1,3,4,5):fill(1)
print(x:nElement())


--  API
--  tensor: size() 返回 dimension 数组
--          nElement() 返回 所有元素个数



