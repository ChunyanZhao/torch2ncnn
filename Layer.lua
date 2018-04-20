local M = {}
local Layer = torch.class('Layer',M)

function Layer:__init()
    self.type = ''
    self.name = ''
    self.index = 0
    self.bottoms = {}
    self.tops = {}

    self.params = {}
    self.weights = {}
end

function Layer:toString()
    local layer = ''
    layer = layer..self.type
    layer = layer..'\t'..self.name

    layer = layer..'\t'..#self.bottoms
    layer = layer..' '..#self.tops

    for i = 1, #self.bottoms do 
        layer = layer..' '..self.bottoms[i]
    end

    for i = 1, #self.tops do 
        layer = layer..' '..self.tops[i]
    end

    if(self.type == 'Eltwise') then
        layer = layer..' 0=1 -23301=0'   
    else
        for i = 1, #self.params do 
            layer = layer..' '..(i-1)..'='..self.params[i]                
        end
    end

    return layer
end

return M.Layer
