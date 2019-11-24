

struct PoseProposalNet
    layers
    lastconv
    function PoseProposalNet(chmodel)
        chmv2 = chmodel.feature_layer
        chlastconv = chmodel.lastconv
        layers = [Convolution2d(chmv2.conv0), ExpandedConvRatio1(chmv2.conv1)]
        for i ∈ 2:17
            chconv = getproperty(chmv2, Symbol("conv$i"))
            push!(layers, ExpandedConv(chconv))
        end
        lastconv = ch2conv(chlastconv)
        new(layers, lastconv)
    end
end


function (ppn::PoseProposalNet)(x)
    h = x
    for lay in ppn.layers
        h = lay(h)
    end
    h = Flux.sigmoid.(ppn.lastconv(h))
    return h
end
