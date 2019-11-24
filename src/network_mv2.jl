clipped_relu(x, bound) = min(max(zero(x), x), typeof(x)(bound))
# prompt to use broadcast
for f in [:clipped_relu]
    @eval $(f)(x::AbstractArray, args...) = error(
    "Use broadcasting (`",
    $(string(f)),
    ".(x)`) to apply activation functions to arrays.",
    )
end

relu6(x) = clipped_relu(x, eltype(x)(6.0))


struct Convolution2d
    conv
    bn
    function Convolution2d(link::PyObject)
        conv = ch2conv(link.conv)
        bn = ch2bn(link.bn)
        new(conv, bn)
    end
end

function (c2d::Convolution2d)(x)
    y = x |> c2d.conv |> c2d.bn .|> relu6
    return y
end

struct ExpandedConvRatio1
    depthwise_conv
    depthwise_bn
    project_conv
    project_bn
    function ExpandedConvRatio1(link::PyObject)
        depthwise_conv = ch2dwconv(link.depthwise_conv)
        depthwise_bn = ch2bn(link.depthwise_bn)
        project_conv = ch2conv(link.project_conv)
        project_bn = ch2bn(link.project_bn)
        new(depthwise_conv, depthwise_bn, project_conv, project_bn)
    end
end

function (ec::ExpandedConvRatio1)(x)
    h = x |> ec.depthwise_conv |> ec.depthwise_bn .|> relu6
    h = h |> ec.project_conv |> ec.project_bn
    if size(h)==size(x)
        return h .+ x
    else
        return h
    end
end

struct ExpandedConv
    expand_conv
    expand_bn
    depthwise_conv
    depthwise_bn
    project_conv
    project_bn
    function ExpandedConv(link::PyObject)
        expand_conv=ch2conv(link.expand_conv)
        expand_bn = ch2bn(link.expand_bn)
        depthwise_conv = ch2dwconv(link.depthwise_conv)
        depthwise_bn = ch2bn(link.depthwise_bn)
        project_conv = ch2conv(link.project_conv)
        project_bn = ch2bn(link.project_bn)
        new(expand_conv,expand_bn,depthwise_conv, depthwise_bn, project_conv, project_bn)
    end
end

function (ec::ExpandedConv)(x)
    h = x |> ec.expand_conv
    h = h |> ec.expand_bn .|> relu6
    h = h |> ec.depthwise_conv |> ec.depthwise_bn .|> relu6
    h = h |> ec.project_conv |> ec.project_bn
    if size(h)==size(x)
        return h .+ x
    else
        return h
    end
end
