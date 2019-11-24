# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Julia 1.2.0
#     language: julia
#     name: julia-1.2
# ---

using Images
using ImageDraw
using PyCall
using PyPlot
using Revise
using Pkg

pkg"add ~/work/Gomah.jl"

using PPN
import Gomah: reversedims

chainer = pyimport("chainer")
chainercv = pyimport("chainercv")

chainercv = pyimport("chainercv")
chimg = pycall(chainercv.utils.read_image, PyObject, "lena.jpeg", dtype = :uint8)
chimg = chainercv.transforms.resize(chimg, (224, 224))
chainercv.visualizations.vis_image(chimg)

img = load("lena.jpeg")
cvimg = channelview(img)
jlimg = Float32.(rawview(cvimg))
chimg = Float32.(chimg)
@show chimg[1:10]
@show jlimg[1:10]
img

# # Run inference with Chainer model

# +
function inferwithchainer()
    chmodel = pyppn.load()
    chimg = pycall(chainercv.utils.read_image, PyObject, "lena.jpeg", dtype = :float32)
    chimg = chainercv.transforms.resize(chimg, (224, 224))
    @show size(chimg)
    @pywith chainer.using_config("train", false) begin
        inp = reshape(chimg, 1, size(chimg)...)
        y = chmodel(inp).array
        @show size(y)
        return y
    end
    # bug ???
    # return y
end

chret = inferwithchainer();
@show chret |> size
# -

# # Run inference with Flux model

# +
function inferwithflux()
    chmodel = pyppn.load()
    chimg = pycall(chainercv.utils.read_image, PyObject, "lena.jpeg", dtype = :float32)
    chimg = chainercv.transforms.resize(chimg, (224, 224))
    @show size(chimg)
    flmodel = PPN.PPN(chmodel)
    inp = reshape(Float32.(chimg), 1, size(chimg)...)
    @show size(inp)
    @show reversedims(inp) |> size
    y = reversedims(flmodel(reversedims(inp)))
    @show size(y)
end

flret = inferwithflux()

# +
function inferwithpureflux()
    chmodel = pyppn.load()
    flmodel = PPN.PPN(chmodel)
    img = "lena.jpeg" |> load
    resized = imresize(img, 224, 224)
    inp = resized |> channelview |> rawview .|> Float32
    inp = reshape(inp, 1, size(inp)...)
    y = reversedims(flmodel(reversedims(inp)))
end

pureflret = inferwithpureflux()
# -

# # Inference
py"""
import numpy as np
ROOT_NODE=0
import logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
from chainercv.utils import non_maximum_suppression
def get_instances(delta, bbox,detection_thresh=0.125):
    root_bbox = bbox[ROOT_NODE]
    score = delta[ROOT_NODE]
    candidate = np.where(score > detection_thresh)
    score = score[candidate]
    logger.info("root_bbox.shape={}".format(root_bbox.shape))
    root_bbox = root_bbox[candidate]
    logger.info("root_bbox.shape={}".format(root_bbox.shape))
    selected = non_maximum_suppression(
        bbox=root_bbox,
        thresh=0.3,
        score=score
    )
    return candidate[0][selected],candidate[1][selected]
"""

# +
const detection_thresh = 0.125
const KEYPOINT_NAMES = pyppn.pose_utils.KEYPOINT_NAMES
const EDGES = 1 .+ pyppn.pose_utils.EDGES
const K = length(KEYPOINT_NAMES)
const E = length(EDGES[:, 1])
const locH = 5
const locW = 5
const inH, inW = 224, 224
const outH = inH ÷ 32
const outW = inW ÷ 32
const gridH, gridW = inH ÷ outH, inW ÷ outW
const COLOR_MAP = pyppn.pose_utils.COLOR_MAP

color_map(k) = RGB{N0f8}((COLOR_MAP[KEYPOINT_NAMES[k]] ./ 255f0...))

_X = [i for i = 0:outW-1, j = 0:outH-1]'
const X = reshape(_X, 1, outH, outH)
_Y = [i for i = 0:outW-1, j = 0:outH-1]
const Y = reshape(_Y, 1, outH, outH)

restore_size(h, w) = (h * inH, w * inW)
restore_xy(y, x) = (@. (y + Y) * gridH, @. (x + X) * gridW)

function extract_feature(feature)
    feature = @view feature[1, :, :, :]
    resp = @view feature[1:K, :, :]
    conf = @view feature[K+1:2K, :, :]
    delta = @views resp .* conf
    x = @view feature[2K+1:3K, :, :]
    y = @view feature[3K+1:4K, :, :]
    w = @view feature[4K+1:5K, :, :]
    h = @view feature[5K+1:6K, :, :]
    #=
    e = reshape(
        feature[6K+1:end,:,:],
        E,
        locH,locW,
        outH,outW,
    )
    =#
    e = reversedims(feature[6K+1:end, :, :])
    e = reshape(e, outW, outH, locW, locH, E)
    e = reversedims(e)
    return delta, x, y, w, h, e
end

# +
function non_maximum_suppression(;bbox,thresh,score)
    if length(bbox)==0 && return [CartesianIndex(1,1)] end
    order = sortperm(score)
    bbox=reshape(bbox,length(bbox)÷4,4)
    bbox = bbox[order,:]
    # define `bbox_area` by (ymax-ymin)*(xmax-xmin)
    bbox_area=prod(bbox[:,3:4] .- bbox[:,1:2],dims=2)
    selec = zeros(Bool,size(bbox)[1])
    empty = zeros(Bool,size(bbox)[1])
    for i in 1:size(bbox)[1]
        b=bbox[i,:]
        if selec != empty
            tl = max.(reshape(b[1:2],(1,2)), bbox[selec, 1:2])
            br = min.(reshape(b[3:4],(1,2)), bbox[selec, 3:4])
            area = prod(br .- tl, dims=1) .* all(tl .< br, dims=1)
            iou = @. area / (bbox_area[i] + bbox_area[selec] - area)
            if any(iou .>= thresh) && continue end
        end
        selec[i] = true
    end
    return order[selec]
end

function get_instances(delta,bbox,detection_thresh=0.125)
    root_bbox = bbox[1,:,:,:]
    score = delta[1,:,:]
    candidate = score .> detection_thresh
    score = score[candidate]
    root_bbox = root_bbox[cat(candidate,candidate,candidate,candidate,dims=3)]
    selected = non_maximum_suppression(
        bbox=root_bbox,
        thresh=0.3,
        score=score,
    )
    inds=findall(isequal(true), candidate)[selected]
    return [i[1] for i in inds],[i[2] for i in inds]
end



# +
const Human = Dict{Int64,Array{Float32,1}}
function estimate()
    feature = inferwithchainer()
    delta, x, y, w, h, e = extract_feature(feature)
    ry, rx = restore_xy(y, x)
    rh, rw = restore_size(h, w)
    ymin, ymax = @. ry - rh / 2, ry + rh / 2
    xmin, xmax = @. rx - rw / 2, rx + rw / 2
    bbox = cat(ymin, xmin, ymax, xmax, dims = 4)
    e = permutedims(e, (1, 4, 5, 2, 3))
    hs,ws=get_instances(delta,bbox)
    #=hs, ws = py"get_instances"(delta, bbox)
    hs .+= 1
    ws .+= 1
    =#
    graphs = 1 .+ pyppn.pose_utils.DIRECTED_GRAPHS
    humans = Human[]
    for hxw in zip(hs, ws)
        human = Dict(1 => bbox[1, hxw[1], hxw[2], :])
        @show typeof(human)
        for gi in [1, 2, 3, 4, 5, 6]
            (i_h, i_w) = hxw
            eis, ts = graphs[gi, 1, :], graphs[gi, 2, :]
            for (ei, t) in zip(eis, ts)
                index = (ei, i_h, i_w)
                uh, uw = Tuple(argmax(e[index..., :, :]))
                j_h = i_h + uh - (div(locH, 2) + 1)
                j_w = i_w + uw - (div(locW, 2) + 1)
                if j_h <= 0 || j_w <= 0 || j_h > outH || j_w > outW
                    break
                end
                if delta[t, j_h, j_w] < detection_thresh
                    break
                end
                human[t] = bbox[t, j_h, j_w, :]
                i_h, i_w = j_h, j_w
            end
        end
        push!(humans, human)
    end
    humans
end

humans = estimate()

# +
chimg = pycall(chainercv.utils.read_image, PyObject, "lena.jpeg", dtype = :float32)
chimg = chainercv.transforms.resize(chimg, (224, 224))
jlimg = colorview(RGB{N0f8}, N0f8.(chimg / 256f0))

function draw_bbox(img, ymin, xmin, ymax, xmax)
    vert = [
        CartesianIndex(ymin, xmin),
        CartesianIndex(ymin, xmax),
        CartesianIndex(ymax, xmax),
        CartesianIndex(ymax, xmin),
    ]
    draw!(img, Polygon(vert))
end

function draw_humans(img, humans::Vector{Human})
    for human in humans
        r = 5
        for (k, bbox) in human
            ymin, xmin, ymax, xmax = Int.(floor.(bbox))
            if k == 1
                @show bbox
                draw_bbox(jlimg, ymin, xmin, ymax, xmax)
            else
                x = (xmin + xmax) ÷ 2
                y = (ymin + ymax) ÷ 2
                draw!(jlimg, CirclePointRadius(Point(x, y), r), color_map(k))
            end
            for (s, t) in zip(EDGES[:, 1], EDGES[:, 2])
                if s in keys(human) && t in keys(human)
                    by = (human[s][1] + human[s][3]) ÷ 2 |> floor |> Int
                    bx = (human[s][2] + human[s][4]) ÷ 2 |> floor |> Int
                    ey = (human[t][1] + human[t][3]) ÷ 2 |> floor |> Int
                    ex = (human[t][2] + human[t][4]) ÷ 2 |> floor |> Int
                    bx, by, ex, ey
                    draw!(jlimg, LineSegment(bx, by, ex, ey), color_map(s))
                end
            end
        end
    end
end

@show typeof(humans)
draw_humans(jlimg, humans)
jlimg
# -




