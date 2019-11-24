const detection_thresh = 0.125
const KEYPOINT_NAMES = pypose_utils.KEYPOINT_NAMES
const EDGES = 1 .+ pypose_utils.EDGES
const COLOR_MAP = pypose_utils.COLOR_MAP
const DIRECTED_GRAPHS = 1 .+ pypose_utils.DIRECTED_GRAPHS
const K = length(KEYPOINT_NAMES)
const E = length(EDGES[:, 1])
const locH = 5
const locW = 5
const inH, inW = 224, 224
const outH = inH ÷ 32
const outW = inW ÷ 32
const gridH, gridW = inH ÷ outH, inW ÷ outW
const Human = Dict{Int,Array{Float32,1}}

py"""
from chainercv.utils import non_maximum_suppression
import numpy as np

ROOT_NODE = 0


def get_instances(delta, bbox, detection_thresh=0.125):
    root_bbox = bbox[ROOT_NODE]
    score = delta[ROOT_NODE]
    candidate = np.where(score > detection_thresh)
    score = score[candidate]
    root_bbox = root_bbox[candidate]
    selected = non_maximum_suppression(
        bbox=root_bbox,
        thresh=0.3,
        score=score,
    )
    return candidate[0][selected], candidate[1][selected]
"""

"""
implement NMS method using Julia. (Not perfect)
We reinterpret Python implementation which is taken from ChainerCV project
"""
function non_maximum_suppression(; bbox, thresh, score)
    length(bbox) == 0 && return [CartesianIndex(1, 1)]
    order = sortperm(score)
    bbox = reshape(bbox, length(bbox) ÷ 4, 4)
    bbox = bbox[order, :]
    # define `bbox_area` by (ymax-ymin)*(xmax-xmin)
    bbox_area = prod(bbox[:, 3:4] .- bbox[:, 1:2], dims = 2)
    selec = zeros(Bool, size(bbox)[1])
    zs = zeros(Bool, size(bbox)[1])
    for i = 1:size(bbox)[1]
        b = bbox[i, :]
        if selec != zs
            tl = max.(reshape(b[1:2], (1, 2)), bbox[selec, 1:2])
            br = min.(reshape(b[3:4], (1, 2)), bbox[selec, 3:4])
            area = prod(br .- tl, dims = 1) .* all(tl .< br, dims = 1)
            iou = @. area / (bbox_area[i] + bbox_area[selec] - area)
            any(iou .>= thresh) && continue
        end
        selec[i] = true
    end
    return order[selec]
end

function get_instances(delta, bbox, detection_thresh = 0.125)
    root_bbox = @view bbox[1, :, :, :]
    score = @view delta[1, :, :]
    candidate = score .> detection_thresh
    isempty(findall(isequal(true), candidate)) && return [], []
    score = score[candidate]
    root_bbox = root_bbox[cat(
        candidate,
        candidate,
        candidate,
        candidate,
        dims = 3,
    )]
    selected = non_maximum_suppression(
        bbox = root_bbox,
        thresh = 0.3,
        score = score,
    )
    inds = findall(isequal(true), candidate)[selected]
    return [i[1] for i in inds], [i[2] for i in inds]
end


function reversedims(a::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    permutedims(a, N:-1:1)
end


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


function estimate(feature)
    delta, x, y, w, h, e = extract_feature(feature)
    ry, rx = restore_xy(y, x)
    rh, rw = restore_size(h, w)
    ymin, ymax = @. ry - rh / 2, ry + rh / 2
    xmin, xmax = @. rx - rw / 2, rx + rw / 2
    bbox = cat(ymin, xmin, ymax, xmax, dims = 4)
    e = permutedims(e, (1, 4, 5, 2, 3))
    #hs, ws = get_instances(delta, bbox)

    hs, ws = py"get_instances"(delta, bbox)
    hs .+= 1
    ws .+= 1

    humans = Human[]
    for hxw in zip(hs, ws)
        human = Dict(1 => bbox[1, hxw[1], hxw[2], :])
        for gi in [1, 2, 3, 4, 5, 6]
            (i_h, i_w) = hxw
            eis, ts = DIRECTED_GRAPHS[gi, 1, :], DIRECTED_GRAPHS[gi, 2, :]
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
