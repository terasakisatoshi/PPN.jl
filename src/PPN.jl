module PPN

using PyCall
using Gomah
import Flux

export pyppn, pypose_utils
# import python files located in the current directory
const pyppn = PyNULL()
const pypose_utils = PyNULL()

function __init__()
    pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__)
    copy!(pyppn, pyimport("pyppn"))
    copy!(pypose_utils, pyimport("pyppn.pose_utils"))
end

include("network_mv2.jl")
include("network_ppn.jl")

end # module
