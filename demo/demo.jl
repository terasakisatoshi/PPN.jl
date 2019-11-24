using PPN
using PPN: pypose_utils

using Images
using ImageDraw
using PyCall
using VideoIO
using Makie

include("estimator.jl")
include("drawer.jl")
include("imgutils.jl")
const flux_model = PPN.PoseProposalNet(pyppn.load())


function inference(img)
    inp = reshape(img, 1, size(img)...)
    feature = reversedims(flux_model(reversedims(inp)))
    humans = estimate(feature)
    imgshow = colorview(RGB{N0f8}, N0f8.(img / 256f0))
    draw_humans(imgshow, humans)
    imgshow
end


function play(f; flipx = false, flipy = false, pixelaspectratio = nothing)
    if pixelaspectratio ≡ nothing # if user did not specify the aspect ratio we'll try to use the one stored in the video file
        pixelaspectratio = VideoIO.aspect_ratio(f)
    end
    sz = min(f.height, f.width)
    scene = Makie.Scene(resolution = (sz, sz))
    buf = read(f)
    disp = center_crop(buf)
    dispW, dispH = size(disp)
    makieimg = Makie.image!(
        scene,
        1:sz,
        1:sz,
        disp,
        show_axis = false,
        scale_plot = false,
    )[end]
    Makie.rotate!(scene, -π / 2)
    if flipx && flipy
        Makie.scale!(scene, -1, -1, 1)
    else
        flipx && Makie.scale!(scene, -1, 1, 1)
        flipy && Makie.scale!(scene, 1, -1, 1)
    end
    display(scene)
    while !eof(f) && isopen(scene)
        etime = @elapsed begin
            read!(f, buf)
            disp = center_crop(buf)
            etime = @elapsed begin
                read!(f, buf)
                img = imresize(center_crop(buf), inW, inH)
                imgshow = img |> channelview |> rawview .|> Float32 |> inference
                makieimg[3] = imresize(imgshow, dispW, dispH)
            end
        end
        @info "FPS $(1/etime)"
        sleep(1 / f.framerate)
    end
end

"""
Load image and pass it to network as input.
"""
function demoimg(imgfile)
    img = loadjlimg(imgfile)
    imgshow = inference(img)
    save("result.jpeg", imgshow)
    return imgshow
end

"""
Open camera to capture image.
It will be passed to network as input.
"""
function democamera(pixelaspectratio = nothing)
    camera = opencamera()
    try
        play(camera, flipx = true, pixelaspectratio = pixelaspectratio)
    catch e
        @show e
        close(camera)
    end
end
