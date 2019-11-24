function loadjlimg(imgfile)
    img = imgfile |> Images.load
    _resized = imresize(img, 224, 224)
    ret = _resized |> channelview |> rawview .|> Float32
    ret[1:3,:,:]
end

function loadchcvimg(imgfile)
    chainercv = pyimport("chainercv")
    chimg = pycall(
        chainercv.utils.read_image,
        PyObject,
        imgfile,
        dtype = :float32,
    )
    ret = chainercv.transforms.resize(chimg, (224, 224))
    return ret
end

function center_crop(img)
    imW, imH = size(img)
    sz = min(imH, imW)
    y_offset = round(Int, (imH - sz) / 2.0)
    x_offset = round(Int, (imW - sz) / 2.0)
    y_slice = range(1 + y_offset, stop = y_offset + sz)
    x_slice = range(1 + x_offset, stop = x_offset + sz)
    cropped = img[x_slice, y_slice]
    cropped
end
