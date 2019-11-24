using PPN
using Test
import Gomah: chainer, np
import Gomah: ch2dwconv, ch2conv, reversedims
using PyCall

const chmodel = PyNULL()
@testset "PPN.jl" begin
    # Write your own tests here.
    copy!(chmodel, pyppn.load())
    @test chmodel != PyNULL()
end

const chmv2 = chmodel.feature_layer
const ImageH = 224
const ImageW = 224

@testset "conv0" begin
    # Write your own tests here.
    BSIZE = 1
    INCH = 3
    inH = ImageH
    inW = ImageW
    chconv0 = chmv2.conv0
    flconv0 = PPN.Convolution2d(chconv0)
    dummyX = 128 * np.ones((BSIZE, INCH, inH, inW), dtype = np.float32)
    @pywith chainer.using_config("train", false) begin
        chret = reversedims(chconv0(dummyX).array)
        flret = flconv0(reversedims(dummyX))
        @test all(isapprox.(abs.(flret - chret), 0.0, atol = 1e-4))
    end
end

@testset "conv1" begin
    # Write your own tests here.
    BSIZE = 1
    INCH = 32
    inH = ImageH ÷ 2
    inW = ImageH ÷ 2
    chconv1 = chmv2.conv1
    flconv1 = PPN.ExpandedConvRatio1(chconv1)
    dummyX = 128 * np.ones((BSIZE, INCH, inH, inW), dtype = np.float32)
    @pywith chainer.using_config("train", false) begin
        chret = reversedims(chconv1(dummyX).array)
        flret = flconv1(reversedims(dummyX))
        @test all(isapprox.(abs.(flret - chret), 0.0, atol = 1e-4))
    end
end


@testset "conv2:17" begin
    # Write your own tests here.
    BSIZE = 1
    INCH = 16
    inH = ImageH ÷ 2
    inW = ImageH ÷ 2
    for i ∈ 2:17
        @testset "conv$i" begin
            chlayer = @eval chmv2.$(Symbol("conv$i"))
            fllayer = PPN.ExpandedConv(chlayer)
            dummyX = 128 * np.ones((BSIZE, INCH, inH, inW), dtype = np.float32)
            @pywith chainer.using_config("train", false) begin
                chret = reversedims(chlayer(dummyX).array)
                inW, inH, INCH, BSIZE = size(chret)
                flret = fllayer(reversedims(dummyX))
                @test all(isapprox.(abs.(flret - chret), 0.0, atol = 1e-4))
            end
        end
    end
end

@testset "lastconv" begin
    BSIZE = 1
    INCH = 320
    inH = 7
    inW = 7
    chlastconv = chmodel.lastconv
    fllastconv = ch2conv(chlastconv)
    dummyX = 128 * np.ones((BSIZE, INCH, inH, inW), dtype = np.float32)
    @pywith chainer.using_config("train", false) begin
        chret = reversedims(chlastconv(dummyX).array)
        flret = fllastconv(reversedims(dummyX))
        @test all(isapprox.(abs.(flret - chret), 0.0, atol = 1e-4))
    end
end


@testset "InitPPN" begin
    @test PPN.PoseProposalNet == typeof(PPN.PoseProposalNet(pyppn.load()))
end

@testset "convertPPN" begin
    BSIZE = 1
    INCH = 3
    inH = ImageH
    inW = ImageW
    dummyX = 128 * np.ones((BSIZE, INCH, inH, inW), dtype = np.float32)
    flmodel = PPN.PoseProposalNet(chmodel)
    @pywith chainer.using_config("train", false) begin
        chret = reversedims(chmodel(dummyX).array)
        flret = flmodel(reversedims(dummyX))
        @test all(isapprox.(abs.(flret - chret), 0.0, atol = 1e-4))
    end
end
