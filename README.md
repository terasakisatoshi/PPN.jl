# PPN.jl

- This a Deep Learning based Pose Estimation Application using Julia.
- We use pre-trained Chainer model provided by [idein/chainer-pose-proposal-net](https://github.com/Idein/chainer-pose-proposal-net), convert it to Flux via [Gomah.jl](https://github.com/terasakisatoshi/Gomah.jl) and finally create application that utilizes it.

# How to use

- In this chapter, you'll learn how to use our software.
- We will assume you are familiar with Python, PyCall and Flux.


## Install Chainer and ChainerCV

- Chainer is a deep learning framework that lets researchers quickly implement, train and evaluate deep learing models.
- ChainerCV is a library for deep learning in computer vision.

```
$ pip install chainer chainercv
```

## Install our application

- Clone this repository and go to `PPN.jl/src` to download pre-trained model:

```
$ cd path/to/your/workspace
$ git clone https://github.com/terasakisatoshi/PPN.jl.git
$ cd PPN.jl/src
$ wget https://github.com/Idein/chainer-pose-proposal-net/releases/download/v1.0/result.zip
$ unzip result.zip
```

## Install Gomah.jl

- After installing PPN.jl and pre-trained model, Do the following procedure to add dependency `Gomah.jl` which is not official Julia package.

```
your-terminal$ cd path/to/your/workspace/PPN.jl
your-terminal$ julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.2.0 (2019-08-20)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia> ] # type ] to switch pkg mode
(v1.2) pkg> activate .
(PPN) pkg> add https://github.com/terasakisatoshi/Gomah.jl.git
(PPN) pkg> # press delete to back to repl mode
julia> exit() # Done!
```

## Run our demo

### Quick start

- Do the following procedure.

```
your-terminal$ cd path/to/your/workspace/PPN.jl
your-terminal$ julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.2.0 (2019-08-20)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia> ] # type ] to switch pkg mode
(v1.2) pkg> activate .
(PPN) pkg> # press delete to back to repl mode
julia> include("demo/demo.jl")
julia> democamera()
```

- It will open web camera on your PC using VideoIO.jl and visualize result via Makie.jl


# License
