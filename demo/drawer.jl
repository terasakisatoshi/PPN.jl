function draw_bbox(img, ymin, xmin, ymax, xmax)
    vert = [
        CartesianIndex(ymin, xmin),
        CartesianIndex(ymin, xmax),
        CartesianIndex(ymax, xmax),
        CartesianIndex(ymax, xmin),
    ]
    draw!(img, Polygon(vert))
end

draw_bbox(img, bbox::NTuple{4,Int}) = draw_bbox(img, bbox...)

function draw_humans(img, humans::Vector{Human})
    for human in humans
        r = 5
        for (k, bbox) in human
            ymin, xmin, ymax, xmax = floor.(Int,bbox)
            if k == 1
                draw_bbox(img, ymin, xmin, ymax, xmax)
            else
                x = (xmin + xmax) ÷ 2
                y = (ymin + ymax) ÷ 2
                draw!(img, CirclePointRadius(ImageDraw.Point(x, y), r), color_map(k))
            end
            for (s, t) in zip(EDGES[:, 1], EDGES[:, 2])
                if s in keys(human) && t in keys(human)
                    by = (human[s][1] + human[s][3]) ÷ 2 |> floor |> Int
                    bx = (human[s][2] + human[s][4]) ÷ 2 |> floor |> Int
                    ey = (human[t][1] + human[t][3]) ÷ 2 |> floor |> Int
                    ex = (human[t][2] + human[t][4]) ÷ 2 |> floor |> Int
                    draw!(img, LineSegment(bx, by, ex, ey), color_map(s))
                end
            end
        end
    end
end
