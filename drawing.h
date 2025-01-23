#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;

namespace DRAWING
{
    enum { XY_SHIFT = 16, XY_ONE = 1 << XY_SHIFT, DRAWING_STORAGE_BLOCK = (1 << 12) - 256 };

    static const int MAX_THICKNESS = 32767;

    struct PolyEdge
    {
        PolyEdge() : y0(0), y1(0), x(0), dx(0), next(0) {}
        //PolyEdge(int _y0, int _y1, int _x, int _dx) : y0(_y0), y1(_y1), x(_x), dx(_dx) {}

        int y0, y1;
        int64 x, dx;
        PolyEdge* next;
    };

    static void
        CollectPolyEdges(Mat& img, const Point2l* v, int npts,
            std::vector<PolyEdge>& edges, const void* color, int line_type,
            int shift, Point offset = Point());

    static void
        FillEdgeCollection(Mat& img, std::vector<PolyEdge>& edges, const void* color);

    void drawingRun();
}
