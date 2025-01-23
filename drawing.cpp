#include "drawing.h"
#include <opencv2/opencv.hpp>
using namespace cv;

namespace DRAWING {

static inline void ICV_HLINE_X(uchar* ptr, int64_t xl, int64_t xr, const uchar* color, int pix_size)
{
    uchar* hline_min_ptr = (uchar*)(ptr)+(xl) * (pix_size);
    uchar* hline_end_ptr = (uchar*)(ptr)+(xr + 1) * (pix_size);
    uchar* hline_ptr = hline_min_ptr;
    if (pix_size == 1)
        memset(hline_min_ptr, *color, hline_end_ptr - hline_min_ptr);
    else//if (pix_size != 1)
    {
        if (hline_min_ptr < hline_end_ptr)
        {
            memcpy(hline_ptr, color, pix_size);
            hline_ptr += pix_size;
        }//end if (hline_min_ptr < hline_end_ptr)
        size_t sizeToCopy = pix_size;
        while (hline_ptr < hline_end_ptr)
        {
            memcpy(hline_ptr, hline_min_ptr, sizeToCopy);
            hline_ptr += sizeToCopy;
            sizeToCopy = std::min(2 * sizeToCopy, static_cast<size_t>(hline_end_ptr - hline_ptr));
        }//end while(hline_ptr < hline_end_ptr)
    }//end if (pix_size != 1)
}
//end ICV_HLINE_X()

static inline void ICV_HLINE(uchar* ptr, int64_t xl, int64_t xr, const void* color, int pix_size)
{
    ICV_HLINE_X(ptr, xl, xr, reinterpret_cast<const uchar*>(color), pix_size);
}
//end ICV_HLINE()

static void
CollectPolyEdges(Mat& img, const Point2l* v, int count, std::vector<PolyEdge>& edges,
    const void* color, int line_type, int shift, Point offset)
{
    int i, delta = offset.y + ((1 << shift) >> 1);
    Point2l pt0 = v[count - 1], pt1;
    pt0.x = (pt0.x + offset.x) << (XY_SHIFT - shift);
    pt0.y = (pt0.y + delta) >> shift;

    edges.reserve(edges.size() + count);

    for (i = 0; i < count; i++, pt0 = pt1)
    {
        Point2l t0, t1;
        PolyEdge edge;

        pt1 = v[i];
        pt1.x = (pt1.x + offset.x) << (XY_SHIFT - shift);
        pt1.y = (pt1.y + delta) >> shift;

        if (line_type < 8)
        {
            t0.y = pt0.y; t1.y = pt1.y;
            t0.x = (pt0.x + (XY_ONE >> 1)) >> XY_SHIFT;
            t1.x = (pt1.x + (XY_ONE >> 1)) >> XY_SHIFT;
            //cv::Line(img, t0, t1, color, line_type);
        }
        else
        {
            t0.x = pt0.x; t1.x = pt1.x;
            t0.y = pt0.y << XY_SHIFT;
            t1.y = pt1.y << XY_SHIFT;
            //LineAA(img, t0, t1, color);
        }

        if (pt0.y == pt1.y)
            continue;

        if (pt0.y < pt1.y)
        {
            edge.y0 = (int)(pt0.y);
            edge.y1 = (int)(pt1.y);
            edge.x = pt0.x;
        }
        else
        {
            edge.y0 = (int)(pt1.y);
            edge.y1 = (int)(pt0.y);
            edge.x = pt1.x;
        }
        edge.dx = (pt1.x - pt0.x) / (pt1.y - pt0.y);
        edges.push_back(edge);
    }
}

struct CmpEdges
{
    bool operator ()(const PolyEdge& e1, const PolyEdge& e2)
    {
        return e1.y0 - e2.y0 ? e1.y0 < e2.y0 :
            e1.x - e2.x ? e1.x < e2.x : e1.dx < e2.dx;
    }
};

void printLink(PolyEdge* next)
{
    if (next) {
        std::cout << " tmp->next: " << next;
        if (next->next)
        {
            std::cout << " tmp->next->next: " << next->next;
            if (next->next->next) {
                std::cout << " tmp->next->next->next: " << next->next->next;
                if (next->next->next->next) {
                    std::cout << " tmp->next->next->next->next: " << next->next->next->next;
                }
            }
        }
    }
    std::cout << std::endl;
}

/**************** helper macros and functions for sequence/contour processing ***********/

static void
FillEdgeCollection(Mat& img, std::vector<PolyEdge>& edges, const void* color)
{
    PolyEdge tmp;
    int i, y, total = (int)edges.size();
    Size size = img.size();
    PolyEdge* e;
    int y_max = INT_MIN, y_min = INT_MAX;
    int64 x_max = 0xFFFFFFFFFFFFFFFF, x_min = 0x7FFFFFFFFFFFFFFF;
    int pix_size = (int)img.elemSize();

    if (total < 2)
        return;

    for (i = 0; i < total; i++)
    {
        PolyEdge& e1 = edges[i];
        CV_Assert(e1.y0 < e1.y1);
        // Determine x-coordinate of the end of the edge.
        // (This is not necessary x-coordinate of any vertex in the array.)
        int64 x1 = e1.x + (e1.y1 - e1.y0) * e1.dx;
        y_min = std::min(y_min, e1.y0);
        y_max = std::max(y_max, e1.y1);
        x_min = std::min(x_min, e1.x);
        x_max = std::max(x_max, e1.x);
        x_min = std::min(x_min, x1);
        x_max = std::max(x_max, x1);
    }

    if (y_max < 0 || y_min >= size.height || x_max < 0 || x_min >= ((int64)size.width << XY_SHIFT))
        return;

    std::sort(edges.begin(), edges.end(), CmpEdges());

    // start drawing
    tmp.y0 = INT_MAX;
    edges.push_back(tmp); // after this point we do not add
    // any elements to edges, thus we can use pointers
    i = 0;
    tmp.next = 0;
    e = &edges[i];
    y_max = MIN(y_max, size.height);

    std::map<long, int>edgeId;
    int id = 0;
    for (const auto& edge : edges)
    {
        std::cout << "edge:x " << (edge.x >> XY_SHIFT) << " y0: " << edge.y0 << " " << edge.y1 << std::endl;
        std::cout << "addr: " << &edge << std::endl;
        edgeId[(long) &edge] = ++id;
    }

    for (y = e->y0; y < y_max; y++)
    {
        PolyEdge* last, * prelast, * keep_prelast=0;
        int draw = 0;
        int clipline = y < 0;


        prelast = &tmp;
        last = tmp.next;

        std::cout << "\nEach loop tmp: " << prelast << std::endl;

        while (last || e->y0 == y)
        {
            std::cout << "tempnext000: " << tmp.next << std::endl;
            if (last) std::cout << "last00: " << edgeId[(long)last] << " last->next: " << edgeId[(long)last->next] << " eidx: " << i + 1 << std::endl;
            std::cout << "[Before]Each loop: keep_prelast= " << edgeId[(long)keep_prelast] << " prelast: " << edgeId[(long)prelast] << " last: " << edgeId[(long)last] << " prelast addr: "<<prelast<<std::endl;
            if (last && last->y1 == y)
            {
                // exclude edge if y reaches its lower point
                prelast->next = last->next;//tmp.next右移，更新
                last = last->next;
                std::cout << "Update last: " << last << std::endl;
                continue;
            }
            keep_prelast = prelast;
            if (last && (e->y0 > y || last->x < e->x))
            {
                // go to the next edge in active list
                prelast = last;
                last = last->next;
                std::cout << "[Active]keep_prelast= " << edgeId[(long)keep_prelast] << " prelast: " << edgeId[(long)prelast] << " last: " << edgeId[(long)last] << std::endl;
                std::cout << "i and total: " << i << " " << total << std::endl;
                if (last) std::cout << "last11: " << edgeId[(long)last] << " last->next: " << edgeId[(long)last->next] << " eidx: " << i + 1 << std::endl;
            }
            else if (i < total)
            {
                // insert new edge into active list if y reaches its upper point
                //if (tmp.next) std::cout << "===================tmpnex1: " << tmp.next << " " << tmp.next->next << "e: "<<e<<std::endl;
                std::cout << "===================tmpnex1: ";
                printLink(tmp.next);
                std::cout << "prelast: " << prelast << std::endl;
                prelast->next = e;//tmp.next=e同时更新,没更改时prelast=tmp,prelast->next=tmp.next
                std::cout << "last: " << last << "tmpnext: "<<tmp.next<<std::endl;
                std::cout << "===================tmpnex2: ";
                printLink(tmp.next);// if (tmp.next)std::cout << "===================tmpnex2: " << tmp.next << " " << tmp.next->next << " e: " << e << std::endl;

                e->next = last;
                std::cout << "===================tmpnex3: ";
                printLink(tmp.next);// if (tmp.next) std::cout << "===================tmpnex3: " << tmp.next << " " << tmp.next->next << std::endl;

                prelast = e;
                std::cout << "===================tmpnex4: ";
                printLink(tmp.next);// if (tmp.next)std::cout << "===================tmpnex4: " << tmp.next << " " << tmp.next->next << std::endl;

                e = &edges[++i];
                std::cout << "[Insert]Each loop: keep_prelast= " << edgeId[(long)keep_prelast] << " prelast: " << edgeId[(long)prelast] << " last: " << edgeId[(long)last] << " Eid: "<<i+1<<std::endl;
                if (last) std::cout << "last22: " << edgeId[(long)last] << " last->next: " << edgeId[(long)last->next] << " eidx: " << i + 1 << std::endl;

            }
            else
                break;

            std::cout << "===================tmpnext end: ";
            printLink(tmp.next);

            std::cout << "[Process]Each loop: keep_prelast= " << edgeId[(long)keep_prelast] << " prelast: " << edgeId[(long)prelast] << " last: " << edgeId[(long)last] << std::endl;
            if (last) std::cout << "last33: " << edgeId[(long)last] << " last->next: " << edgeId[(long)last->next] << " eidx: " << i + 1 << std::endl;


            if (draw)
            {
                if (!clipline)
                {
                    // convert x's from fixed-point to image coordinates
                    uchar* timg = img.ptr(y);
                    int x1, x2;

                    if (keep_prelast->x > prelast->x)
                    {
                        x1 = (int)((prelast->x + XY_ONE - 1) >> XY_SHIFT);
                        x2 = (int)(keep_prelast->x >> XY_SHIFT);
                    }
                    else
                    {
                        x1 = (int)((keep_prelast->x + XY_ONE - 1) >> XY_SHIFT);
                        x2 = (int)(prelast->x >> XY_SHIFT);
                    }

                    // clip and draw the line
                    if (x1 < size.width && x2 >= 0)
                    {
                        if (x1 < 0)
                            x1 = 0;
                        if (x2 >= size.width)
                            x2 = size.width - 1;
                        ICV_HLINE(timg, x1, x2, color, pix_size);
                        std::cout << "[draw line]keep_prelast: " << edgeId[(long)keep_prelast] << " prelast: " << edgeId[(long)prelast] << " last: " << edgeId[(long)last] << std::endl;
                    }
                }
                keep_prelast->x += keep_prelast->dx;
                prelast->x += prelast->dx;
            }

            std::cout << "y: " << y << std::endl;
            std::cout << "[After]Each loop: keep_prelast= " << edgeId[(long)keep_prelast] << " prelast: " << edgeId[(long)prelast] << " last: " << edgeId[(long)last] << std::endl;

            draw ^= 1;
        }

        // sort edges (using bubble sort)
        /*keep_prelast = 0;
        std::cout << "tmp0: " << tmp.next << std::endl;
        do
        {
            prelast = &tmp;
            last = tmp.next;
            PolyEdge* last_exchange = 0;

            while (last != keep_prelast && last->next != 0)
            {
                PolyEdge* te = last->next;

                // swap edges
                if (last->x > te->x)
                {
                    prelast->next = te;
                    last->next = te->next;
                    te->next = last;
                    prelast = te;
                    last_exchange = prelast;
                }
                else
                {
                    prelast = last;
                    last = te;
                }
            }
            if (last_exchange == NULL)
                break;
            keep_prelast = last_exchange;
        } while (keep_prelast != tmp.next && keep_prelast != &tmp);*/
        std::cout << "tmp1: " << tmp.next << std::endl;
        std::cout << "after sorting: " << edgeId[long(keep_prelast)] << " prelast: " << edgeId[(long)prelast] << " lst: " << edgeId[long(last)] << std::endl; getchar();
    }
}

template <typename T> static inline
void scalarToRawData(const Scalar& s, T* const buf, const int cn, const int unroll_to)
{
    int i = 0;
    for (; i < cn; i++)
        buf[i] = saturate_cast<T>(s.val[i]);
    for (; i < unroll_to; i++)
        buf[i] = buf[i - cn];
}

void drawingRun()
{
    std::vector<Point2l>pts = {
        {5,1},{11,3},{11,8},{5,5},{2,7},{2,2}
    };

    for (auto& pt : pts) pt = pt + Point2l{10, 10};

    Scalar color{4};
    double buf[4];

    cv::Mat img(30, 30, 0, Scalar(0));
    scalarToRawData(color, buf, img.type(), 0);

    std::vector<PolyEdge> edges;
    int i, total = 0;
    edges.reserve(total + 1);
    
    CollectPolyEdges(img, pts.data(), pts.size(), edges, buf, 4, 0, Point(0, 0));

    FillEdgeCollection(img, edges, buf);

    std::cout << img << std::endl;
}

}
