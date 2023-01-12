#include <array>

#include "Neon/Neon.h"
#include "Neon/domain/bGrid.h"

#include "polyscope/point_cloud.h"


int main(int argc, char** agrv)
{
    Neon::init();

    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        polyscope::init();

        auto             runtime = Neon::Runtime::stream;
        std::vector<int> gpu_ids{0};
        Neon::Backend    backend(gpu_ids, runtime);
        Neon::int32_3d   dim(10, 10, 10);

        Neon::domain::bGrid grid(
            backend, dim,
            [&](const Neon::index_3d id) { return true; },
            Neon::domain::Stencil::s7_Laplace_t());

        auto field = grid.newField<float>("X", 1, 0);

        std::vector<Neon::int32_3d> points;
        std::vector<Neon::float_3d> color;
        points.reserve(dim.rMul());
        color.reserve(dim.rMul());

        field.forEachActiveCell(
            [&](const Neon::int32_3d id, const int card, float) {
                points.push_back(id);
                Neon::float_3d rgb(0.9, float(id.y) / float(dim.y), 0.9);
                color.push_back(rgb);
            },
            Neon::computeMode_t::computeMode_e::seq);


        auto ps = polyscope::registerPointCloud("bGrid", points);
        
        ps->addColorQuantity("color", color);

        polyscope::show();
    }

    return 0;
}