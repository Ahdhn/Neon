#pragma once

void MultiResParent()
{
    using Type = int32_t;
    const int              nGPUs = 1;
    const Neon::int32_3d   dim(24, 24, 24);
    const std::vector<int> gpusIds(nGPUs, 0);

    const Neon::domain::internal::bGrid::bGridDescriptor descriptor({1, 1, 1});

    for (auto runtime : {Neon::Runtime::stream, Neon::Runtime::openmp}) {

        auto bk = Neon::Backend(gpusIds, runtime);

        Neon::domain::bGrid BGrid(
            bk,
            dim,
            {[&](const Neon::index_3d id) -> bool {
                 return id.x < 8;
             },
             [&](const Neon::index_3d& id) -> bool {
                 return id.x >= 8 && id.x < 16;
             },
             [&](const Neon::index_3d& id) -> bool {
                 return id.x >= 16;
             }},
            Neon::domain::Stencil::s7_Laplace_t(),
            descriptor);
        //BGrid.topologyToVTK("bGrid111.vtk", false);

        auto XField = BGrid.newField<Type>("XField", 1, -1);
        auto hasParentField = BGrid.newField<Type>("hasParent", 1, -1);

        //Init fields
        for (int l = 0; l < descriptor.getDepth(); ++l) {
            XField.forEachActiveCell(
                l,
                [&](const Neon::int32_3d, const int, Type& val) {
                    val = l;
                });
            hasParentField.forEachActiveCell(
                l,
                [&](const Neon::int32_3d, const int, Type& val) {
                    val = -1;
                });
        }

        if (bk.runtime() == Neon::Runtime::stream) {
            XField.updateCompute();
            hasParentField.updateCompute();
        }
        //XField.ioToVtk("f", "f");


        for (int level = 0; level < descriptor.getDepth(); ++level) {
            XField.setCurrentLevel(level);
            hasParentField.setCurrentLevel(level);
            BGrid.setCurrentLevel(level);

            auto container = BGrid.getContainer(
                "Parent", [&, level](Neon::set::Loader& loader) {
                    auto& xLocal = loader.load(XField);
                    auto& hasParentLocal = loader.load(hasParentField);

                    return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {
                        if (xLocal.hasParent(cell)) {
                            hasParentLocal(cell, 0) = 1;
                            xLocal(cell, 0) = xLocal.parent(cell, 0);
                        }
                    };
                });

            container.run(0);
            BGrid.getBackend().syncAll();
        }

        if (bk.runtime() == Neon::Runtime::stream) {
            XField.updateIO();
            hasParentField.updateIO();
        }


        //verify
        for (int l = 0; l < descriptor.getDepth(); ++l) {
            hasParentField.forEachActiveCell(
                l,
                [&](const Neon::int32_3d, const int, Type& val) {
                    if (l != descriptor.getDepth() - 1) {
                        EXPECT_EQ(val, 1);
                    } else {
                        EXPECT_EQ(val, -1);
                    }
                });


            XField.forEachActiveCell(
                l,
                [&](const Neon::int32_3d, const int, Type& val) {
                    if (l != descriptor.getDepth() - 1) {
                        EXPECT_EQ(val, l + 1);
                    } else {
                        EXPECT_EQ(val, l);
                    }
                });
        }
    }
}
TEST(MultiRes, Parent)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        MultiResParent();
    }
}