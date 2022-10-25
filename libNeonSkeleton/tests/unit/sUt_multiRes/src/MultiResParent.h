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


void MultiResAtomicAddParent()
{
    using Type = int32_t;
    const int              nGPUs = 1;
    const Neon::int32_3d   dim(24, 24, 24);
    const std::vector<int> gpusIds(nGPUs, 0);

    const Neon::domain::internal::bGrid::bGridDescriptor descriptor({1, 1, 1});

    for (auto runtime : {
             Neon::Runtime::openmp,
             Neon::Runtime::stream}) {

        auto bk = Neon::Backend(gpusIds, runtime);

        int SectionX[3];
        SectionX[0] = 8;
        SectionX[1] = 16;
        SectionX[2] = 24;

        Neon::domain::bGrid BGrid(
            bk,
            dim,
            {[&](const Neon::index_3d id) -> bool {
                 return id.x < SectionX[0];
             },
             [&](const Neon::index_3d& id) -> bool {
                 return id.x >= SectionX[0] && id.x < SectionX[1];
             },
             [&](const Neon::index_3d& id) -> bool {
                 return id.x >= SectionX[1] && id.x < SectionX[2];
             }},
            Neon::domain::Stencil::s7_Laplace_t(),
            descriptor);
        //BGrid.topologyToVTK("bGrid111.vtk", false);

        auto XField = BGrid.newField<Type>("XField", 1, -1);


        //Init fields
        for (int l = 0; l < descriptor.getDepth(); ++l) {
            XField.forEachActiveCell(
                l,
                [&](const Neon::int32_3d, const int, Type& val) {
                    val = l + 1;
                });
        }

        if (bk.runtime() == Neon::Runtime::stream) {
            XField.updateCompute();
        }
        //XField.ioToVtk("f", "f");


        for (int level = 0; level < descriptor.getDepth(); ++level) {
            XField.setCurrentLevel(level);
            BGrid.setCurrentLevel(level);

            auto container = BGrid.getContainer(
                "Parent", [&, level](Neon::set::Loader& loader) {
                    auto& xLocal = loader.load(XField);

                    return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {
                        if (xLocal.hasParent(cell)) {

#ifdef NEON_PLACE_CUDA_DEVICE
                            atomicAdd(&xLocal.parent(cell, 0), xLocal(cell, 0));
#else

#pragma omp atomic
                            xLocal.parent(cell, 0) += xLocal(cell, 0);
#endif
                        }
                    };
                });

            container.run(0);
            BGrid.getBackend().syncAll();
        }

        if (bk.runtime() == Neon::Runtime::stream) {
            XField.updateIO();
        }


        //verify
        for (int l = 0; l < descriptor.getDepth(); ++l) {

            XField.forEachActiveCell(
                l,
                [&](const Neon::int32_3d id, const int, Type& val) {
                    if (l == 0) {
                        //the lowest/most refined level won't chance since it does not
                        //have children to write into it
                        EXPECT_EQ(val, l + 1);
                    } else {
                        //only the part of this level that overlaps with Level 0 will change
                        //otherwise, it will stay the same since there is no children to write to it
                        if (id.x < SectionX[l - 1]) {
                            EXPECT_EQ(val, (l + 1) +  //init value on this level
                                                      //value added by all children
                                               (l)*descriptor.getRefFactor(l) * descriptor.getRefFactor(l) * descriptor.getRefFactor(l));
                        } else {
                            EXPECT_EQ(val, l + 1);
                        }
                    }
                });
        }
    }
}
TEST(MultiRes, AtomicAddParent)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        MultiResAtomicAddParent();
    }
}