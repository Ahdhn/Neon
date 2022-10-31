#pragma once

#include "type_traits"

#include "Neon/set/Containter.h"

namespace Neon::set {
namespace internal {

namespace tmp {
// From:
// https://en.cppreference.com/w/cpp/experimental/is_detected
namespace detail {
template <class Default, class AlwaysVoid, template <class...> class Op, class... Args>
struct detector
{
    using value_t = std::false_type;
    using type = Default;
};

template <class Default, template <class...> class Op, class... Args>
struct detector<Default, std::void_t<Op<Args...>>, Op, Args...>
{
    using value_t = std::true_type;
    using type = Op<Args...>;
};

}  // namespace detail

struct nonesuch
{
    ~nonesuch() = delete;
    nonesuch(nonesuch const&) = delete;
    void operator=(nonesuch const&) = delete;
};

template <template <class...> class Op, class... Args>
using is_detected = typename detail::detector<nonesuch, void, Op, Args...>::value_t;

template <template <class...> class Op, class... Args>
using detected_t = typename detail::detector<nonesuch, void, Op, Args...>::type;

template <class Default, template <class...> class Op, class... Args>
using detected_or = detail::detector<Default, void, Op, Args...>;

template <template <class...> class Op, class... Args>
constexpr bool is_detected_v = is_detected<Op, Args...>::value;

}  // namespace tmp

template <typename Field_ta>
struct DataTransferExtractor
{
    // field.haloUpdate(bk, opt);
    // const Neon::set::Backend& /*bk*/,
    // Neon::set::HuOptions_t& /*opt*/
   private:
    template <typename T>
    using HaloUpdate = decltype(std::declval<T>().haloUpdate(std::declval<Neon::set::TransferMode>(),
                                                             std::declval<Neon::set::StencilSemantic>()));

    template <typename T>
    static constexpr bool HasHaloUpdateMethod = tmp::is_detected_v<HaloUpdate, T>;

   public:
    static auto get([[maybe_unused]] const Field_ta& field) -> auto
    {
        if constexpr (HasHaloUpdateMethod<Field_ta>) {
            auto huFun = [field](Neon::set::TransferMode    transferMode,
                                 Neon::set::StencilSemantic transferSemantic) {
                field.haloUpdate(transferMode, transferSemantic);
            };
            return huFun;
        } else {
            auto huFun = [field](Neon::set::TransferMode    transferMode,
                                 Neon::set::StencilSemantic transferSemantic) {
                (void)transferMode;
                (void)transferSemantic;
            };

            return huFun;
        }
        NEON_THROW_UNSUPPORTED_OPTION("");
    }
};

template <typename Field_ta>
struct HaloUpdatePerDeviceExtractor_t
{
    // field.haloUpdate(bk, opt);
    // const Neon::set::Backend& /*bk*/,
    // Neon::set::HuOptions_t& /*opt*/
   private:
    template <typename T>
    using HaloUpdatePerDevice = decltype(std::declval<T>().haloUpdate(std::declval<Neon::SetIdx&>(),
                                                                      std::declval<Neon::set::HuOptions&>()));

    template <typename T>
    static constexpr bool HasHaloUpdatePerDevice = tmp::is_detected_v<HaloUpdatePerDevice, T>;

   public:
    static auto get([[maybe_unused]] const Field_ta& field) -> auto
    {
        if constexpr (HasHaloUpdatePerDevice<Field_ta>) {
            auto huFun = [field](Neon::SetIdx          setIdx,
                                 Neon::set::HuOptions& opt) {
                field.haloUpdate(setIdx, opt);
            };
            return huFun;
        } else {
            auto huFun = [field](Neon::SetIdx          setIdx,
                                 Neon::set::HuOptions& opt) {
                (void)opt;
                (void)setIdx;
            };

            return huFun;
        }
        NEON_THROW_UNSUPPORTED_OPTION("");
    }
};

}  // namespace internal

template <typename Field_ta>
auto Loader::
    load(Field_ta&       field,
         Neon::Compute   computeE,
         StencilSemantic stencilSemantic)
        -> std::enable_if_t<!std::is_const_v<Field_ta>, typename Field_ta::Partition&>
{

    switch (m_loadingMode) {
        case Neon::set::internal::LoadingMode_e::PARSE_AND_EXTRACT_LAMBDA: {
            using namespace Neon::set::dataDependency;
            Neon::set::dataDependency::MultiXpuDataUid uid = field.getUid();
            constexpr auto                             access = Neon::set::dataDependency::AccessType::WRITE;
            Compute                                    compute = computeE;
            Token                                      token(uid, access, compute);

            if (compute == Neon::Compute::STENCIL &&
                (stencilSemantic == StencilSemantic::standard ||
                 stencilSemantic == StencilSemantic::streaming)) {
                Neon::NeonException exp("Loader");
                exp << "Loading a non const field for a stencil operation is not supported in Neon";
                NEON_THROW(exp);
            }

            m_container.addToken(token);

            return field.getPartition(m_devE, m_setIdx, m_dataView);
        }
        case Neon::set::internal::LoadingMode_e::EXTRACT_LAMBDA: {
            return field.getPartition(m_devE, m_setIdx, m_dataView);
        }
    }
    NEON_DEV_UNDER_CONSTRUCTION("");
}

auto Loader::
    computeMode() -> bool
{
    return m_loadingMode == Neon::set::internal::LoadingMode_e::EXTRACT_LAMBDA;
}

/**
 * Loading a const field
 */
template <typename Field_ta>
auto Loader::
    load(Field_ta&       field,
         Neon::Compute   computeE,
         StencilSemantic stencilSemantic)
        -> std::enable_if_t<std::is_const_v<Field_ta>, const typename Field_ta::Partition&>
{
    switch (m_loadingMode) {
        case Neon::set::internal::LoadingMode_e::PARSE_AND_EXTRACT_LAMBDA: {
            using namespace Neon::set::dataDependency;
            Neon::set::dataDependency::MultiXpuDataUid uid = field.getUid();
            constexpr auto                             access = Neon::set::dataDependency::AccessType::READ;
            Neon::Compute                              compute = computeE;
            Token                                      token(uid, access, compute);

            if (compute == Neon::Compute::STENCIL) {
                token.setDataTransferContainer(
                    [&](Neon::set::TransferMode transferMode)
                        -> Neon::set::Container {
                        // TODO: add back following line with template metaprogramming
                        // field.haloUpdate(bk, opt);
                        // https://gist.github.com/fenbf/d2cd670704b82e2ce7fd
                        auto                 huFun = internal::DataTransferExtractor<Field_ta>::get(field);
                        Neon::set::Container container = huFun(transferMode, stencilSemantic);

                        if(container.getContainerInterface().getContainerExecutionType() != ContainerExecutionType::graph){
                            NEON_THROW_UNSUPPORTED_OPERATION("Halo update Containers type should be Graph");
                        }

                        return container;
                    });
            }
            m_container.addToken(token);

            return field.getPartition(m_devE, m_setIdx, m_dataView);
        }
        case Neon::set::internal::LoadingMode_e::EXTRACT_LAMBDA: {
            return field.getPartition(m_devE, m_setIdx, m_dataView);
        }
    }
    NEON_DEV_UNDER_CONSTRUCTION("");
}

}  // namespace Neon::set