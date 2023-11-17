#ifndef HILBERT_SORT__HILBERT_H__INCLUDED
#define HILBERT_SORT__HILBERT_H__INCLUDED

#include <algorithm>
#include <array>
#include <cstdlib>

#include <morton.h>

namespace hilbert
{

    namespace detail
    {
        template <typename T>
        struct dependent_false : std::false_type
        {
        };

        template <class CoordT, std::size_t NDim>
        struct MortonEncoder
        {
            static_assert(dependent_false<CoordT>::value, "Only up to 3 dimensions are supported");
        };

        template <class CoordT>
        struct MortonEncoder<CoordT, 2>
        {
            uint64_t operator()(std::array<CoordT, 2> point)
            {
                return libmorton::morton2D_64_encode(std::get<1>(point), std::get<0>(point));
            }
        };

        template <class CoordT>
        struct MortonEncoder<CoordT, 3>
        {
            uint64_t operator()(std::array<CoordT, 3> point)
            {
                return libmorton::morton3D_64_encode(std::get<2>(point), std::get<1>(point),
                                                     std::get<0>(point));
            }
        };

    }  // namespace detail

    template <class PointT, size_t NDim>
    static uint64_t transpose_to_hilbert_integer(PointT point)
    {
        return detail::MortonEncoder<typename PointT::value_type, NDim>{}(point);
    }

    template <class PointT, std::size_t HilbertOrder, std::size_t NDim>
    constexpr inline uint64_t hilbert_distance_by_coords(PointT point)
    {
        using coord_t = typename PointT::value_type;

        static_assert(NDim < 4, "we support up to 3 dimensions only");
        static_assert(NDim * HilbertOrder < 64, "we cannot compute this hilbert value");
        static_assert(std::is_integral<coord_t>::value, "we support only integral types");

        std::for_each(std::begin(point), std::end(point),
                      [](const coord_t val) { assert(!(val > ((1 << HilbertOrder) - 1))); });

        constexpr const std::size_t M = 1 << (HilbertOrder - 1);

        // Inverse Undo Excess work
        for (auto Q = M; Q > 1; Q >>= 1)
        {
            auto P = Q - 1;
            for (std::size_t i = 0; i < NDim; ++i)
            {
                if (point[i] & Q)
                {
                    point[0] ^= P;
                }
                else
                {
                    auto t = (point[0] ^ point[i]) & P;
                    point[0] ^= t;
                    point[i] ^= t;
                }
            }
        }

        // Gray encode
        for (std::size_t i = 1; i < NDim; ++i)
        {
            point[i] ^= point[i - 1];
        }

        coord_t t{0};

        for (auto Q = M; Q > 1; Q >>= 1)
        {
            if (point[NDim - 1] & Q)
            {
                t ^= Q - 1;
            }
        }

        for (std::size_t i = 0; i < NDim; ++i)
        {
            point[i] ^= t;
        }

        // the hilbert integer can be computed through the Morton Value
        return transpose_to_hilbert_integer<PointT, NDim>(point);
    }
}  // namespace hilbert

#endif
