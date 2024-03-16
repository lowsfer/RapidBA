//
// Created by yao on 3/12/18.
//

#include "SchurSolverPCG.h"
#include "../traits/Traits.h"
#include <boost/preprocessor/seq/for_each.hpp>

namespace rba{
namespace solver{
template <typename Traits>
std::unique_ptr<SchurSolver<Traits>> createSchurSolver(typename SchurSolver<Traits>::Type type)
{
    switch (type){
        case SchurSolver<Traits>::Type::PCG:
            return std::make_unique<SchurSolverPCG<Traits>>();
    }
    throw std::runtime_error("fatal error");
}

#define INSTANTIATE_TEMPLATES(r, data, TRAITS)\
    template std::unique_ptr<SchurSolver<TRAITS>> createSchurSolver(typename SchurSolver<TRAITS>::Type type);
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TEMPLATES, data, ALL_TRAITS)
#undef INSTANTIATE_TEMPLATES
}
}
