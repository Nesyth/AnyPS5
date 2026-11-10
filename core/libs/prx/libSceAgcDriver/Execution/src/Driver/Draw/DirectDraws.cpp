#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DirectDrawReport.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/LeanDraw.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadReport.hpp"

namespace AgcDriver::DriverDetail {

void Driver::finishDirectDraw(const LeanDrawJob& job) {
    using DirectDraws::Counter;
    const auto& anchor = leanDraws.AnchorAt(job.anchor);
    DirectDraws::Add(Counter::Draws);
    struct FallbackMark {
        ~FallbackMark() { ThreadReport::MarkItem(ThreadReport::ItemMark::PlanFallback); }
    } fallbackMark;
    DirectDraws::Add(Counter::FallbackPending);
    finishLeanDraw(job, anchor);
}

}
