#include <pybind11/pybind11.h>
#include "includes/Engine.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_engine,m){
    m.def("create_sequence", &Engine::AddSequence, "Add a new sequence to the engine",
        py::arg("seq_id"), py::arg("token_ids"));
}