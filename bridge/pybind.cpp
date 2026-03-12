#include <pybind11/pybind11.h>
#include "includes/Engine.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_engine,m){
    m.def("create_sequence", &Engine::AddSequence, "Add a new sequence to the engine",
        py::arg("seq_id"), py::arg("token_ids"));
    m.def("get_sequence_output", &Engine::GetSequenceOutput, "Get the output token IDs for a given sequence ID",
        py::arg("seq_id"), py::arg("output_token_ids"));
    m.def("check_sequence_state", &Engine::CheckSequenceState, "Check the state of a given sequence ID",
        py::arg("seq_id"), py::arg("state"));
}