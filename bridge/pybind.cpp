
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "include/Engine.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_engine, m) {
    py::class_<Engine>(m, "Engine")
        .def(py::init<>())
        .def("create_sequence", &Engine::create_sequence, py::arg("seq_id"), py::arg("token_ids"),
             "Add a new sequence to the engine")
        .def("get_sequence_output",
             [](Engine& self, size_t seq_id) {
                 std::vector<size_t> output_token_ids;
                 self.get_sequence_output(seq_id, output_token_ids);
                 return output_token_ids;
             },
             py::arg("seq_id"),
             "Get the output token IDs for a given sequence ID")
        .def("check_sequence_state",
             [](Engine& self, size_t seq_id) {
                 SequenceState state;
                 self.check_sequence_state(seq_id, state);
                 return state;
             },
             py::arg("seq_id"),
             "Check the state of a given sequence ID");
    py::enum_<SequenceState>(m, "SequenceState")
        .value("PREPARE", SequenceState::PREPARE)
        .value("READY", SequenceState::READY)
        .value("PREFILLING", SequenceState::PREFILLING)
        .value("PREFILLED", SequenceState::PREFILLED)
        .value("DECODING", SequenceState::DECODING)
        .value("FINISHED", SequenceState::FINISHED);
}