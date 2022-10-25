#include "init_models.h"


void init_models(pybind11::module_ &m){
    pybind11::class_<Species>(m, "Species")
        .def(pybind11::init<std::string, int, int, int>(),
            pybind11::arg("name"),
            pybind11::arg("lower") = 0,
            pybind11::arg("upper") = 1,
            pybind11::arg("default") = 0)
        .def_property("name", &Species::get_name, &Species::set_name)
        .def_property("lower", &Species::get_lower, &Species::set_lower)
        .def_property("upper", &Species::get_upper, &Species::set_upper)
        .def_property("default", &Species::get_default, &Species::set_default)
        .def_property("index", &Species::get_index, &Species::set_index)
        .def_property_readonly("dim", &Species::get_index);
    pybind11::class_<Rate>(m, "Rate")
        .def(pybind11::init<const std::string&, double>(),
            pybind11::arg("name"),
            pybind11::arg("value") = 1)
        .def_property_readonly("name", &Rate::get_name)
        .def_property_readonly("value", &Rate::get_value);
    pybind11::class_<Event>(m, "Event")
        .def(pybind11::init<const Event&>(),
            pybind11::arg("event"))
        .def(pybind11::init<std::string, std::vector<std::string>, std::vector<std::string>, Rate, pybind11::tuple, std::vector<int>, pybind11::dict>(),
            pybind11::arg("name"),
            pybind11::arg("input_species"),
            pybind11::arg("output_species"),
            pybind11::arg("rate"),
            pybind11::arg("propensity_callable"),
            pybind11::arg("change_vec"),
            pybind11::arg("species_dict")=pybind11::dict())
        .def_property("name", &Event::get_name, &Event::set_name)
        .def_property_readonly("input_species", &Event::get_input_species)
        .def_property_readonly("output_species", &Event::get_output_species)
        .def_property_readonly("rate", &Event::get_rate)
        .def_property_readonly("change_vec", &Event::get_change_vec)
        .def_property_readonly("species_dict", &Event::get_species_dict)
        .def("propensity", &Event::propensity_np,
            pybind11::arg("state"))
        .def("hazard", &Event::hazard_np,
            pybind11::arg("state"));
    pybind11::class_<MJP>(m, "MJP")
        .def(pybind11::init<std::string>(),
            pybind11::arg("name"))
        .def_property("name", &MJP::get_name, &MJP::set_name)
        .def_property_readonly("num_species", &MJP::get_num_species)
        .def_property_readonly("num_events", &MJP::get_num_events)
        .def_property_readonly("num_states", &MJP::get_num_states)
        .def_property_readonly("num_rates", &MJP::get_num_rates)
        .def_property_readonly("dims", &MJP::get_dims)
        .def_property_readonly("default_state", &MJP::get_default_state)
        .def_property_readonly("species_dict", &MJP::get_species_dict)
        .def_property_readonly("event_dict", &MJP::get_event_dict)
        .def_property_readonly("species_list", &MJP::get_species_list)
        .def_property_readonly("event_list", &MJP::get_event_list)
        .def_property_readonly("rate_list", &MJP::get_rate_list)
        .def_property_readonly("rate_array", &MJP::get_rate_array)
        .def_property_readonly("input_species", &MJP::get_input_species)
        .def_property_readonly("output_species", &MJP::get_output_species)
        .def_property_readonly("change_vectors", &MJP::get_change_vectors)
        .def("add_species", &MJP::add_species, 
            pybind11::arg("species"))
        .def("add_species", &MJP::make_add_species,
            pybind11::arg("name"),
            pybind11::arg("lower") = 0,
            pybind11::arg("upper") = 1,
            pybind11::arg("default_value") = 0)
        .def("add_rate", &MJP::add_rate,
            pybind11::arg("rate"))
        .def("add_rate", &MJP::make_add_rate,
            pybind11::arg("name"),
            pybind11::arg("value"))
        .def("add_event", &MJP::add_event,
            pybind11::arg("event"))
        .def("add_event", &MJP::make_add_event,
            pybind11::arg("name"),
            pybind11::arg("input_species"),
            pybind11::arg("output_species"),
            pybind11::arg("rate"),
            pybind11::arg("hazard_callable"),
            pybind11::arg("change_vec"))
        .def("build", &MJP::build)
        .def("hazard", &MJP::hazard_out,
            pybind11::arg("state"))
        .def("propensity", &MJP::propensity_out,
            pybind11::arg("state"))
        .def("update_state", &MJP::update_state_out,
            pybind11::arg("state"),
            pybind11::arg("event"))
        .def("state2ind", &MJP::state2ind,
            pybind11::arg("state"))
        .def("ind2state", &MJP::ind2state_np,
            pybind11::arg("ind"))
        .def("species_index", static_cast<unsigned (MJP::*)(const std::string& )>(&MJP::species_index),
        pybind11::arg("species"))
        .def("species_index", static_cast<std::vector<unsigned> (MJP::*)(const std::vector<std::string>& )>(&MJP::species_index),
            pybind11::arg("species_list"))
        .def("event_index", static_cast<unsigned (MJP::*)(const std::string& )>(&MJP::event_index),
        pybind11::arg("event"))
        .def("event_index", static_cast<std::vector<unsigned> (MJP::*)(const std::vector<std::string>& )>(&MJP::event_index),
            pybind11::arg("event_list"))
        .def("rate_index", static_cast<unsigned (MJP::*)(const std::string& )>(&MJP::rate_index),
        pybind11::arg("rate"))
        .def("parse_clusters", &MJP::parse_clusters);
    pybind11::class_<ObservationModel>(m, "ObservationModel")
        .def(pybind11::init<MJP*, const std::vector<std::string>&, pybind11::tuple, pybind11::tuple, pybind11::tuple, unsigned, unsigned>(),
            pybind11::arg("transition_model"),
            pybind11::arg("rv_list"),
            pybind11::arg("transformation_callable"),
            pybind11::arg("sample_callable"),
            pybind11::arg("llh_callable"),
            pybind11::arg("transform_dim"),
            pybind11::arg("obs_dim"))
        .def("transform", &ObservationModel::transform,
            pybind11::arg("time"),
            pybind11::arg("state"),
            pybind11::arg("param"))
        .def("transform", &ObservationModel::llh,
            pybind11::arg("time"),
            pybind11::arg("state"),
            pybind11::arg("param"))
        .def("transform", &ObservationModel::sample_np,
            pybind11::arg("time"),
            pybind11::arg("state"),
            pybind11::arg("param"),
            pybind11::arg("seed"));
    return;
}
