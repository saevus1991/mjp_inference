#include "species.h"

// constructor

Species::Species(std::string name_, int lower_, int upper_, int default_value_) : 
    name(name_),
    lower(lower_),
    upper(upper_),
    default_value(default_value_),
    index(0)
    { }
