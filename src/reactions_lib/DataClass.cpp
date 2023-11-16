#include "DataClass.hpp"
#include <iostream>

DataClass::DataClass() { _message = std::string{"default constructor"}; }
DataClass::DataClass(std::string message) { _message = message; }

std::string DataClass::message(void) const { return _message; }

void DataClass::setMessage(const std::string message) { _message = message; }
