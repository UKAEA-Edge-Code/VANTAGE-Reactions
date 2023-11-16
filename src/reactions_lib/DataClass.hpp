#include <iostream>
#include <string>

#pragma once

/** update the wrap in `python/<reactions>_pybind11.cpp` if this header API changed
*/
class DataClass {
public:
  DataClass();
  DataClass(std::string); // this is the constructor, which takes a string
  std::string
  message(void) const; // this a method that does not mutate the object
  void
  setMessage(const std::string message); // this method does mutate the object

private:
  std::string _message; // this is private member variable
};
