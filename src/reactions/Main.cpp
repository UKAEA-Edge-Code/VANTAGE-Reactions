// Copyright [2020] UKAEA
#include <iostream>
#include "DataClass.hpp"

int main() {
  std::string constructorMessage = "Hello World";
  // create an object of a class that is defined in the data data
  DataClass data(constructorMessage);
  // class a member function of the object
  const auto messageFromData = data.message();
  std::cout << messageFromData << std::endl;

  return 0;
}
