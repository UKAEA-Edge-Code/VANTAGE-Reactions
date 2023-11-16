#include "DataClass.hpp"
#include <gtest/gtest.h>

TEST(DataClassTest, CheckDefaultData) {
  DataClass data;
  EXPECT_EQ("default constructor", data.message());
}
