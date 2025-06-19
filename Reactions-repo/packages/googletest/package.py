from spack import *
import os
import shutil
from spack.pkg.builtin.googletest import Googletest as builtinGoogletest

class Googletest(builtinGoogletest):
    depends_on("adaptivecpp")

