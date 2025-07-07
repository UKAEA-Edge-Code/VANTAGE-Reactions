from spack import *
import os
import shutil
from spack.pkg.builtin.googletest import Googletest as builtinGoogletest

class Googletest(builtinGoogletest):
    variant("csd3", default=False, description="Variant for defining when googletest will be built on CSD3.")
    depends_on("adaptivecpp", when = "+csd3")

