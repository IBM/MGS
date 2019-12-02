#! /usr/bin/env python3
"""
help to generate Makefile
"""
# pylint: disable=too-many-lines

from __future__ import print_function
# from contextlib import contextmanager
import os
import os.path
import sys
import distutils.spawn
if sys.version_info[0] < 3:
    import popen2

    def find_version(name, full_path_name):
        cmd = full_path_name + " --version"
        (stdoutFile, stdinFile, stderrFile) = popen2.popen3(cmd)
        stderr = stderrFile.read()
        if stderr != "":
            raise FatalError(name + " has an error for command " + cmd)
        stdout = stdoutFile.readline()
        if stdout.find("GNU") == -1:
            raise FatalError(full_path_name + " is not GNU")
        tokens = stdout.split()
        file_version = tokens[-1]
        return file_version
else:
    import subprocess

    def find_version(name, full_path_name):
        print(type(full_path_name))
        cmd = [full_path_name, "--version"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        stdout, stderr = p.communicate()
        lines = stdout.splitlines()
        line = lines[0].decode('ascii')
        tokens = line.split()
        file_version = tokens[-1]
        return file_version

# import pip
# import string

# if 'builtins' not in sys.modules.keys():
        # pip.main(['install', 'future'])
import getopt
from builtins import range

# If we decide to put modules in the scripts directory use the following to
# be able to import modules
# sys.path.append("scripts/")

# pylint:disable=invalid-name,
# pylint:disable=C0301,too-many-branches
# Constants
USE = 1
DONTUSE = 0
UNDEF = -1

PROJECT_NAME = "Lens"
CONFIG_HEADER = "framework/factories/include/LensRootConfig.h"
EXTENSIONS_MK = "./Extensions.mk"
CLEAN_SCRIPT = "./clean.sh"
CONFIG_LOG = "./config.log"
TAB = "   "

AIX_GNU_CPP_COMPILER = "g++"  # -3.2.3"
AIX_GNU_C_COMPILER = "gcc"
AIX_XL_CPP_COMPILER = "xlC_r"
AIX_XL_C_COMPILER = "xlc_r"

AIX_GNU_64BIT_FLAG = "-maix64"
AIX_XL_64BIT_FLAG = "-q64"

XL_RUNTIME_TYPE_INFO_FLAG = "-qrtti=all"
XL_BG_RUNTIME_TYPE_INFO_FLAG = "-qrtti"
BGP_FLAGS = "-qarch=450 -qtune=450 -qdebug=unlimitin"
BGQ_FLAGS = "-qarch=qp -qtune=qp -qdebug=unlimitin"

GNU_WARNING_FLAGS = "-Wall -Wno-unused -Wpointer-arith -Wcast-qual -Wcast-align"

LINUX_EXTRA_FLAGS = "-DLINUX"
LINUX_SHARED_PFIX = "-shared -Wl,--export-dynamic -Wl,-soname,lib$(notdir $@)"
LINUX_SHARED_CC = "$(CC) $(SHARED_PFIX) -o $@ $(filter %.o, $^)"
LINUX_FINAL_TARGET_FLAG = "-Wl,--export-dynamic"

AIX_GNU_MINIMAL_TOC_FLAG = "-mminimal-toc"

AIX_EXTRA_FLAGS = "-DAIX"
AIX_COMMON_SHARED_PFIX = "$(MAKE64)"
AIX_GNU_SHARED_PFIX = AIX_COMMON_SHARED_PFIX + " -shared -Wl,-bnoautoexp"
AIX_XL_SHARED_PFIX = AIX_COMMON_SHARED_PFIX + " -G"
AIX_SHARED_CC = "$(CC) $(SHARED_PFIX) -o $@ $(filter %.o, $^) $(shell grep -A 1 %$(notdir $(basename $@)):% so/Dependfile | grep -v :%)"
AIX_GNU_FINAL_TARGET_FLAG = "$(MAKE64) "
AIX_XL_FINAL_TARGET_FLAG = AIX_GNU_FINAL_TARGET_FLAG + " " + XL_RUNTIME_TYPE_INFO_FLAG

GNU_DEBUGGING_FLAG = "-ggdb -g"
XL_DEBUGGING_FLAG = "-g"

PROFILING_FLAGS = "-pg"

O_OPTIMIZATION_FLAG = "-O"
O2_OPTIMIZATION_FLAG = "-O2"
O3_OPTIMIZATION_FLAG = "-O3"
O4_OPTIMIZATION_FLAG = "-O4"
O5_OPTIMIZATION_FLAG = "-O5"
OG_OPTIMIZATION_FLAG = "-Og"

DEBUG_ASSERT = "-DDEBUG_ASSERT"
DEBUG_HH = "-DDEBUG_HH"
DEBUG_LOOPS = "-DDEBUG_LOOPS"
DEBUG_CPTS = "-DDEBUG_CPTS"
NOWARNING_DYNAMICCAST = "-DNOWARNING_DYNAMICCAST"

COMMON_DX_CFLAGS = " -I . -I$(DX_INCLUDE) -I$(DX_BASE)/include $(MAKE64)"
AIX_DX_CFLAGS = "-O -Dibm6000 " + COMMON_DX_CFLAGS
LINUX_DX_CFLAGS = " -O -Dlinux " + COMMON_DX_CFLAGS

COMMON_DX_LITELIBS = "-lDXlite -lm"
AIX_DX_LITELIBS = "-L$(DX_BASE)/lib_ibm6000 " + COMMON_DX_LITELIBS
LINUX_DX_LITELIBS = "-L$(DX_BASE)/lib_linux " + COMMON_DX_LITELIBS

EXTRA_PARSER_TARGERS_FOR_DX = "$(DX_DIR)/EdgeSetSubscriberSocket $(DX_DIR)/NodeSetSubscriberSocket $(OBJS_DIR)/socket.o"

LENSPARSER_TARGETS = "$(OBJS_DIR)/speclang.tab.o $(OBJS_DIR)/lex.yy.o $(OBJS_DIR)/socket.o $(BASE_OBJECTS) "

# # Pre python 2.3 compatibility
# True = 1
# False = 0


class FatalError(Exception):
    """
    class handle error
    """
    def __init__(self, value=""):
        super(FatalError, self).__init__()
        self.value = value

    def __str__(self):
        return repr(self.value)

    def printError(self):
        """print msg"""
        if self.value != "":
            print("Fatal error:", self.value)


class InternalError(Exception):
    """internal error handling"""
    def __init__(self, value=""):
        super(InternalError, self).__init__()
        self.value = value

    def __str__(self):
        return repr(self.value)


def printWarning(warning):
    """print any message considered as warning"""
    print("Warning:", warning, "\n")


def printFeedback(feedback, extraLine=False):
    """print any message considered as feedback"""
    print(feedback)
    if extraLine is True:
        print()


def findFile(name, required=False):
    """
    find file path
    """
    # cmd = "which " + name
    stdout = distutils.spawn.find_executable(name)
    if stdout is None:
        if required is True:
            raise FatalError("Required file " + name + " could not be found.")
        return ""
    # strip the \n at the end
    stdout = stdout.rstrip()
    return stdout


def getFileStatus(name):
    """return if file exists"""
    retStr = ""
    if name != "":
        retStr = os.path.basename(name) + " is found at: " + os.path.dirname(name)
    return retStr


def addUnderScore(name, scoreChar='='):
    """add equal sign"""
    retStr = name + "\n"
    if name != "":
        for _i in range(len(name)):
            retStr += scoreChar
        retStr += "\n"
    return retStr


def getFirst(a):
    """return first element in pair"""
    (first, _second) = a
    return first


def createConfigHeader():
    """create config header"""
    rootDir = os.getcwd()
    create = True
    try:
        current = open(CONFIG_HEADER, "r")
        lines = current.readlines()
        line = lines[2]
        begin = line.find('"') + 1
        end = line.find('"', begin)
        if not (begin == -1 or end == -1):
            currentDir = line[begin:end]
            if currentDir == rootDir:
                create = False
    except Exception:
        pass

    if create is True:
        cmd = "#include <string>\n\n"
        cmd += "const std::string LENSROOT = \"" + rootDir + "\";\n"
        header = open(CONFIG_HEADER, "w")
        header.write(cmd)
        header.close()
    else:
        printFeedback(CONFIG_HEADER + " looks up to date, not overwriting", True)


def touchExtensionsMk():
    """ renew extension file"""
    if os.path.isfile(EXTENSIONS_MK) is False:
        f = open(EXTENSIONS_MK, "w")
        f.close()


class DxInfo:
    """ deal with DX"""
    def __init__(self):
        self.bin = ""
        self.exists = False
        self.binPath = ""
        self.mainPath = ""
        self.liteLib = ""
        self.include = ""

    def setup(self, operatingSystem):
        """setup"""
        self.bin = findFile("dx")
        if self.bin == "":
            print("DX is not found", self.mainPath)
            return
        self.binPath = os.path.dirname(self.bin)
        self.mainPath = os.path.dirname(self.binPath) + "/dx"
        if os.path.isdir(self.mainPath) is not True:
            print("DX main path could not be found at", self.mainPath)
            return

        self.liteLib = self.mainPath
        if (operatingSystem == "AIX"):
            self.liteLib += "/lib_ibm6000/"
        else:
            self.liteLib += "/lib_linux/"

        self.liteLib += "libDXlite.a"
        if os.path.isfile(self.liteLib) is not True:
            print("libDXlite.a could not be found at", self.liteLib)
            return

        self.include = self.mainPath + "/include/dx/dx.h"
        if os.path.isfile(self.include) is not True:
            print("dx.h could not be found at", self.include)
            return

        self.exists = True

    def getInfo(self):
        """return info """
        retStr = TAB + "DX executable is found at" + self.bin + "\n"
        retStr += TAB + "DX main path is found at" + self.mainPath + "\n"
        retStr += TAB + "DX library is found at" + self.liteLib + "\n"
        retStr += TAB + "DX include file is found at" + self.include + "\n"
        return retStr


class Options:
    """handle options"""
    def __init__(self, argv):
        """init file"""
        # options
        self.compilationMode = "undefined"  # 32, 64, undefined
        self.withDX = UNDEF  # USE, DONTUSE, UNDEF
        self.compiler = "undefined"  # gcc, xl, undefined
        self.silent = False  # True, False
        self.verbose = False  # True, False
        self.extMode = False  # True, False
        self.debug = DONTUSE  # USE, DONTUSE, UNDEF
        self.debug_assert = False  # True, False
        self.debug_hh = False  # True, False
        self.debug_loops = False  # True, False
        self.debug_cpts = False  # True, False
        self.nowarning_dynamiccast = False  # True, False
        self.profile = DONTUSE  # USE, DONTUSE, UNDEF
        self.tvMemDebug = DONTUSE  # USE, DONTUSE, UNDEF
        self.mpiTrace = DONTUSE  # USE, DONTUSE, UNDEF
        self.optimization = "undefined"  # O, O2, O3, O4, O5, Og, undefined
        self.dynamicLoading = False  # True, False
        self.domainLibrary = False  # True, False
        self.pthreads = True  # True, False
        self.withMpi = False  # True, False
        self.withGpu = False  # True, False
        self.withArma = False  # True, False
        self.blueGeneL = False  # True, False
        self.blueGeneP = False  # True, False
        self.blueGeneQ = False  # True, False
        self.blueGene = False  # True, False
        self.rebuild = False  # True, False
        self.asNts = False  # True, False
        self.asNtsNVU = False  # True, False
        self.asNGS = False  # True, False
        self.asMgs = False  # True, False
        self.colab = False  # True, False
        self.help = False  # True, False

        self.cmdOptions = [("32-bit", "32 bit compilation mode"),
                           ("64-bit", "64 bit compilation mode"),
                           ("with-dx", "build with dx support"),
                           ("without-dx", "build without dx support"),
                           ("with-gcc", "compile using GNU compilers"),
                           ("with-xl", "compile using XL compilers"),
                           ("silent", "silence command line and loader output"),
                           ("verbose", "set -DVERBOSE for arbitrary user instrumentation"),
                           ("ext-mode", "run in extensions mode, Dependfile is not overwritten"),
                           ("O", "optimize at level 1"),
                           ("O2", "optimize at level 2"),
                           ("O3", "optimize at level 3"),
                           ("O4", "optimize at level 4"),
                           ("O5", "optimize at level 5"),
                           ("Og", "optimize with debug information"),
                           ("debug", "compile with debugging flags"),
                           ("debug_assert", "compile with debugging flags for assert"),
                           ("debug_hh", "compile with debugging flags for Hodgkin-Huxley compartments"),
                           ("debug_loops", "compile with debugging flags for methods to be called iteratively (time loop)"),
                           ("debug_cpts", "compile with debugging flags to print out the information for every compartments"),
                           ("nowarning_dynamiccast", "disable printing out Dynamic Cast failed warning messages"),
                           ("tvMemDebug", "enable totalview memory debugging for parallel jobs (perfomance impact)"),
                           ("mpiTrace", "enable mpiTrace profiling (for BG)"),
                           ("profile", "enable generating profiling (to be used by gprof)"),
                           ("enable-dl", "enable dynamic loading, else everything is statically linked"),
                           ("domainLib", "link to domain specific library"),
                           ("disable-pthreads", "disable pthreads, there will be a single thread"),
                           ("with-mpi", "enables mpi"),
                           ("with-gpu", "enables gpu extensions"),
                           ("with-arma", "enables Armadillo extensions"),
                           ("blueGeneL", "configures for blueGeneL environment"),
                           ("blueGeneP", "configures for blueGeneP environment"),
                           ("blueGeneQ", "configures for blueGeneQ environment"),
                           ("rebuild", "rebuilds the project"),
                           ("as-NTS", "configures as NTS"),
                           ("as-NTS-NVU", "configures as NTS+NVU"),
                           ("as-NGS", "configures as NGS"),
                           ("as-MGS", "configures as MGS"),
                           ("colab", "build for collaborators"),
                           ("as-both-Nts-Mgs", "configures as both NTS and MGS"),
                           ("help", "displays the available options")]

        self.parseOptions(argv)

    def getOptionList(self):
        """return list of options"""
        return map(getFirst, self.cmdOptions)

    def usage(self, argv):
        """how to use"""
        print(addUnderScore("Possible command line options:"))
        for i in self.cmdOptions:
            m_str = TAB + "--" + i[0] + ": " + i[1]
            print(m_str)

    def parseOptions(self, argv):  # noqa
        """
        parsing the command-line inputs
        """
        try:
            opts, args = getopt.getopt(argv[1:], '', self.getOptionList())
            for o, a in opts:
                if o == "--32-bit":
                    if self.compilationMode == "undefined":
                        self.compilationMode = "32"
                    else:
                        raise FatalError("32-bit and 64-bit are used at the same time.")
                if o == "--64-bit":
                    if self.compilationMode == "undefined":
                        self.compilationMode = "64"
                    else:
                        raise FatalError("32-bit and 64-bit are used at the same time.")
                if o == "--with-dx":
                    if self.withDX == UNDEF:
                        self.withDX = USE
                    else:
                        raise FatalError(
                            "with-dx and without-dx are used at the same time.")
                if o == "--without-dx":
                    if self.withDX == UNDEF:
                        self.withDX = DONTUSE
                    else:
                        raise FatalError(
                            "with-dx and without-dx are used at the same time.")
                if o == "--with-gcc":
                    if self.compiler == "undefined":
                        self.compiler = "gcc"
                    else:
                        raise FatalError(
                            "with-gcc and with-xl are used at the same time.")
                if o == "--with-xl":
                    if self.compiler == "undefined":
                        self.compiler = "xl"
                    else:
                        raise FatalError(
                            "with-gcc and with-xl are used at the same time.")
                if o == "--ext-mode":
                    self.extMode = True
                if o == "--O":
                    if self.optimization == "undefined":
                        self.optimization = "O"
                    else:
                        raise FatalError("O, O2, O3, O4, O5, and/or Og are used at the same time.")
                if o == "--O2":
                    if self.optimization == "undefined":
                        self.optimization = "O2"
                    else:
                        raise FatalError("O, O2, O3, O4, O5, and/or Og are used at the same time.")
                if o == "--O3":
                    if self.optimization == "undefined":
                        self.optimization = "O3"
                    else:
                        raise FatalError("O, O2, O3, O4, O5, and/or Og are used at the same time.")
                if o == "--O4":
                    if self.optimization == "undefined":
                        self.optimization = "O4"
                    else:
                        raise FatalError("O, O2, O3, O4, O5, and/or Og are used at the same time.")
                if o == "--O5":
                    if self.optimization == "undefined":
                        self.optimization = "O5"
                    else:
                        raise FatalError("O, O2, O3, O4, O5, and/or Og are used at the same time.")
                if o == "--Og":
                    if self.optimization == "undefined":
                        self.optimization = "Og"
                    else:
                        raise FatalError("O, O2, O3, O4, O5, and/or Og are used at the same time.")
                if o == "--debug":
                    self.debug = USE
                if o == "--debug_assert":
                    self.debug_assert = True
                    self.debug = USE
                if o == "--debug_hh":
                    self.debug_hh = True
                    self.debug = USE
                if o == "--debug_loops":
                    self.debug_loops = True
                    self.debug = USE
                if o == "--debug_cpts":
                    self.debug_cpts = True
                    self.debug = USE
                if o == "--nowarning_dynamiccast":
                    self.nowarning_dynamiccast = True
                if o == "--profile":
                    self.profile = USE
                if o == "--tvMemDebug":
                    self.tvMemDebug = USE
                if o == "--mpiTrace":
                    self.mpiTrace = USE
                if o == "--enable-dl":
                    self.dynamicLoading = True
                if o == "--domainLib":
                    self.domainLibrary = True
                if o == "--disable-pthreads":
                    self.pthreads = False
                if o == "--with-mpi":
                    self.withMpi = True
                if o == "--with-gpu":
                    self.withGpu = True
                if o == "--with-arma":
                    self.withArma = True
                if o == "--silent":
                    self.silent = True
                if o == "--verbose":
                    self.verbose = True
                if o == "--blueGeneL":
                    self.blueGeneL = True
                if o == "--blueGeneP":
                    self.blueGeneP = True
                if o == "--blueGeneQ":
                    self.blueGeneQ = True
                if o == "--rebuild":
                    self.rebuild = True
                if o == "--as-NTS":
                    self.asNts = True
                if o == "--as-NTS-NVU":
                    self.asNtsNVU = True
                if o == "--as-NGS":
                    self.asNGS = True
                if o == "--as-MGS":
                    self.asMgs = True
                if o == "--colab":
                    self.colab = True
                if o == "--help":
                    self.help = True

            # Just display help and return
            if self.help is True:
                self.usage(argv)
                return

            if self.extMode is True and self.rebuild is True:
                raise FatalError("Do not rebuild with the ext-mode turned on.")

            if self.debug == USE and self.optimization != "Og":
                printWarning("Debugging is turned on even though optimization is " +
                             self.optimization + ".\nSetting optimization to --Og")
                self.optimization = "Og"

            if self.profile == USE and self.optimization != "Og":
                printWarning("Profiling is turned on even though optimization is " +
                             self.optimization + ".\nSetting optimization to --Og")
                self.optimization = "Og"

            # if self.profile == USE and self.debug != USE:
            #    printWarning("Profiling is turned on so debugging turned on by default.")
            #    self.debug = USE

            if self.debug != USE and self.profile != USE and self.optimization == "undefined":
                printFeedback(
                    "Debugging, profiling, and optimization are not defined, choosing O3 as default.")
                print
                self.optimization = "O3"

            if self.withMpi is True and self.dynamicLoading is True:
                printFeedback("Dynamic loading will be disabled due to existence of MPI.")
                self.dynamicLoading = False

            if self.blueGeneL is True:
                self.blueGene = True
                self.withMpi = True
                self.pthreads = False
                self.dynamicLoading = False
                self.withDX = DONTUSE
                self.compilationMode = "32"
                self.silent = True
                self.optimization = "O3"

            if self.blueGeneP is True:
                self.blueGene = True
                self.withMpi = True
                self.pthreads = True
                self.dynamicLoading = False
                self.withDX = DONTUSE
                self.compilationMode = "32"
                self.silent = True
                self.optimization = "O3"

            if self.blueGeneQ is True:
                self.blueGene = True
                self.withMpi = True
                self.pthreads = True
                self.dynamicLoading = False
                self.withDX = DONTUSE
                self.compilationMode = "32"
                self.silent = True
                self.optimization = "O3"

        except getopt.GetoptError:
            # print help information and exit:
            print("Error in parsing configure.py")
            self.usage(argv)
            raise FatalError


class BuildSetup:
    """to build"""
    def __init__(self):
        """constructor"""
        # System defining variables
        self.operatingSystem = ""
        self.hostName = ""
        self.architecture = ""
        self.objectMode = ""
        self.numCPUs = os.sysconf("SC_NPROCESSORS_ONLN")

        # Binaries
        self.bisonBin = ""
        self.flexBin = ""
        self.grepBin = ""
        self.makeBin = ""

        # versions
        self.bisonVersion = ""

        # DX Variable
        self.dx = ""

        # Compilers
        self.cCompiler = ""
        self.cppCompiler = ""
        self.nvccCompiler = ""

        # decide if we want to compile all codes via nvcc
        # or just those with CUDA code using nvcc
        self.separate_compile = True #False

        # Command line options
        self.options = Options(sys.argv)

        self.unameInfo = os.uname()

        self.operatingSystem = self.unameInfo[0]
        self.hostName = self.unameInfo[1]

        if self.operatingSystem == "AIX":
            self.architecture = "ppc64"
            if self.options.compilationMode == "undefined":
                self.objectMode = os.getenv("OBJECT_MODE")
                if self.objectMode != "64":
                    self.objectMode = "32"
            else:
                self.objectMode = self.options.compilationMode
        elif self.operatingSystem == "Linux":
            self.architecture = self.unameInfo[4]
            self.objectMode = "64"
        else:
            raise FatalError(PROJECT_NAME + " only runs in AIX or Linux")

        self.bisonBin = findFile("bison", True)
        self.flexBin = findFile("flex", True)
        self.grepBin = findFile("grep", True)
        self.makeBin = findFile("make", True)

    def getSystemInfo(self):  # noqa
        """return sys info"""
        retStr = addUnderScore("System information:")
        retStr += TAB + "Operating system: " + self.operatingSystem + "\n"
        retStr += TAB + "HostName: " + self.hostName + "\n"
        retStr += TAB + "Architecture: " + self.architecture + "\n"
        retStr += TAB + "Object mode: " + self.objectMode + "\n"
        retStr += TAB + "Number of CPUs: " + str(self.numCPUs) + "\n"
        retStr += TAB + "C compiler: " + self.cCompiler + "\n"
        retStr += TAB + "C++ compiler: " + self.cppCompiler + "\n"

        retStr += TAB + "Silent mode: "
        if self.options.silent is True:
            retStr += "On"
        else:
            retStr += "Off"
        retStr += "\n"

        retStr += TAB + "Verbose mode: "
        if self.options.verbose is True:
            retStr += "On"
        else:
            retStr += "Off"
        retStr += "\n"

        retStr += TAB + "Extensions mode: "
        if self.options.extMode is True:
            retStr += "On"
        else:
            retStr += "Off"
        retStr += "\n"

        retStr += TAB + "Debugging: "
        if self.options.debug == USE:
            retStr += "On\n"
            if self.options.debug_assert is True:
                retStr += TAB + TAB + "- DEBUG_ASSERT: YES\n"
            if self.options.debug_hh is True:
                retStr += TAB + TAB + "- DEBUG_HH : YES\n"
            if self.options.debug_loops is True:
                retStr += TAB + TAB + "- DEBUG_LOOPS : YES\n"
            if self.options.debug_cpts is True:
                retStr += TAB + TAB + "- DEBUG_CPTS : YES\n"
        else:
            retStr += "Off\n"
        # retStr += "\n"

        retStr += TAB + "Profiling: "
        if self.options.profile == USE:
            retStr += "On"
        else:
            retStr += "Off"
        retStr += "\n"

        retStr += TAB + "Totalview Memory Debugging: "
        if self.options.tvMemDebug is True:
            retStr += "On"
        else:
            retStr += "Off"
        retStr += "\n"

        retStr += TAB + "MPI Trace Profiling: "
        if self.options.mpiTrace is True:
            retStr += "On"
        else:
            retStr += "Off"
        retStr += "\n"

        if self.options.optimization != "undefined":
            retStr += TAB + "Optimization level: " + self.options.optimization + "\n"

        retStr += TAB + "Dynamic loading: "
        if self.options.dynamicLoading is True:
            retStr += "Enabled"
        else:
            retStr += "Disabled"
        retStr += "\n"

        retStr += TAB + "Domain specific library: "
        if self.options.domainLibrary is True:
            retStr += "Enabled"
        else:
            retStr += "Disabled"
        retStr += "\n"

        retStr += TAB + "Pthreads: "
        if self.options.pthreads is True:
            retStr += "Enabled"
        else:
            retStr += "Disabled"
        retStr += "\n"

        retStr += TAB + "MPI: "
        if self.options.withMpi is True:
            retStr += "Used"
        else:
            retStr += "Not used"
        retStr += "\n"

        retStr += TAB + "GPU: "
        if self.options.withGpu is True:
            retStr += "Used"
        else:
            retStr += "Not used"
        retStr += "\n"

        retStr += TAB + "Armadillo: "
        if self.options.withArma is True:
            retStr += "Used"
        else:
            retStr += "Not used"
        retStr += "\n"

        retStr += TAB + "NTI: "
        if (self.options.asNts is True) or (self.options.asNtsNVU is True):
            retStr += "Used"
        else:
            retStr += "Not used"
        retStr += "\n"

        retStr += TAB + "DX: "

        if self.dx.exists is True:
            retStr += "Used\n"
        else:
            retStr += "Not used\n"

        retStr += "\n"
        retStr += addUnderScore("Absolute paths for tools and packages that are going to be used:")
        retStr += TAB + getFileStatus(self.bisonBin) + "\n"
        retStr += TAB + getFileStatus(self.flexBin) + "\n"
        retStr += TAB + getFileStatus(self.grepBin) + "\n"
        retStr += TAB + getFileStatus(self.makeBin) + "\n"

        if self.dx.exists is True:
            retStr += self.dx.getInfo()

        return retStr

    def createLog(self):
        """log file"""
        fStr = "Configure script is run as follows:\n"
        for i in sys.argv:
            fStr += i + " "

        fStr += "\n\n"

        fStr += self.getSystemInfo()

        f = open(CONFIG_LOG, "w")
        f.write(fStr)
        f.close()

    def bisonVersionFinder(self):
        """bison version"""
        self.bisonVersion = find_version("bison", self.bisonBin)

    def main(self):
        """main action"""
        if self.options.help is True:
            return

        self.setCompilers()
        self.setDX()
        self.bisonVersionFinder()
        print(self.getSystemInfo())

        # At this point if there has been no FatalError, the environment
        # and the options are ok, so create the config log before doing
        # any more work.
        self.createLog()

        if self.options.blueGene is True or self.operatingSystem == "Linux":
            os.system("mv -f framework/networks/include/pthread.h framework/networks/include/pthread.h.bak > /dev/null 2>&1")

        if self.options.rebuild is True:
            # print "Cleaning the project using:", CLEAN_SCRIPT, "\n"
            # os.system(CLEAN_SCRIPT)
            os.system("make LINUX clean")

        createConfigHeader()
        touchExtensionsMk()

        self.generateMakefile("Makefile")

        if self.options.rebuild is True:
            cmd = "make -j " + str(self.numCPUs)
            print("Starting the make process with:", cmd)
            os.system(cmd)

    def setDX(self):
        """configure DX"""
        self.dx = DxInfo()

        if self.objectMode == "64":
            if self.options.withDX == USE:
                raise FatalError("DX can not be enabled in 64 bit object mode")
        else:
            if self.options.withDX == UNDEF:
                # printFeedback(
                #     "Usage of dx not specified, checking if it exists.")
                # self.dx.setup(self.operatingSystem)
                # if self.dx.exists is True:
                #     printFeedback("DX found, using...\n")
                self.options.withDX = DONTUSE
            elif self.options.withDX == USE:
                self.dx.setup(self.operatingSystem)
                if self.dx.exists is False:
                    raise FatalError("DX requested but not found.")

    def setCompilers(self):  # noqa
        """
        options to GCC
        """
        def setCompilersMPI(self):
            if self.operatingSystem == "AIX":
                if self.options.compiler == "gcc":
                    self.options.compiler = "gcc"
                    self.cCompiler = findFile("gcc", True)
                    findFile("gcc", True)
                    self.cppCompiler = findFile("g++", True)
                else:
                    self.options.compiler = "xl"
                    self.cCompiler = findFile("xlc_r", True)
                    # xlC_r is internally required for mpCC_r
                    findFile("xlC_r", True)
                    self.cppCompiler = findFile("mpCC_r", True)
            else:
                if self.options.blueGeneL is True:
                    self.options.compiler = "xl"
                    self.cCompiler = findFile("/opt/ibmcmp/vac/bg/9.0/bin/blrts_xlc", True)
                    self.cppCompiler = findFile("/opt/ibmcmp/vacpp/bg/9.0/bin/blrts_xlC", True)
                else:
                    if self.options.blueGeneP is True:
                        self.options.compiler = "xl"
                        self.cCompiler = findFile("mpixlc_r", True)
                        self.cppCompiler = findFile("mpixlcxx_r", True)
                    else:
                        if self.options.blueGeneQ is True:
                            self.options.compiler = "xl"
                            self.cCompiler = findFile("mpixlc_r", True)
                            self.cppCompiler = findFile("mpixlcxx_r", True)
                        else:
                            if self.operatingSystem == "Linux":
                                self.options.compiler = "mpicc"
                                self.cCompiler = findFile("mpicc", True)
                                findFile("mpicc", True)
                                self.cppCompiler = findFile("mpiCC", True)
                                findFile("mpiCC", True)
                                # raise FatalError("Currently MPI is only used by AIX")

        def setCompilersMPI_GPU(self):
            if self.operatingSystem == "Linux":
                self.options.compiler = "nvcc"
                # consider switching to using gcc and nvcc for different codes
                if self.separate_compile:
                    self.nvccCompiler = findFile("nvcc", True)
                    # self.nvccCompiler += " -ccbin g++"
                    # self.cCompiler = findFile("gcc", True)
                    # self.cppCompiler = findFile("g++", True)
                    self.cCompiler = findFile("mpicc", True)
                    self.cppCompiler = findFile("mpiCC", True)
                else:
                    self.cCompiler = findFile("nvcc", True)
                    # self.cCompiler += " -ccbin g++"
                    # findFile("nvcc", True)
                    self.cppCompiler = findFile("nvcc", True)
                # findFile("nvcc", True)
            else:
                raise FatalError("Currently MPI+GPU is only available on LINUX")

        if self.options.withMpi is True:
            if self.options.withGpu is True:
                setCompilersMPI_GPU(self)
            else:
                setCompilersMPI(self)
            return  # important do not continue this function after here.

        if not (self.options.compiler == "xl" or self.options.compiler == "gcc"):
            printFeedback("XL or GNU is not selected as compiler, choosing default, gcc\n")
            self.options.compiler = "gcc"

        if self.options.compiler == "xl":
            if self.architecture != "ppc64":
                raise FatalError("XL compilers can only be used in ppc64 architectures.")
            self.cCompiler = findFile("xlc_r", True)
            self.cppCompiler = findFile("xlC_r", True)
        elif self.options.compiler == "gcc":
            self.cCompiler = findFile("gcc", True)
            if self.operatingSystem == "AIX":
                self.cppCompiler = findFile("g++", True)
            else:
                self.cppCompiler = findFile("g++", True)
        else:  # error condition
            raise InternalError("Should not have hit here in setCompilers")

    def getInitialValues(self):
        """get the init part to Makefile"""
        retStr = \
            """\
# This is a code generated file, generated using configure.py
# To change any options please rerun ./configure.py with the desired options.
# To build the project execute make -j <number of processes>
.DEFAULT_GOAL:= final

BIN_DIR?=./bin
EXE_FILE=gslparser

"""
        if (self.options.asNts is True) or (self.options.asNtsNVU is True):
            retStr += \
                """\
NTI_DIR=../nti
NTI_OBJ_DIR=$(NTI_DIR)/obj
NTI_INC_DIR=$(NTI_DIR)/include

"""
        retStr += \
            """\
BISON=$(shell which bison)
FLEX=$(shell which flex)

LENSROOT = $(shell pwd)
OPERATING_SYSTEM = $(shell uname)

SCRIPTS_DIR := scripts
DX_DIR := dx
SO_DIR := $(LENSROOT)/so
PARSER_PATH := framework/parser
STD_UTILS_OBJ_PATH := utils/std/obj
TOTALVIEW_LIBPATH := /opt/toolworks/totalview.8.4.1-7/rs6000/lib
"""

        # BlueGene MPI flags
        if self.options.blueGeneL is True:
            retStr += \
                """\
BGL_ROOT=/bgl/BlueLight/ppcfloor
MPI_LIBS = -L$(BGL_ROOT)/bglsys/lib -lmpich.rts -lmsglayer.rts -lrts.rts -ldevices.rts
MPI_TRACE_LIBS = /bgl/local/lib/libmpitrace.a
MPI_INC = -I$(BGL_ROOT)/bglsys/include

"""
        if self.options.blueGeneP is True:
            retStr += \
                """\
BGP_ROOT=/bgsys/drivers/ppcfloor
MPI_INC = -I$(BGP_ROOT)/arch/include

"""

        if self.options.withGpu is True:
            retStr += """
# CUDA 7.5 only work until gcc 4.10
# CUDA 9.2. only work until gcc 6.x and ompi 1.10.6
"""
            retStr += "NVCC := " + self.nvccCompiler + " -ccbin g++\n"
        retStr += "CC := " + self.cppCompiler + "\n"
        retStr += "C_COMP := " + self.cCompiler + "\n"
        if self.options.withMpi is True:
            retStr += "HAVE_MPI := 1\n"
        if self.options.withGpu is True:
            retStr += "HAVE_GPU := 1\n"
        if self.options.withArma is True:
            retStr += "HAVE_ARMA := 1\n"
        if self.options.silent is True:
            retStr += "SILENT_MODE := 1\n"
        if self.options.verbose is True:
            retStr += "VERBOSE := 1\n"
        if self.options.blueGene is True:
            retStr += "USING_BLUEGENE := 1\n"
        if self.options.pthreads is True:
            retStr += "HAVE_PTHREADS := 1\n"
        if self.options.profile == USE:
            retStr += "PROFILING := 1\n"

        if self.options.withGpu is True:
            retStr += """
# Gencode arguments
#SMS ?= 35 37 50 52 60 61 70 75
SMS ?= 70 

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
    # Generate SASS code for each SM architecture listed in $(SMS)
    $(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
    # Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
    HIGHEST_SM := $(lastword $(sort $(SMS)))
    ifneq ($(HIGHEST_SM),)
        GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
    endif
endif
"""
        return retStr

    def getMake64(self):
        """check if 64-bit"""
        retStr = "MAKE64 = "
        if self.objectMode == "64":
            if self.operatingSystem == "AIX":
                if self.options.compiler == "gcc":
                    retStr += AIX_GNU_64BIT_FLAG
                elif self.options.compiler == "xl":
                    retStr += AIX_XL_64BIT_FLAG
                else:
                    raise InternalError("Compiler " + self.options.compiler + " is not found")
        retStr += "\n"
        return retStr

    def getModuleDefinitions(self):  # noqa
        """get info of all modules"""
        retStr = \
            """\
####################################################################
# Name of all submodules we want to build
# 1. modules from the GSL frameworks
# 2. modules from the extension (i.e. user-defined)
#
# part 1 --> which include framework/...
#               utils/...
"""
        _framework_modules = \
            """\
\t\tdca \\
\t\tdataitems \\
\t\tfactories \\
\t\tnetworks \\
\t\tparser \\
\t\tsimulation \\
\t\tfunctors \\
            """
        if self.options.colab is False:
            retStr += \
                """\
FRAMEWORK_MODULES := \\
""" + _framework_modules
        else:
            retStr += \
                """\
COLAB_FRAMEWORK_MODULES := \\
""" + _framework_modules
        _utils_modules = \
            """\
\t\tstd \\
\t\timg \\
\t\tmnist\\
"""
        if self.options.colab is False:
            retStr += \
                """\

UTILS_MODULES := \\
""" + _utils_modules
        else:
            retStr += \
                """

COLAB_UTILS_MODULES := \\
""" + _utils_modules

        if self.options.withMpi is True:
            retStr += \
                """\
\t\tstreams \\

"""

        retStr += \
            """\

BASE_MODULES := $(patsubst %,framework/%,$(FRAMEWORK_MODULES))
BASE_MODULES += $(patsubst %,utils/%,$(UTILS_MODULES))

CONSTANT_MODULES :=

EDGE_MODULES :=

INTERFACE_MODULES :=

NODE_MODULES :=

"""

        if self.options.colab is True:
            if (self.options.asNts is True) or (self.options.asNtsNVU is True):
                retStr += \
                    """\
COLAB_NODE_MODULES :=  \\
        HodgkinHuxleyVoltage \\
        VoltageEndPoint \\
        HodgkinHuxleyVoltageJunction \\
        VoltageJunctionPoint \\
        IP3Concentration \\
        IP3ConcentrationEndPoint \\
        IP3ConcentrationJunction \\
        IP3ConcentrationJunctionPoint \\
        CaConcentration \\
        CaConcentrationEndPoint \\
        CaConcentrationJunction \\
        CaConcentrationJunctionPoint \\
        CaERConcentration \\
        CaERConcentrationEndPoint \\
        CaERConcentrationJunction \\
        CaERConcentrationJunctionPoint \\
        ForwardSolvePoint1  \\
        ForwardSolvePoint2  \\
        ForwardSolvePoint3  \\
        ForwardSolvePoint4  \\
        ForwardSolvePoint5  \\
        ForwardSolvePoint6  \\
        ForwardSolvePoint7  \\
        BackwardSolvePoint0  \\
        BackwardSolvePoint1  \\
        BackwardSolvePoint2  \\
        BackwardSolvePoint3  \\
        BackwardSolvePoint4  \\
        BackwardSolvePoint5  \\
        BackwardSolvePoint6  \\
        ChannelNat  \\
        ChannelNap  \\
        ChannelNas  \\
        ChannelKIR  \\
        ChannelKDR  \\
        ChannelKAf  \\
        ChannelKAs  \\
        ChannelKRP  \\
        ChannelBKalphabeta  \\
        ChannelSK  \\
        KCaChannel \\
        CalChannel \\
        CahChannel \\
        ChannelHCN \\
        ChannelCaLv12_GHK \\
        ChannelCaLv13_GHK \\
        ChannelCaN_GHK \\
        ChannelCaPQ_GHK \\
        ChannelCaR_GHK \\
        ChannelCaT_GHK \\
        PumpPMCA \\
        SynapticCleft \\
        PreSynapticPoint \\
        AMPAReceptor \\
        AMPAReceptor_Markov \\
        NMDAReceptor \\
        GABAAReceptor \\
        SpineAttachment_Vm \\
        SpineAttachment_VmCai \\
        SpineAttachment_VmCaiCaER \\
        Connexon \\

"""  # noqa
        retStr += \
            """\

STRUCT_MODULES := CoordsStruct \\

"""
        if self.options.colab is True:
                retStr += \
                    """\
TRIGGER_MODULES :=

COLAB_TRIGGER_MODULES := UnsignedServiceTrigger \\

"""
        else:
                retStr += \
                    """\
TRIGGER_MODULES := UnsignedServiceTrigger \\

"""

        if self.options.colab is False:
            retStr += \
                """\
VARIABLE_MODULES := \\

"""  # noqa

        if self.options.colab is True:
            if (self.options.asNts is True) or (self.options.asNtsNVU is True):
                retStr += \
                    """\
COLAB_VARIABLE_MODULES := BasicNodeSetVariable \\
        NodeSetSPMVariable \\
        VoltageClamp \\
        PointCurrentSource \\
        PointCalciumSource \\
        CurrentPulseGenerator \\
        RampCurrentGenerator \\
        VoltageDisplay \\
        ConductanceDisplay \\
        ReversalPotentialDisplay \\
        CurrentDisplay \\
        CaCurrentDisplay \\
        AnyCurrentDisplay \\
        AnyFluxDisplay \\
        CalciumDisplay \\
        CalciumDomainDisplay \\
        AnyConcentrationDisplay \\
        CalciumVisualization \\
        VoltageVisualization \\
        SimulationSetter \\
        SimulationInfo \\
        DetectDataChangeOneCompartment \\

"""  # noqa
        if self.options.colab is False:
            retStr += \
                """\
FUNCTOR_MODULES := BidirectConnectNodeSetsFunctor \\
        BinomialDist \\
        CombineNVPairs \\
        ConnectNodeSetsFunctor \\
        DstDimensionConstrainedSampler \\
        DstRefDistanceModifier \\
        DstRefGaussianWeightModifier \\
        DstRefSumRsqrdInvWeightModifier \\
        DstScaledContractedGaussianWeightModifier \\
        DstScaledGaussianWeightModifier \\
        ExecuteShell \\
        Exp \\
        FloatArrayMaker \\
        GetDstNodeCoordFunctor \\
        GetNodeCoordFunctor \\
        GetPostNodeCoordFunctor \\
        GetPreNodeCoordFunctor \\
        GetPreNodeIndex \\
       	IsoSampler \\
        IsoSamplerHybrid \\
       	GradientLayout \\
       	NormalizedGradientLayout \\
        LoadMatrix \\
        LoadSparseMatrix \\
        Log \\
        ModifyParameterSet \\
        NameReturnValue \\
        Neg \\
        PolyConnectorFunctor \\
        RandomDispersalLayout \\
        RefAngleModifier \\
        RefDistanceModifier \\
        ReversedDstRefGaussianWeightModifier \\
        ReversedSrcRefGaussianWeightModifier \\
        ReverseFunctor \\
        Round \\
        Scale \\
        ServiceConnectorFunctor \\
        SetSourceArrayIndexFunctor \\
        SrcDimensionConstrainedSampler \\
        SrcRefDistanceModifier \\
        SrcRefDoGWeightModifier \\
        SrcRefGaussianWeightModifier \\
        SrcRefPeakedWeightModifier \\
        SrcRefSumRsqrdInvWeightModifier \\
        SrcScaledContractedGaussianWeightModifier \\
        SrcScaledGaussianWeightModifier \\
        Threshold \\
        ToroidalRadialSampler \\
        UniformDiscreteDist \\
"""
        _tissuefunctor_modules = \
            """\
        TissueConnectorFunctor \\
        TissueFunctor \\
        TissueLayoutFunctor \\
        TissueMGSifyFunctor \\
        TissueNodeInitFunctor \\
        TissueProbeFunctor \\
        Zipper \\
        ConnectNodeSetsByVolumeFunctor \\"""
        if self.options.colab is False:
            if (self.options.asNts is True) or (self.options.asNtsNVU is True):
                retStr += _tissuefunctor_modules + \
                    """

"""
        else:
                retStr += """\
COLAB_FUNCTOR_MODULES :=  \\
""" + _tissuefunctor_modules + \
                    """
        BidirectConnectNodeSetsFunctor \\
        BinomialDist \\
        CombineNVPairs \\
        ConnectNodeSetsFunctor \\
        DstDimensionConstrainedSampler \\
        DstRefDistanceModifier \\
        DstRefGaussianWeightModifier \\
        DstRefSumRsqrdInvWeightModifier \\
        DstScaledContractedGaussianWeightModifier \\
        DstScaledGaussianWeightModifier \\
        Exp \\
        FloatArrayMaker \\
        GetDstNodeCoordFunctor \\
        GetNodeCoordFunctor \\
        GetPostNodeCoordFunctor \\
        GetPreNodeCoordFunctor \\
        GetPreNodeIndex \\
        IsoSampler \\
        Log \\
        ModifyParameterSet \\
        NameReturnValue \\
        Neg \\
        PolyConnectorFunctor \\
        RandomDispersalLayout \\
        RefAngleModifier \\
        RefDistanceModifier \\
        ReversedDstRefGaussianWeightModifier \\
        ReversedSrcRefGaussianWeightModifier \\
        ReverseFunctor \\
        Round \\
        Scale \\
        ServiceConnectorFunctor \\
        SrcDimensionConstrainedSampler \\
        SrcRefDistanceModifier \\
        SrcRefDoGWeightModifier \\
        SrcRefGaussianWeightModifier \\
        SrcRefPeakedWeightModifier \\
        SrcRefSumRsqrdInvWeightModifier \\
        SrcScaledContractedGaussianWeightModifier \\
        SrcScaledGaussianWeightModifier \\
        Threshold \\
        ToroidalRadialSampler \\
        UniformDiscreteDist \\

"""  # noqa
        if (self.options.asMgs is True):
            retStr += \
                """\
"""
        retStr += \
            """\

# part 2 --> extension/...
# this files list all the modules we want to build
# so that we don't have to modify this Makefile
include Extensions.mk

## hold the relative path to all extension subfolders
EXTENSION_MODULES += $(patsubst %,constant/%,$(CONSTANT_MODULES))
EXTENSION_MODULES += $(patsubst %,edge/%,$(EDGE_MODULES))
EXTENSION_MODULES += $(patsubst %,functor/%,$(FUNCTOR_MODULES))
EXTENSION_MODULES += $(patsubst %,node/%,$(NODE_MODULES))
EXTENSION_MODULES += $(patsubst %,struct/%,$(STRUCT_MODULES))
EXTENSION_MODULES += $(patsubst %,trigger/%,$(TRIGGER_MODULES))
EXTENSION_MODULES += $(patsubst %,variable/%,$(VARIABLE_MODULES))

# those with only header files
SPECIAL_EXTENSION_MODULES += $(patsubst %,interface/%,$(INTERFACE_MODULES))

EXTENSION_MODULES := $(patsubst %,extensions/%,$(EXTENSION_MODULES))
SPECIAL_EXTENSION_MODULES := $(patsubst %,extensions/%,$(SPECIAL_EXTENSION_MODULES))

MODULES := $(BASE_MODULES)
MODULES += $(EXTENSION_MODULES)

SOURCES_DIRS := $(patsubst %,%/src, $(MODULES))
MYSOURCES := $(foreach dir,$(SOURCES_DIRS),$(wildcard $(dir)/*.C))

#IMPORTANT: If you want to ignore some files
#           add them here (just source file names)
MYSOURCES := $(filter-out %MatrixParser.C %ReadImage.C %Ini.C %Img.C %Matrx.C %ImgUtil.C %Wbuf.C %Lzwbuf.C %Pal.C %Bitbuf.C , $(MYSOURCES))
HEADERS_DIRS := $(patsubst %,%/include, $(MODULES))

SOURCES_FILENAME_ONLY :=$(shell for file in $(notdir $(MYSOURCES)); do \\
\t       echo $${file} ; \\
\t       done)
PURE_OBJS := $(patsubst %.C, %.o, $(SOURCES_FILENAME_ONLY))

"""
        if (self.options.asNts is True) or (self.options.asNtsNVU is True):
            retStr += \
                """\
NTI_OBJS := $(foreach dir,$(NTI_OBJ_DIR),$(wildcard $(dir)/*.o))
TEMP := $(filter-out $(NTI_OBJ_DIR)/neuroGen.o $(NTI_OBJ_DIR)/neuroDev.o $(NTI_OBJ_DIR)/touchDetect.o, $(NTI_OBJS))
NTI_OBJS := $(TEMP)

"""
        retStr += \
            """\
COMMON_DIR := ../common/obj
COMMON_OBJS := $(foreach dir,$(COMMON_DIR), $(wildcard $(dir)/*.o))

OBJS_DIR := obj
OBJS := $(patsubst %, $(OBJS_DIR)/%, $(PURE_OBJS))

vpath %.C $(SOURCES_DIRS)
vpath %.c $(SOURCES_DIRS)
vpath %.h $(HEADERS_DIRS) framework/parser/generated

$(OBJS) : | $(OBJS_DIR)

$(OBJS_DIR):
\tmkdir $(OBJS_DIR)

$(BIN_DIR):
\tmkdir $(BIN_DIR)
"""
        if self.options.colab is True:
            retStr += \
                """
COLAB_BASE_MODULES := $(patsubst %,framework/%,$(COLAB_FRAMEWORK_MODULES))
COLAB_BASE_MODULES += $(patsubst %,utils/%,$(COLAB_UTILS_MODULES))
COLAB_EXTENSION_MODULES += $(patsubst %,functor/%,$(COLAB_FUNCTOR_MODULES))
COLAB_EXTENSION_MODULES += $(patsubst %,node/%,$(COLAB_NODE_MODULES))
COLAB_EXTENSION_MODULES += $(patsubst %,trigger/%,$(COLAB_TRIGGER_MODULES))
COLAB_EXTENSION_MODULES += $(patsubst %,variable/%,$(COLAB_VARIABLE_MODULES))
COLAB_EXTENSION_MODULES := $(patsubst %,extensions/%,$(COLAB_EXTENSION_MODULES))

COLAB_MODULES = $(COLAB_BASE_MODULES)
COLAB_MODULES += $(COLAB_EXTENSION_MODULES)
COLAB_HEADERS_DIRS := $(patsubst %,%/include, $(COLAB_MODULES))
COLAB_SOURCES_DIRS := $(patsubst %,%/src, $(COLAB_MODULES))
COLAB_MYSOURCES := $(foreach dir,$(COLAB_SOURCES_DIRS),$(wildcard $(dir)/*.C))
COLAB_SOURCES_FILENAME_ONLY :=$(shell for file in $(notdir $(COLAB_MYSOURCES)); do \\
\t       echo $${file} ; \\
\t       done)
COLAB_OBJS_FILENAME_ONLY := $(patsubst %.C, %.o, $(COLAB_SOURCES_FILENAME_ONLY))
COLAB_OBJS := $(patsubst %, $(OBJS_DIR)/%, $(COLAB_OBJS_FILENAME_ONLY))
COLAB_OBJS += obj/socket.o obj/speclang.tab.o obj/lex.yy.o

NEEDED_PURE_OBJS := $(filter-out $(foreach file, ${COLAB_OBJS_FILENAME_ONLY}, $(file)), $(PURE_OBJS))
NEEDED_OBJS := $(patsubst %, $(OBJS_DIR)/%, $(NEEDED_PURE_OBJS))

COLAB_SOURCES_DIRS := $(patsubst %,%/src, $(COLAB_MODULES))
vpath %.C $(COLAB_SOURCES_DIRS)
vpath %.c $(COLAB_SOURCES_DIRS)
vpath %.h $(COLAB_HEADERS_DIRS) framework/parser/generated
COLAB_CFLAGS := $(patsubst %,-I%/include,$(COLAB_MODULES))
"""
        if self.separate_compile is True:
            retStr += \
                """
CUDA_NODE_MODULES := LifeNode
CUDA_EXTENSION_MODULES += $(patsubst %,node/%,$(CUDA_NODE_MODULES))
CUDA_EXTENSION_MODULES := $(patsubst %,extensions/%,$(CUDA_EXTENSION_MODULES))
CUDA_MODULES := $(CUDA_EXTENSION_MODULES)
CUDA_SOURCES_DIRS := $(patsubst %,%/src, $(CUDA_MODULES))
CUDA_CODE := $(foreach dir,$(CUDA_SOURCES_DIRS),$(wildcard $(dir)/CG_*CompCategory.C))

CUDA_SOURCES_FILENAME_ONLY :=$(shell for file in $(notdir $(CUDA_CODE)); do \
	       echo $${file} ; \
	       done)

#CUDA_PURE_OBJS := $(patsubst %.C, %.o, $(CUDA_CODE))
CUDA_PURE_OBJS := $(patsubst %.C, %.o, $(CUDA_SOURCES_FILENAME_ONLY))

CUDA_OBJS := $(patsubst %, $(OBJS_DIR)/%, $(CUDA_PURE_OBJS))
"""
        return retStr

    def getObjectOnlyFlags(self):
        retStr = "#OBJECTONLYFLAGS is flags that only apply to objects, depend.sh generated code.\n"
        retStr += "OBJECTONLYFLAGS :="
        if self.objectMode == "64":
            retStr += " $(MAKE64)"
        if self.options.compiler == "xl":
            if self.options.blueGene is True:
                retStr += " " + XL_BG_RUNTIME_TYPE_INFO_FLAG
                if self.options.blueGeneP is True:
                    retStr += " " + BGP_FLAGS
                if self.options.blueGeneQ is True:
                    retStr += " " + BGQ_FLAGS
            else:
                retStr += " " + XL_RUNTIME_TYPE_INFO_FLAG
        retStr += "\n"
        return retStr

    def getHeaderPaths(self):
        retStr = """
SRC_HEADER_INC := $(patsubst %,-I%/include,$(MODULES)) $(patsubst %,-I%/generated,$(PARSER_PATH)) $(patsubst %,-I%/include,$(SPECIAL_EXTENSION_MODULES))
SRC_HEADER_INC += -I../common/include
        """
        return retStr

    def getNVCCFlags(self):  # noqa
        retStr = \
            """\
OTHER_LIBS := -lgmp \
"""
        retStr += "-I$(MGS_PYTHON_INCLUDE_DIR) -L$(MGS_PYTHON_LIB) -l{} ".format(self.pythonLibName)

        retStr += "\n"
        retStr += \
            """\
OTHER_LIBS_HEADER := -I$(MGS_PYTHON_INCLUDE_DIR)
"""
        retStr += "\n"
        retStr += \
            """\
ifeq ($(USE_SUITESPARSE), 1)
OTHER_LIBS += -I$(SUITESPARSE)/include -L$(SUITESPARSE)/lib -lcxsparse
OTHER_LIBS_HEADER += -I$(SUITESPARSE)/include -DUSE_SUITESPARSE
endif
#LDFLAGS := -shared
"""
        """
# https://github.com/OpenGP/htrack/blob/master/cmake/ConfigureCUDA.cmake
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -w -Xcompiler -fPIC" )
# set(CUDA_NVCC_FLAGS "-gencode arch=compute_50,code=sm_50") # GTX980
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g") # HOST debug mode
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -G") # DEV debug mode
# Tesla : -gencode arch=compute_10,code=sm_10
# Fermi : -gencode arch=compute_20,code=sm_20

#CFLAGS := $(patsubst %,-I%/include,$(MODULES)) $(patsubst %,-I%/generated,$(PARSER_PATH)) $(patsubst %,-I%/include,$(SPECIAL_EXTENSION_MODULES))  -DLINUX -DDISABLE_DYNAMIC_LOADING -DHAVE_MPI
#CFLAGS += -I../common/include -std=c++14 -Wno-deprecated-declarations
#CFLAGS += --compiler-options -fPIC -MMD
#CFLAGS += -g -fno-inline -fno-eliminate-unused-debug-types -DDEBUG_ASSERT -DNOWARNING_DYNAMICCAST -Og -DDISABLE_DYNAMIC_LOADING -DHAVE_MPI -DHAVE_GPU
"""
        retStr += self.getHeaderPaths()

        retStr += \
            """
# SOURCE_AS_CPP := -x c++
SOURCE_AS_CPP := -x cu
CUDA_NVCC_FLAGS := --compiler-options -fPIC -std=c++14 -Xcompiler -Wno-deprecated-declarations \
"""
        retStr += " -DLINUX -DDISABLE_DYNAMIC_LOADING -Xcompiler -DHAVE_MPI -Xcompiler -DHAVE_GPU"
        if self.options.debug_assert is True:
            retStr += " " + DEBUG_ASSERT

        if self.options.debug_hh is True:
            retStr += " " + DEBUG_HH

        if self.options.debug_loops is True:
            retStr += " " + DEBUG_LOOPS

        if self.options.debug_cpts is True:
            retStr += " " + DEBUG_CPTS

        if self.options.nowarning_dynamiccast is True:
            retStr += " " + NOWARNING_DYNAMICCAST

        if self.options.profile == USE:
            retStr += " " + PROFILING_FLAGS

        if self.options.debug == USE:
            retStr += " -g -G -Xcompiler -rdynamic"
            # to add '-pg' run with --profile
        else:
            if self.options.optimization == "O":
                retStr += " " + O_OPTIMIZATION_FLAG
            if self.options.optimization == "O2":
                retStr += " " + O2_OPTIMIZATION_FLAG
            if self.options.optimization == "O3":
                retStr += " " + O3_OPTIMIZATION_FLAG
            if self.options.optimization == "O4":
                retStr += " " + O4_OPTIMIZATION_FLAG
            if self.options.optimization == "O5":
                retStr += " " + O5_OPTIMIZATION_FLAG
            # if self.options.optimization == "Og":
            #    retStr += " " + OG_OPTIMIZATION_FLAG

        Gencode_Tesla = " -gencode arch=compute_10,code=sm_10 "  # NOQA
        Gencode_Fermi = " -gencode arch=compute_20,code=sm_20 "  # NOQA
        Gencode_Kepler_1 = " -gencode arch=compute_30,code=sm_30 "  # NOQA
        Gencode_Kepler_2 = " -gencode arch=compute_35,code=sm_35 "  # NOQA
        Gencode_Kepler_3 = " -gencode arch=compute_35,code=compute_35 "  # NOQA
        """ NVCC 7.5 (Kepler + Maxwell native; Pascal PTX JIT)"""
        Gencode_Maxwell_1 = " -gencode=arch=compute_50,code=sm_50 "  # NOQA
        Gencode_Maxwell_2 = " -gencode=arch=compute_52,code=sm_52 "  # NOQA
        Gencode_Maxwell_3 = " -gencode=arch=compute_52,code=compute_52 "  # NOQA
        Gencode_Kepler = Gencode_Kepler_1 + Gencode_Kepler_2 + Gencode_Kepler_3  # NOQA
        Gencode_Maxwell = Gencode_Maxwell_1 + Gencode_Maxwell_2 + Gencode_Maxwell_3  # NOQA
        Gencode_NVCC7_5 = """ -gencode=arch=compute_30,code=sm_30 \
  -gencode=arch=compute_35,code=sm_35 \
  -gencode=arch=compute_50,code=sm_50 \
  -gencode=arch=compute_52,code=sm_52  \
  -gencode=arch=compute_52,code=compute_52  \
        """  # NOQA
        """ NVCC 8.0 (Maxwell + Pascal native)"""
        Gencode_NVCC8_0 = """  -gencode=arch=compute_30,code=sm_30 \
  -gencode=arch=compute_35,code=sm_35 \
  -gencode=arch=compute_50,code=sm_50 \
  -gencode=arch=compute_52,code=sm_52 \
  -gencode=arch=compute_60,code=sm_60 \
  -gencode=arch=compute_61,code=sm_61 \
  -gencode=arch=compute_61,code=compute_61 \
  """
        """ NVCC 9.0 (Volta native)"""
        Gencode_NVCC9_0 = """ -gencode=arch=compute_50,code=sm_50 \
  -gencode=arch=compute_52,code=sm_52 \
  -gencode=arch=compute_60,code=sm_60 \
  -gencode=arch=compute_61,code=sm_61 \
  -gencode=arch=compute_70,code=sm_70 \
  -gencode=arch=compute_70,code=compute_70 \
 """
        """ NVCC 10.0 (Turing native)"""
        Gencode_NVCC10_0 = """ -gencode=arch=compute_50,code=sm_50 \
  -gencode=arch=compute_52,code=sm_52 \
  -gencode=arch=compute_60,code=sm_60 \
  -gencode=arch=compute_61,code=sm_61 \
  -gencode=arch=compute_70,code=sm_70 \
  -gencode=arch=compute_70,code=compute_70 \
  -gencode=arch=compute_75,code=compute_75 \
 """
        Gencode_Volta = """ -gencode=arch=compute_70,code=sm_70 \
  -gencode=arch=compute_70,code=compute_70 \
 """
        machine_choice = Gencode_Kepler_2
        # machine_choice = Gencode_NVCC8_0
        # machine_choice = Gencode_NVCC9_0
        # machine_choice = Gencode_Volta
        # retStr += machine_choice + " -dc "

        retStr += "\n"

        if not self.separate_compile:
            retStr += """LDFLAGS:= """+ machine_choice
        retStr += """
CUDA_NVCC_FLAGS += $(GENCODE_FLAGS) -dc
CUDA_NVCC_FLAGS += $(SOURCE_AS_CPP)

#--compiler-options -mcpu=power9
# NVCC fails with this --compiler-options -flto
#https://devtalk.nvidia.com/default/topic/1026826/link-time-optimization-with-cuda-on-linux-flto-/?offset=6

CUDA_NVCC_LDFLAGS :=  $(GENCODE_FLAGS) -dlink 
CUDA_NVCC_COMBINED_LDFLAGS :=   $(GENCODE_FLAGS) -lib
"""
        return retStr

    def getCFlags(self):  # noqa
        retStr = \
            """\
OTHER_LIBS :=-lgmp \
"""
        retStr += "-I$(MGS_PYTHON_INCLUDE_DIR) -L$(MGS_PYTHON_LIB) -l{}".format(self.pythonLibName)

        retStr += "\n"
        retStr += \
            """\
OTHER_LIBS_HEADER := -I$(MGS_PYTHON_INCLUDE_DIR)
"""
        retStr += "\n"
        retStr += \
            """\
ifeq ($(USE_SUITESPARSE), 1)
OTHER_LIBS += -I$(SUITESPARSE)/include -L$(SUITESPARSE)/lib -lcxsparse
OTHER_LIBS_HEADER += -I$(SUITESPARSE)/include -DUSE_SUITESPARSE
endif
#LDFLAGS := -shared
"""
        if self.options.withGpu is False:
            retStr += self.getHeaderPaths()
        retStr += \
            """
CFLAGS := -fPIC -std=c++14 -Wno-deprecated-declarations
CFLAGS +=  \
"""
        if self.options.colab is True:
            retStr += \
                """$(COLAB_CFLAGS) \
"""
        if self.options.blueGene is False:
            retStr += \
                """\
-MMD \
"""
        if self.options.compiler == "gcc":
            if self.options.withMpi is True:
                retStr += "-I$(LAMHOME)/include"
            retStr += " " + GNU_WARNING_FLAGS

        if self.operatingSystem == "Linux":
            retStr += " " + LINUX_EXTRA_FLAGS
            if self.options.compiler == "gcc":
                retStr += " -fpic "
        elif self.operatingSystem == "AIX":
            if self.options.compiler == "gcc":
                retStr += " " + AIX_GNU_MINIMAL_TOC_FLAG
            retStr += " " + AIX_EXTRA_FLAGS
        else:
            raise InternalError("Unknown OS " + self.operatingSystem)

        if self.options.debug == USE:
            if self.options.compiler in ["gcc", "g++", "mpicc", "mpiCC"]:
                retStr += " " + GNU_DEBUGGING_FLAG
            elif self.options.compiler == "xl":
                retStr += " " + XL_DEBUGGING_FLAG
            else:
                retStr += " -g -fno-inline -fno-eliminate-unused-debug-types"

        if self.options.debug_assert is True:
            retStr += " " + DEBUG_ASSERT

        if self.options.debug_hh is True:
            retStr += " " + DEBUG_HH

        if self.options.debug_loops is True:
            retStr += " " + DEBUG_LOOPS

        if self.options.debug_cpts is True:
            retStr += " " + DEBUG_CPTS

        if self.options.nowarning_dynamiccast is True:
            retStr += " " + NOWARNING_DYNAMICCAST

        if self.options.profile == USE:
            retStr += " " + PROFILING_FLAGS

        if self.options.optimization == "O":
            retStr += " " + O_OPTIMIZATION_FLAG
        if self.options.optimization == "O2":
            retStr += " " + O2_OPTIMIZATION_FLAG
        if self.options.optimization == "O3":
            retStr += " " + O3_OPTIMIZATION_FLAG
        if self.options.optimization == "O4":
            retStr += " " + O4_OPTIMIZATION_FLAG
        if self.options.optimization == "O5":
            retStr += " " + O5_OPTIMIZATION_FLAG
        if self.options.optimization == "Og":
            retStr += " " + OG_OPTIMIZATION_FLAG

        if self.options.dynamicLoading is False:
            retStr += " -DDISABLE_DYNAMIC_LOADING"

        if self.options.pthreads is False:
            retStr += " -DDISABLE_PTHREADS"

        if self.options.withMpi is True:
            retStr += " -DHAVE_MPI"
            if self.options.compiler == "gcc":
                retStr += " -DLAM_BUILDING"

        if self.options.withGpu is True:
            retStr += " -DHAVE_GPU"

        if self.options.withArma is True:
            retStr += " -DHAVE_ARMA"

        if self.options.profile == USE:
            retStr += " -DPROFILING"

        if self.options.silent is True:
            retStr += " -DSILENT_MODE"

        if self.options.verbose is True:
            retStr += " -DVERBOSE"

        if self.options.compiler == "gcc":
            retStr += " -DWITH_GCC"

        if self.options.blueGene is True:
            retStr += " -DUSING_BLUEGENE"

        if self.options.blueGeneL is True:
            retStr += " -DUSING_BLUEGENEL"
            retStr += " -DMPICH_IGNORE_CXX_SEEK"
            retStr += " -DMPICH_SKIP_MPICXX"

        if self.options.blueGeneP is True:
            retStr += " -DUSING_BLUEGENEP"

        if self.options.blueGeneQ is True:
            retStr += " -DUSING_BLUEGENEQ"

        if self.options.blueGene is True:
            retStr += " $(MPI_INC)"

        retStr += "\n"
        return retStr

    def getLensLibs(self):
        retStr = "LENS_LIBS := "
        if self.options.domainLibrary is True:
            # retStr += "lib/liblensdomain.a "
            retStr += "lib/liblens.a\n"
        return retStr

    def getLibs(self):  # noqa
        retStr = "# add libs"
        if self.options.colab is True:
            if self.options.debug == USE:
                retStr += \
                    """
NTS_LIBS := ${LENSROOT}/lib/libnts_db.so\n

"""
            else:
                retStr += \
                    """
NTS_LIBS := ${LENSROOT}/lib/libnts.so\n

"""
        # retStr += "LENS_LIBS_EXT := $(LENS_LIBS) lib/liblensext.a\n"
        retStr += """
LIBS := """
        if self.options.dynamicLoading is True:
            retStr += "-ldl "
        if self.options.pthreads is True:
            retStr += "-lpthread "
        if self.architecture == "ppc32":
            retStr += "-lmass "
        elif self.architecture == "ppc64":
            retStr += "-lmass "
        else:
            retStr += "-lm "
        if self.operatingSystem == "AIX":
            retStr += "$(LENS_LIBS_EXT) "
        elif self.operatingSystem == "Linux":
            # retStr += "-Wl,--whole-archive "
            retStr += "$(LENS_LIBS_EXT) "
            # retStr += "-Wl,-no-whole-archive "
        else:
            raise InternalError("Unknown OS " + self.operatingSystem)

        if self.options.mpiTrace is True and self.options.blueGene is not True:
            printWarning("MPI Trace profiling not available.")
        elif self.options.mpiTrace is True:
            retStr += " $(MPI_TRACE_LIBS)"

        if self.options.blueGene is True:
            retStr += " $(MPI_LIBS)"

        if self.options.compiler == "gcc":
            if self.options.withMpi is True:
                retStr += " -L$(LAMHOME)/lib -lmpi -llammpi++ -llam -ldl -lpthread"

        if self.options.withGpu is True:
            retStr += " -lcuda -lcudart -lcurand"
            if self.options.withMpi is True:
                retStr += " -lmpi_cxx -lmpi -lopen-pal"

        if self.options.withArma is True:
            retStr += " -llapack -lopenblas -larmadillo"

        if self.options.colab is True:
            if self.options.debug == USE:
                retStr += " -I$(NTI_INC_DIR) -L$(shell pwd)/lib/ -Wl,--rpath=$(shell pwd)/lib/ -lnti_db -lnts_db -lutils_db"
            else:
                retStr += " -I$(NTI_INC_DIR) -L$(shell pwd)/lib/ -Wl,--rpath=$(shell pwd)/lib/ -lnti -lnts -lutils"
        retStr += " $(OTHER_LIBS)"

        if self.options.withArma is True:
            retStr += " -llapack -lopenblas -larmadillo"

        retStr += "\n"
        return retStr

    def getSharedPFix(self):
        retStr = "SHARED_PFIX = "

        if self.operatingSystem == "Linux":
            retStr += LINUX_SHARED_PFIX
        elif self.operatingSystem == "AIX":
            if self.options.compiler == "gcc":
                retStr += " " + AIX_GNU_SHARED_PFIX
            elif self.options.compiler == "xl":
                retStr += " " + AIX_XL_SHARED_PFIX
            else:
                raise InternalError("Unknown compiler " + self.options.compiler)
        else:
            raise InternalError("Unknown OS " + self.operatingSystem)
        retStr += "\n"
        return retStr

    def getSharedCC(self):
        retStr = "SHAREDCC = "

        if self.operatingSystem == "Linux":
            retStr += LINUX_SHARED_CC
        elif self.operatingSystem == "AIX":
            retStr += AIX_SHARED_CC
        else:
            raise InternalError("Unknown OS " + self.operatingSystem)
        retStr += "\n"
        return retStr

    def getFinalTargetFlag(self):
        retStr = "FINAL_TARGET_FLAG = "
        if self.options.debug == USE:
            retStr += " -g"

        if self.operatingSystem == "Linux":
            if self.options.withGpu is True:
                pass
                # retStr += "-Xlinker -rdynamic"
            else:
                retStr += LINUX_FINAL_TARGET_FLAG
        elif self.operatingSystem == "AIX":

            if self.options.dynamicLoading is True:
                retStr += " -Wl,-bnoautoexp -Wl,-bE:$(SO_DIR)/main.def"
            if self.options.compiler == "gcc":
                retStr += " " + AIX_GNU_FINAL_TARGET_FLAG
            elif self.options.compiler == "xl":
                retStr += " " + AIX_XL_FINAL_TARGET_FLAG
            else:
                raise InternalError("Unknown compiler " + self.options.compiler)
        else:
            raise InternalError("Unknown OS " + self.operatingSystem)
        retStr += "\n"
        return retStr

    def getXLinker(self):
        retStr = "XLINKER := "

        if self.operatingSystem == "AIX":
            if self.options.compiler == "gcc":
                retStr += "-Xlinker"
            if self.objectMode == "64":
                retStr += " -bmaxdata:0xF00000000 /usr/gnu/lib/ppc64/libstdc++.a"
            else:
                retStr += " -bmaxdata:0x80000000"
        retStr += "\n"
        return retStr

    def getDXSpecificCode(self):
        retStr = \
            """\
# Macros that are needed to compile DX modules
DX_INCLUDE := framework/dca/include

"""
        retStr += "DX_BASE := " + self.dx.mainPath + "\n"

        if self.operatingSystem == "Linux":
            retStr += "DX_CFLAGS := " + LINUX_DX_CFLAGS + "\n"
            retStr += "DX_LITELIBS := " + LINUX_DX_LITELIBS + "\n"
        elif self.operatingSystem == "AIX":
            retStr += "DX_CFLAGS := " + AIX_DX_CFLAGS + "\n"
            retStr += "DX_LITELIBS := " + AIX_DX_LITELIBS + "\n"
        else:
            raise InternalError("Unknown OS " + self.operatingSystem)

#         retStr += \
# """\
# FILES_EDGESETSUBSCRIBERSOCKET = $(DCA_OBJ)/edgeSetOutboard.o $(DCA_OBJ)/EdgeSetSubscriberSocket.o $(DCA_OBJ)/socket.o
# FILES_NODESETSUBSCRIBERSOCKET = $(DCA_OBJ)/nodeSetOutboard.o $(DCA_OBJ)/NodeSetSubscriberSocket.o $(DCA_OBJ)/socket.o
#
# $(DCA_OBJ)/edgeSetOutboard.o: $(DCA_SRC)/outboard.c
# 	$(C_COMP) $(DX_CFLAGS) -DUSERMODULE=m_EdgeWatchSocket -c $(DX_BASE)/lib/outboard.c -o $(DCA_OBJ)/edgeSetOutboard.o
#
# $(DCA_OBJ)/nodeSetOutboard.o: $(DCA_SRC)/outboard.c
# 	$(C_COMP) $(DX_CFLAGS) -DUSERMODULE=m_NodeWatchSocket -c $(DX_BASE)/lib/outboard.c -o $(DCA_OBJ)/nodeSetOutboard.o
#
# $(DX_DIR)/EdgeSetSubscriberSocket: $(DCA_OBJ)/edgeSetOutboard.o $(DCA_SRC)/EdgeSetSubscriberSocket.c $(DCA_OBJ)/socket.o
# 	$(C_COMP) $(DX_CFLAGS) -c $(DCA_SRC)/EdgeSetSubscriberSocket.c -o $(DCA_OBJ)/EdgeSetSubscriberSocket.o
# 	$(C_COMP) $(FILES_EDGESETSUBSCRIBERSOCKET) $(DX_LITELIBS) -o $(DX_DIR)/EdgeSetSubscriberSocket;
#
# $(DX_DIR)/NodeSetSubscriberSocket: $(DCA_OBJ)/nodeSetOutboard.o $(DCA_SRC)/NodeSetSubscriberSocket.c $(DCA_OBJ)/socket.o
# 	$(C_COMP) $(DX_CFLAGS) -c $(DCA_SRC)/NodeSetSubscriberSocket.c -o $(DCA_OBJ)/NodeSetSubscriberSocket.o
# 	$(C_COMP) $(FILES_NODESETSUBSCRIBERSOCKET) $(DX_LITELIBS) -o $(DX_DIR)/NodeSetSubscriberSocket;
#
# """
        return retStr

    def getSuffixRules(self):
        retStr = \
            """\
# clear all default suffixes
.SUFFIXES:

# Our Suffix rules
#%.d:
#	@$(SCRIPTS_DIR)/depend.sh $(subst /obj,,${@D}) ${*F} $(CFLAGS) > $@

#.C:
#	true

#.h:
#	true
"""
        if self.options.withGpu is True and self.separate_compile is True:
            retStr += """
$(OBJS_DIR)/CG_%CompCategory.o : CG_%CompCategory.C
\t$(NVCC) $(CUDA_NVCC_FLAGS) $(SRC_HEADER_INC) $(OTHER_LIBS_HEADER) """
            if (self.options.asNts is True) or (self.options.asNtsNVU is True):
                retStr += "-I$(NTI_INC_DIR) "
            if (self.options.withGpu is True):
                retStr += """-c $< -o $@
    """
            else:
                retStr += """-c $< -o $@
    """

        retStr += \
"""
$(OBJS_DIR)/%.o : %.C
\t$(CC) $(CFLAGS) $(SRC_HEADER_INC) $(OTHER_LIBS_HEADER) """
        # if (self.options.withGpu is True):
        #     retStr += "$(SOURCE_AS_CPP) "
        if (self.options.asNts is True) or (self.options.asNtsNVU is True):
            retStr += "-I$(NTI_INC_DIR) "
        if (self.options.withGpu is True):
            retStr += """-c $< -o $@
"""
        else:
            retStr += """-c $< -o $@
"""
        return retStr

    def getCreateDFTargets(self):
        retStr = \
            """\
#For the DependFile Tool
DEPENDFILE_OBJS := $(STD_UTILS_OBJ_PATH)/DependLine.o $(STD_UTILS_OBJ_PATH)/DependFile.o
OBJS += $(DEPENDFILE_OBJS)

$(BIN_DIR)/createDF: $(DEPENDFILE_OBJS)
	$(CC) $(MAKE64) -O2 -o $@ $^
#	$(CC) $(CFLAGS) -o $@ $^

"""  # noqa
        return retStr

    def getModuleAndObjectIncludes(self):
        """docstring"""
        retStr = \
            """\
# Include the description of each module.
# This will be in the root/project, where root is the root subdir (E.g. dir1)
# include $(patsubst %,%/module.mk,$(MODULES))

# Include the C include dependencies
-include $(OBJS:.o=.d)

"""
        return retStr

    def getParserTargets(self):
        """docstring"""
        # bisonOutputPrefix = ""
        # if self.bisonVersion == "1.35":
        #     bisonOutputPrefix = "framework/parser/bison/"

        retStr = \
            """\

# Parser targets
speclang.tab.h:
\tcd framework/parser/bison; $(BISON) -v -d speclang.y; \\
\tmv speclang.tab.c ../generated/speclang.tab.C 2>/dev/null; mv speclang.tab.h ../generated

framework/parser/generated/speclang.tab.C: framework/parser/bison/speclang.y
\tcd framework/parser/bison; $(BISON) -v -d speclang.y; \\
\tmv speclang.tab.c ../generated/speclang.tab.C 2>/dev/null; mv speclang.tab.h ../generated

"""
        if self.options.withGpu is True:
            retStr += \
                """
$(OBJS_DIR)/speclang.tab.o: framework/parser/generated/speclang.tab.C framework/parser/bison/speclang.y
\t$(CC) $(SRC_HEADER_INC) -c $< -DYYDEBUG $(CFLAGS) -o $@

"""
        else:
            retStr += \
                """
$(OBJS_DIR)/speclang.tab.o: framework/parser/generated/speclang.tab.C framework/parser/bison/speclang.y
\t$(CC) $(SRC_HEADER_INC) -c $< -DYYDEBUG $(CFLAGS) -o $@

"""

        return retStr

    def getScannerTargets(self):
        """docstring"""
        if self.options.blueGene is False:
            retStr = \
                   """\
# Scanner targets
framework/parser/generated/lex.yy.C: framework/parser/flex/speclang.l
\t$(FLEX) -+ framework/parser/flex/speclang.l
\tsed 's/class istream;/#include <FlexFixer.h>/' \
           lex.yy.cc > lex.yy.cc.edited
\tmv -f lex.yy.cc.edited framework/parser/generated/lex.yy.C
\trm lex.yy.cc
"""
        else:
            retStr = \
                   """\
# Scanner targets
framework/parser/generated/lex.yy.C: framework/parser/flex/speclang.l
\tcp framework/parser/flex/lex.yy.C.linux.i386 framework/parser/generated/lex.yy.C
"""

        if self.options.withGpu is True:
            retStr += \
                """
$(OBJS_DIR)/lex.yy.o: framework/parser/generated/lex.yy.C framework/parser/flex/speclang.l
\t$(CC) -c $< $(CFLAGS) $(SRC_HEADER_INC) -o $@
"""
        else:
            retStr += \
                """
$(OBJS_DIR)/lex.yy.o: framework/parser/generated/lex.yy.C framework/parser/flex/speclang.l
\t$(CC) -c $< $(CFLAGS) $(SRC_HEADER_INC) -o $@
"""
        retStr += "\n"
        return retStr

    def getAllTarget(self):
        retStr = "cleanfirst:\n"
        retStr += "\t-rm $(BIN_DIR)/$(EXE_FILE)\n\n"
        if self.options.colab is True:
            retStr += "final: cleanfirst $(OBJS) $(LENS_LIBS_EXT) "
        else:
            retStr += "final: cleanfirst speclang.tab.h $(OBJS)  $(OBJS_DIR)/speclang.tab.o $(OBJS_DIR)/lex.yy.o $(OBJS_DIR)/socket.o  $(LENS_LIBS_EXT) "
        if self.options.dynamicLoading is True:
            retStr += " $(DEF_SYMBOLS) $(UNDEF_SYMBOLS) $(BIN_DIR)/createDF $(SHARED_OBJECTS) "
        if self.dx.exists is True:
            retStr += " $(DX_DIR)/EdgeSetSubscriberSocket $(DX_DIR)/NodeSetSubscriberSocket "
        retStr += " | $(BIN_DIR)"
        retStr += "\n"

        if self.options.withGpu and self.separate_compile:
            if self.options.colab is True:
                retStr += "\t$(NVCC) $(CUDA_NVCC_LDFLAGS) $(LIBS) $(FINAL_TARGET_FLAG) $(NEEDED_OBJS) "
            else:
                retStr += "\t$(NVCC) -Xlinker -DHAVE_GPU $(CUDA_NVCC_LDFLAGS) $(LIBS) $(FINAL_TARGET_FLAG) $(OBJS_DIR)/speclang.tab.o $(OBJS_DIR)/lex.yy.o $(OBJS_DIR)/socket.o $(OBJS) "
            if self.options.colab is True:
                pass
            else:
                if (self.options.asNts is True) or (self.options.asNtsNVU is True):
                    retStr += "$(NTI_OBJS) "
            retStr += "$(COMMON_OBJS)  -o $(OBJS_DIR)/gpuCode.o"
            retStr += "\n"
        if self.options.colab is True:
            retStr += "\t$(CC) $(FINAL_TARGET_FLAG) $(NEEDED_OBJS) "
        else:
            retStr += "\t$(CC) $(FINAL_TARGET_FLAG) $(OBJS_DIR)/speclang.tab.o $(OBJS_DIR)/lex.yy.o $(OBJS_DIR)/socket.o $(OBJS) "
        # retStr = "final: $(BASE_OBJECTS) $(LENS_LIBS_EXT) $(BIN_DIR)/$(EXE_FILE) $(DCA_OBJ)/socket.o $(OBJS) $(MODULE_MKS)"
        retStr += "$(COMMON_OBJS) "
        if self.options.withGpu and self.separate_compile:
            retStr += "$(OBJS_DIR)/gpuCode.o "
        if self.options.colab is True:
            pass
        else:
            if (self.options.asNts is True) or (self.options.asNtsNVU is True):
                retStr += "$(NTI_OBJS) "
        retStr += "$(LDFLAGS) $(LIBS) -o $(BIN_DIR)/$(EXE_FILE) "
        return retStr

    def getDependfileTarget(self):
        retStr = "$(SO_DIR)/Dependfile: $(DEF_SYMBOLS) $(UNDEF_SYMBOLS) $(BIN_DIR)/createDF\n"
        if self.options.extMode is True:
            retStr += "\techo DependFile not generated, working in EXT_MODE"
        else:
            retStr += "\t$(BIN_DIR)/createDF $(SO_DIR) > $@"
        retStr += "\n"
        return retStr

    def getLibsTarget(self):
        retStr = ""
        if self.options.colab is True:
            retStr += \
                """
library: speclang.tab.h  $(COLAB_OBJS)
\t$(CC) $(LDFLAGS) -o ${NTS_LIBS} ${COLAB_OBJS}
"""
        return retStr

    def getLibLensTarget(self):
        retStr = \
            """\
# I could not find a way to ar it in one time in AIX, the command gets too
# long, also I cannot do this with the foreach loop using BASE_MODULES because
# it concatenates into one single line.
"""
        retStr += "lib/liblens.a: $(BASE_OBJECTS)\n"
        moduleList = ["parser", "dca", "dataitems", "factories", "networks",
                      "simulation", "functors", "std", "img"]
        if self.options.withMpi is True:
            moduleList.append("streams")

        extraToken = ""
        if self.objectMode == "64":
            extraToken = " -X32_64"
        arCmd = "\tar" + extraToken + " rvu $@"
        for i in moduleList:
            retStr += arCmd + " $(OBJ_" + i + ")\n"
        retStr += "\tranlib $@\n\n"
        retStr += "lib/liblensext.a: $(EXTENSION_OBJECTS) $(LENS_LIBS)"
        if self.options.dynamicLoading is False:
            retStr += " $(GENERATED_DL_OBJECTS)"
        retStr += "\n"
        retStr += arCmd + " $(EXTENSION_OBJECTS)\n"
        if self.options.dynamicLoading is False:
            retStr += arCmd + " $(GENERATED_DL_OBJECTS)\n"
        retStr += "\tranlib $@\n\n"
        return retStr

    def getMainDefTarget(self):
        retStr = \
            """\
$(SO_DIR)/main.def: $(LENS_LIBS_EXT)
\techo \#\!. > $@
\t$(SCRIPTS_DIR)/gen_def.sh $^ >> $@
"""
        return retStr

    def getLensparserTarget(self):
        retStr = "$(BIN_DIR)/$(EXE_FILE): cleanfirst "

        if self.operatingSystem == "AIX":
            retStr += LENSPARSER_TARGETS
            if self.options.dynamicLoading is True:
                retStr += "$(SO_DIR)/main.def "
            else:
                retStr += "$(LENS_LIBS_EXT) "
        elif self.operatingSystem == "Linux":
            retStr += LENSPARSER_TARGETS + "$(LENS_LIBS_EXT) "
        else:
            raise InternalError("Unknown OS " + self.operatingSystem)
        if self.options.dynamicLoading is True:
            retStr += " $(SHARED_OBJECTS)"
        retStr += " | $(BIN_DIR)"
        retStr += "\n"
        retStr += "\t$(CC) $(FINAL_TARGET_FLAG) $(NEEDED_OBJS) $(COMMON_OBJS) $(OBJS_DIR)/speclang.tab.o $(OBJS_DIR)/lex.yy.o $(OBJS_DIR)/socket.o "
        retStr += "$(LIBS) -o $@"
        if self.options.tvMemDebug is True and self.operatingSystem != "AIX":
            printWarning("Totalview memory debugging not available.")
        elif self.options.tvMemDebug is True:
            if self.objectMode == "64":
                retStr += "-L$(TOTALVIEW_LIBPATH) -L$(TOTALVIEW_LIBPATH) $(TOTALVIEW_LIBPATH)/aix_malloctype64_5.o "
            else:
                retStr += "-L$(TOTALVIEW_LIBPATH) -L$(TOTALVIEW_LIBPATH) $(TOTALVIEW_LIBPATH)/aix_malloctype.o "
        if self.dx.exists is True:
                retStr += "$(DX_LITELIBS)"
        if self.operatingSystem == "AIX":
            retStr += "$(XLINKER)"
        if self.options.profile == USE:
            retStr += PROFILING_FLAGS
            retStr += "$(CFLAGS) -o $(BIN_DIR)/$(EXE_FILE)\n"
        return retStr

    def getSocketTarget(self):
        retStr = "$(OBJS_DIR)/socket.o: "

        if self.dx.exists is True:
            retStr += "socket.c\n"
        else:
            retStr += "fakesocket.c\n"

        retStr += "\t$(C_COMP)"
        if self.objectMode == "64":
            retStr += " $(MAKE64)"
        if self.dx.exists is True:
            retStr += " $(DX_CFLAGS) -c $< "
        else:
            retStr += " -c $<"
        retStr += " -o $@\n"
        return retStr

    def getCleanTarget(self):
        retStr = \
            """\
.PHONY: clean
clean:
\t-rm -f dx/EdgeSetSubscriberSocket
\t-rm -f dx/NodeSetSubscriberSocket
\t-rm -f $(BIN_DIR)/$(EXE_FILE)
\t-rm -f $(BIN_DIR)/createDF
\t-rm -f lib/liblens.a
\t-rm -f lib/liblensext.a
\t-rm -f $(SO_DIR)/Dependfile
\t-rm -f framework/parser/bison/speclang.output
\t-rm -f framework/parser/generated/lex.yy.C
\t-rm -f framework/parser/generated/lex.yy.C
\t-rm -f framework/parser/generated/speclang.tab.C
\t-rm -f framework/parser/generated/speclang.tab.h
\t-rm -f $(OBJS_DIR)/*
"""
        return retStr

    def getPythonLibName(self):
        import os
        import fnmatch

        def find(pattern, path):
            result = []
            for roots, dirs, files in os.walk(path):
                for name in files:
                    if fnmatch.fnmatch(name, pattern):
                        result.append(name)
            return result
        # NOTE: must match the one requested from set_env script
        libs_found = find("libpython3.6*.so", os.environ["MGS_PYTHON_LIB"])
        if not libs_found:
            self.pythonLibName = "python2.7"
        else:
            self.pythonLibName = libs_found[0][:-3][3:]

    def generateMakefile(self, fileName):
        fileBody = self.getInitialValues()
        fileBody += "\n"
        fileBody += self.getMake64()
        fileBody += "\n"
        fileBody += self.getModuleDefinitions()

        fileBody += self.getObjectOnlyFlags()

        self.getPythonLibName()
        if self.options.withGpu is True:
            fileBody += self.getNVCCFlags()
            if self.separate_compile:
                fileBody += self.getCFlags()
            else:
                fileBody += \
                    """\
CFLAGS := ${CUDA_NVCC_FLAGS}
"""
#                fileBody += """\
# CFLAGS := --compiler-bindir  {0}
# """.format(findFile("g++", True))

        else:
            fileBody += self.getCFlags()
        fileBody += "\n"
        # fileBody += self.getLensLibs()
        fileBody += self.getLibs()
        fileBody += "\n"

        if self.options.dynamicLoading is True:
            fileBody += self.getSharedPFix()
            fileBody += self.getSharedCC()

        fileBody += self.getFinalTargetFlag()
        fileBody += self.getXLinker()
        fileBody += "\n"
        fileBody += self.getSuffixRules()
        # getCreateDFTargets() has to be before getModuleAndObjectIncludes()
        if self.options.dynamicLoading is True:
            fileBody += self.getCreateDFTargets()
        fileBody += self.getModuleAndObjectIncludes()

        if self.dx.exists is True:
            fileBody += self.getDXSpecificCode()

        fileBody += self.getParserTargets()
        fileBody += self.getScannerTargets()
        fileBody += self.getAllTarget()
        fileBody += "\n"
        if self.options.dynamicLoading is True:
            fileBody += self.getDependfileTarget()
            fileBody += "\n"
        # fileBody += self.getLibLensTarget()
        fileBody += self.getLibsTarget()
        fileBody += "\n"
        if self.options.dynamicLoading is True:
            fileBody += self.getMainDefTarget()
            fileBody += "\n"
        fileBody += self.getLensparserTarget()
        fileBody += "\n"
        fileBody += self.getSocketTarget()
        fileBody += "\n"
        fileBody += self.getCleanTarget()

        f = open(fileName, "w")
        f.write(fileBody)
        f.close()


def prereq_packages():
    """
    this function install pre-requisite packages
    """
    # import pip

    required_pkgs = ['builtins']

    for package in required_pkgs:
        try:
            __import__(package)
        except ImportError:
            print("Please install: sudo pip install --user %s" % (package))


if __name__ == "__main__":
    prereq_packages()
    try:
        buildSetup = BuildSetup()
        buildSetup.main()
    except FatalError as error:
        error.printError()
        sys.exit(-1)
