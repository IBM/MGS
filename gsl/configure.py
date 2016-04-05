#!/usr/bin/python

import os
import sys
import popen2
#import string
import getopt
import os.path

# If we decide to put modules in the scripts directory use the following to
# be able to import modules
# sys.path.append("scripts/")

# Constants
USE = 1
DONTUSE = 0
UNDEF = -1

PROJECT_NAME= "Lens"
CONFIG_HEADER = "framework/factories/include/LensRootConfig.h"
EXTENSIONS_MK = "./Extensions.mk"
CLEAN_SCRIPT = "./clean.sh"
CONFIG_LOG = "./config.log"
TAB = "   "

AIX_GNU_CPP_COMPILER = "g++"  #-3.2.3"
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

GNU_DEBUGGING_FLAG = "-ggdb"
XL_DEBUGGING_FLAG = "-g"

PROFILING_FLAGS = "-pg"

O_OPTIMIZATION_FLAG = "-O"
O2_OPTIMIZATION_FLAG = "-O2"
O3_OPTIMIZATION_FLAG = "-O3"
O4_OPTIMIZATION_FLAG = "-O4"
O5_OPTIMIZATION_FLAG = "-O5"

DEBUG_ASSERT = "-DDEBUG_ASSERT"
DEBUG_HH = "-DDEBUG_HH"
DEBUG_LOOPS = "-DDEBUG_LOOPS"

COMMON_DX_CFLAGS = " -I . -I$(DX_INCLUDE) -I$(DX_BASE)/include $(MAKE64)"
AIX_DX_CFLAGS = "-O -Dibm6000 " + COMMON_DX_CFLAGS
LINUX_DX_CFLAGS = " -O -Dlinux " + COMMON_DX_CFLAGS

COMMON_DX_LITELIBS = "-lDXlite -lm"
AIX_DX_LITELIBS = "-L$(DX_BASE)/lib_ibm6000 " + COMMON_DX_LITELIBS
LINUX_DX_LITELIBS = "-L$(DX_BASE)/lib_linux " + COMMON_DX_LITELIBS

EXTRA_PARSER_TARGERS_FOR_DX = "$(DX_DIR)/EdgeSetSubscriberSocket $(DX_DIR)/NodeSetSubscriberSocket $(DCA_OBJ)/socket.o"

LENSPARSER_TARGETS = "$(PARSER_GENERATED)/speclang.tab.o $(PARSER_GENERATED)/lex.yy.o $(DCA_OBJ)/socket.o $(BASE_OBJECTS) "

# Pre python 2.3 compatibility
True = 1
False = 0

class FatalError(Exception):
    def __init__(self, value = ""):
        self.value = value
    def __str__(self):
        return repr(self.value)
    def printError(self):
        if self.value != "":
            print "Fatal error:", error

class InternalError(Exception):
    def __init__(self, value = ""):
        self.value = value
    def __str__(self):
        return repr(self.value)

def printWarning(warning):
    print "Warning:", warning, "\n"

def printFeedback(feedback, extraLine = False):
    print feedback
    if extraLine == True:
        print

def findFile(name, required = False):
    cmd = "which " + name
    (stdoutFile, stdinFile, stderrFile) = popen2.popen3(cmd)
    stdout = stdoutFile.read()
    stderr = stderrFile.read()
    if stderr != "" or stdout.find(" ") != -1:
        if required == True:
            raise FatalError("Required file " + name + " could not be found.")
        return ""
    # strip the \n at the end
    stdout = stdout.rstrip()
    return stdout

def getFileStatus(name):
    retStr = ""
    if name != "":
        retStr = os.path.basename(name) + " is found at: " + os.path.dirname(name)
    return retStr

def addUnderScore(name, scoreChar = '='):
    retStr = name + "\n"
    if name != "":
        for i in xrange(len(name)):
            retStr += scoreChar
        retStr += "\n"
    return retStr

def getFirst(a):
    (first, second) = a
    return first

def createConfigHeader():
    rootDir = os.getcwd()
    create = True
    try:
        current = open(CONFIG_HEADER, "r")
        lines = current.readlines()
        line = lines[2]
        begin = line.find('"') + 1
        end = line.find('"', begin)
        if not (begin == -1 or end == -1):
            currentDir =  line[begin:end]
            if currentDir == rootDir:
                create = False
    except:
        pass

    if create == True:
        cmd =  "#include <string>\n\n"
        cmd += "const std::string LENSROOT = \"" + rootDir + "\";\n"
        header = open(CONFIG_HEADER, "w")
        header.write(cmd)
        header.close()
    else:
        printFeedback(CONFIG_HEADER + " looks up to date, not overwriting", True)

def touchExtensionsMk():
    if os.path.isfile(EXTENSIONS_MK) == False:
        f = open(EXTENSIONS_MK, "w")
        f.close()

class DxInfo:

    def __init__(self):

        self.bin      = ""
        self.exists   = False
        self.binPath  = ""
        self.mainPath = ""
        self.liteLib  = ""
        self.include  = ""

    def setup(self, operatingSystem):
        self.bin = findFile("dx")
        if self.bin == "":
            print "DX is not found", self.mainPath
            return
        self.binPath = os.path.dirname(self.bin)
        self.mainPath = os.path.dirname(self.binPath) + "/dx"
        if os.path.isdir(self.mainPath) != True:
            print "DX main path could not be found at", self.mainPath
            return

        self.liteLib = self.mainPath
        if (operatingSystem == "AIX"):
            self.liteLib += "/lib_ibm6000/"
        else:
            self.liteLib += "/lib_linux/"

        self.liteLib += "libDXlite.a"
        if os.path.isfile(self.liteLib) != True:
            print "libDXlite.a could not be found at", self.liteLib
            return

        self.include =  self.mainPath + "/include/dx/dx.h"
        if os.path.isfile(self.include) != True:
            print "dx.h could not be found at", self.include
            return

        self.exists = True

    def getInfo(self):
        retStr = TAB + "DX executable is found at" + self.bin + "\n"
        retStr += TAB + "DX main path is found at" + self.mainPath + "\n"
        retStr += TAB + "DX library is found at" + self.liteLib + "\n"
        retStr += TAB + "DX include file is found at" + self.include + "\n"
        return retStr

class Options:

    def __init__(self, argv):

        # options
        self.compilationMode = "undefined" # 32, 64, undefined
        self.withDX = UNDEF # USE, DONTUSE, UNDEF
        self.compiler = "undefined" # gcc, xl, undefined
        self.silent = False # True, False
        self.verbose = False # True, False
        self.extMode = False  # True, False
        self.debug = DONTUSE # USE, DONTUSE, UNDEF
        self.debug_assert = False # True, False
        self.debug_hh = False # True, False
        self.debug_loops = False # True, False
        self.profile = DONTUSE # USE, DONTUSE, UNDEF
        self.tvMemDebug = DONTUSE # USE, DONTUSE, UNDEF
        self.mpiTrace = DONTUSE # USE, DONTUSE, UNDEF
        self.optimization = "undefined" # O1, O2, O3, undefined
        self.dynamicLoading = False # True, False
        self.domainLibrary = False # True, False
        self.pthreads = True # True, False
        self.withMpi = False # True, False
        self.blueGeneL = False # True, False
        self.blueGeneP = False # True, False
        self.blueGeneQ = False # True, False
        self.blueGene = False # True, False
        self.rebuild = False # True, False
        self.help = False # True, False

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
                           ("debug", "compile with debugging flags"),
                           ("debug_assert", "compile with debugging flags for assert"),
                           ("debug_hh", "compile with debugging flags for Hodgkin-Huxley compartments"),
                           ("debug_loops", "compile with debugging flags for methods to be called iteratively (time loop)"),
                           ("profile", "compile with profile flags"),
                           ("tvMemDebug", "enable totalview memory debugging for parallel jobs (perfomance impact)"),
                           ("mpiTrace", "enable mpiTrace profiling (for BG)"),
                           ("enable-dl", "enable dynamic loading, else everything is statically linked"),
                           ("domainLib", "link to domain specific library"),
                           ("disable-pthreads", "disable pthreads, there will be a single thread"),
                           ("with-mpi", "enables mpi"),
                           ("blueGeneL", "configures for blueGeneL environment"),
                           ("blueGeneP", "configures for blueGeneP environment"),
                           ("blueGeneQ", "configures for blueGeneQ environment"),
                           ("rebuild", "rebuilds the project"),
                           ("help", "displays the available options")]

        self.parseOptions(argv)

    def getOptionList(self):
        return map(getFirst, self.cmdOptions)

    def usage(self, argv):
        print addUnderScore("Possible command line options:"),
        for i in self.cmdOptions:
            str = TAB + "--" + i[0] + ": " + i[1]
            print str

    def parseOptions(self, argv):
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
                        raise FatalError("O, O2, O3, O4, and/or O5 are used at the same time.")
                if o == "--O2":
                    if self.optimization == "undefined":
                        self.optimization = "O2"
                    else:
                        raise FatalError("O, O2, O3, O4, and/or O5 are used at the same time.")
                if o == "--O3":
                    if self.optimization == "undefined":
                        self.optimization = "O3"
                    else:
                        raise FatalError("O, O2, O3, O4, and/or O5 are used at the same time.")
                if o == "--O4":
                    if self.optimization == "undefined":
                        self.optimization = "O4"
                    else:
                        raise FatalError("O, O2, O3, O4, and/or O5 are used at the same time.")
                if o == "--O5":
                    if self.optimization == "undefined":
                        self.optimization = "O5"
                    else:
                        raise FatalError("O, O2, O3, O4, and/or O5 are used at the same time.")
                if o == "--debug":
                    self.debug = USE
                if o == "--debug_assert":
                    self.debug_assert = True
                if o == "--debug_hh":
                    self.debug_hh = True
                if o == "--debug_loops":
                    self.debug_loops = True
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
                if o == "--help":
                    self.help = True

            # Just display help and return
            if self.help == True:
                self.usage(argv)
                return

            if self.extMode == True and self.rebuild == True:
                raise FatalError("Do not rebuild with the ext-mode turned on.")

            if self.debug == USE and self.optimization != "O":
                printWarning("Debugging is turned on even though optimization is " +
                             self.optimization + ".\nSetting optimization to --O")
                self.optimization = "O"

            if self.profile == USE and self.optimization != "O":
                printWarning("Profiling is turned on even though optimization is " +
                             self.optimization + ".\nSetting optimization to --O")
                self.optimization = "O"

            #if self.profile == USE and self.debug != USE:
            #    printWarning("Profiling is turned on so debugging turned on by default.")
            #    self.debug = USE

            if self.debug != USE and self.profile != USE and self.optimization == "undefined":
                printFeedback(
                    "Debugging, profiling, and optimization are not defined, choosing O3 as default.")
                print
                self.optimization = "O3"

            if self.withMpi == True and self.dynamicLoading == True:
                printFeedback("Dynamic loading will be disabled due to existence of MPI.")
                self.dynamicLoading = False

            if self.blueGeneL == True:
                self.blueGene = True
                self.withMpi = True
                self.pthreads = False
                self.dynamicLoading = False
                self.withDX = DONTUSE
                self.compilationMode = "32"
                self.silent = True
                self.optimization = "O3"

            if self.blueGeneP == True:
                self.blueGene = True
                self.withMpi = True
                self.pthreads = True
                self.dynamicLoading = False
                self.withDX = DONTUSE
                self.compilationMode = "32"
                self.silent = True
                self.optimization = "O3"

            if self.blueGeneQ == True:
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
            self.usage(argv)
            raise FatalError

class BuildSetup:

    def __init__(self):

        # System defining variables
        self.operatingSystem = ""
        self.hostName        = ""
        self.architecture    = ""
        self.objectMode      = ""
        self.numCPUs         = os.sysconf("SC_NPROCESSORS_ONLN")

        # Binaries
        self.bisonBin   = ""
        self.flexBin    = ""
        self.grepBin    = ""
        self.makeBin    = ""

        # versions
        self.bisonVersion = ""

        # DX Variable
        self.dx = ""

        # Compilers
        self.cCompiler = ""
        self.cppCompiler = ""

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
            self.objectMode = "32"
        else:
            raise FatalError(PROJECT_NAME + " only runs in AIX or Linux")

        self.bisonBin = findFile("bison", True)
        self.flexBin = findFile("flex", True)
        self.grepBin = findFile("grep", True)
        self.makeBin = findFile("make", True)

    def getSystemInfo(self):
        retStr = addUnderScore("System information:")
        retStr += TAB + "Operating system: " + self.operatingSystem + "\n"
        retStr += TAB + "HostName: " + self.hostName + "\n"
        retStr += TAB + "Architecture: " + self.architecture + "\n"
        retStr += TAB + "Object mode: " + self.objectMode + "\n"
        retStr += TAB + "Number of CPUs: " + str(self.numCPUs) + "\n"
        retStr += TAB + "C compiler: " + self.cCompiler + "\n"
        retStr += TAB + "C++ compiler: " + self.cppCompiler + "\n"

        retStr += TAB + "Silent mode: "
        if self.options.silent == True:
            retStr += "On"
        else :
            retStr += "Off"
        retStr += "\n"

        retStr += TAB + "Verbose mode: "
        if self.options.verbose == True:
            retStr += "On"
        else :
            retStr += "Off"
        retStr += "\n"

        retStr += TAB + "Extensions mode: "
        if self.options.extMode == True:
            retStr += "On"
        else :
            retStr += "Off"
        retStr += "\n"

        retStr += TAB + "Debugging: "
        if self.options.debug == True:
            retStr += "On"
        else :
            retStr += "Off"
        retStr += "\n"

        retStr += TAB + "Profiling: "
        if self.options.profile == USE:
            retStr += "On"
        else :
            retStr += "Off"
        retStr += "\n"

        retStr += TAB + "Totalview Memory Debugging: "
        if self.options.tvMemDebug == True:
            retStr += "On"
        else :
            retStr += "Off"
        retStr += "\n"

        retStr += TAB + "MPI Trace Profiling: "
        if self.options.mpiTrace == True:
            retStr += "On"
        else :
            retStr += "Off"
        retStr += "\n"

        if self.options.optimization != "undefined":
            retStr += TAB + "Optimization level: " + self.options.optimization + "\n"

        retStr += TAB + "Dynamic loading: "
        if self.options.dynamicLoading == True:
            retStr += "Enabled"
        else :
            retStr += "Disabled"
        retStr += "\n"

        retStr += TAB + "Domain specific library: "
        if self.options.domainLibrary == True:
            retStr += "Enabled"
        else :
            retStr += "Disabled"
        retStr += "\n"

        retStr += TAB + "Pthreads: "
        if self.options.pthreads == True:
            retStr += "Enabled"
        else :
            retStr += "Disabled"
        retStr += "\n"

        retStr += TAB + "MPI: "
        if self.options.withMpi == True:
            retStr += "Used"
        else :
            retStr += "Not used"
        retStr += "\n"

        retStr += TAB + "DX: "

        if self.dx.exists == True:
            retStr += "Used\n"
        else:
            retStr += "Not used\n"

        retStr += "\n"
        retStr += addUnderScore("Absolute paths for tools and packages that are going to be used:")
        retStr += TAB + getFileStatus(self.bisonBin) + "\n"
        retStr += TAB + getFileStatus(self.flexBin) + "\n"
        retStr += TAB + getFileStatus(self.grepBin) + "\n"
        retStr += TAB + getFileStatus(self.makeBin) + "\n"

        if self.dx.exists == True:
            retStr += self.dx.getInfo()

        return retStr

    def createLog(self):
        fStr = "Configure script is run as follows:\n"
        for i in sys.argv:
            fStr += i + " "

        fStr += "\n\n"

        fStr += self.getSystemInfo()

        f = open(CONFIG_LOG, "w")
        f.write(fStr)
        f.close()

    def bisonVersionFinder(self):
        cmd = self.bisonBin + " --version"
        (stdoutFile, stdinFile, stderrFile) = popen2.popen3(cmd)
        stderr = stderrFile.read()
        if stderr != "":
            raise FatalError(self.bisonBin + " has an error for command " + cmd)
        stdout = stdoutFile.readline()
        if stdout.find("GNU") == -1:
            raise FatalError(self.bisonBin + " is not GNU")
        tokens = stdout.split()
        self.bisonVersion = tokens[-1]

    def main(self):
        if self.options.help == True:
            return

        self.setCompilers()
        self.setDX()
        self.bisonVersionFinder()
        print self.getSystemInfo()

        ## At this point if there has been no FatalError, the environment
        ## and the options are ok, so create the config log before doing
        ## any more work.
        self.createLog()

        if self.options.blueGene == True or self.operatingSystem == "Linux":
            os.system("mv -f framework/networks/include/pthread.h framework/networks/include/pthread.h.bak > /dev/null 2>&1")

        if self.options.rebuild == True:
            #print "Cleaning the project using:", CLEAN_SCRIPT, "\n"
            #os.system(CLEAN_SCRIPT)
            os.system("make LINUX clean")

        createConfigHeader()
        touchExtensionsMk()

        #self.generateMakefile("Makefile_NTS")
        self.generateMakefile("Makefile")

        if self.options.rebuild == True:
            cmd = "make -j " + str(self.numCPUs)
            print "Starting the make process with:", cmd
            os.system(cmd)

    def setDX(self):
        self.dx = DxInfo()

        if self.objectMode == "64":
            if self.options.withDX == USE:
                raise FatalError("DX can not be enabled in 64 bit object mode")
        else:
            if self.options.withDX == UNDEF:
#                printFeedback(
#                    "Usage of dx not specified, checking if it exists.")
#                self.dx.setup(self.operatingSystem)
#                if self.dx.exists == True:
#                    printFeedback("DX found, using...\n")
                self.options.withDX = DONTUSE
            elif self.options.withDX == USE:
                self.dx.setup(self.operatingSystem)
                if self.dx.exists == False:
                    raise FatalError("DX requested but not found.")

    def setCompilers(self):

        if self.options.withMpi == True:
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
                if self.options.blueGeneL == True:
                   self.options.compiler = "xl"
                   self.cCompiler = findFile("/opt/ibmcmp/vac/bg/9.0/bin/blrts_xlc", True)
                   self.cppCompiler = findFile("/opt/ibmcmp/vacpp/bg/9.0/bin/blrts_xlC", True)
                else:
                   if self.options.blueGeneP == True:
                      self.options.compiler = "xl"
                      self.cCompiler = findFile("mpixlc_r", True)
                      self.cppCompiler = findFile("mpixlcxx_r", True)
                   else:
                      if self.options.blueGeneQ == True:
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
                   #raise FatalError("Currently MPI is only used by AIX")
            return # important do not continue this function after here.

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
        else: # error condition
            raise InternalError("Should not have hit here in setCompilers")

    def getInitialValues(self):
        retStr = \
"""\
# This is a code generated file, generated using configure.py
# To change any options please rerun ./configure.py with the desired options.
# To build the project execute make -j <number of processes>
.DEFAULT_GOAL:= final

BIN_DIR?=./bin
EXE_FILE=gslparser

NTI_DIR=../nti
NTI_OBJ_DIR=$(NTI_DIR)/obj
NTI_INC_DIR=$(NTI_DIR)/include

BISON=$(shell which bison)
FLEX=$(shell which flex)

LENSROOT = $(shell pwd)
OPERATING_SYSTEM = $(shell uname)

SCRIPTS_DIR := scripts
DX_DIR := dx
SO_DIR := $(LENSROOT)/so
PARSER_PATH := framework/parser
PARSER_GENERATED := $(PARSER_PATH)/generated
STD_UTILS_OBJ_PATH := utils/std/obj
TOTALVIEW_LIBPATH := /opt/toolworks/totalview.8.4.1-7/rs6000/lib

"""
#DCA_OBJ := framework/dca/obj
#DCA_SRC := framework/dca/src

# Each module adds to these initial empty definitions
#SRC :=
#OBJS :=
#SHARED_OBJECTS :=
#BASE_OBJECTS :=
#DEF_SYMBOLS := $(SO_DIR)/main.def
#UNDEF_SYMBOLS :=
#
## EXTENSION_OBJECTS is the generated modules that must be linked in statically
#EXTENSION_OBJECTS :=
#
## GENERATED_DL_OBJECTS will be added to liblens if dynamic loading is disabled.
#GENERATED_DL_OBJECTS :=

        # BlueGene MPI flags
        if self.options.blueGeneL == True:
            retStr += \
"""\
BGL_ROOT=/bgl/BlueLight/ppcfloor
MPI_LIBS = -L$(BGL_ROOT)/bglsys/lib -lmpich.rts -lmsglayer.rts -lrts.rts -ldevices.rts
MPI_TRACE_LIBS = /bgl/local/lib/libmpitrace.a
MPI_INC = -I$(BGL_ROOT)/bglsys/include

"""
        if self.options.blueGeneP == True:
            retStr += \
"""\
BGP_ROOT=/bgsys/drivers/ppcfloor
MPI_INC = -I$(BGP_ROOT)/arch/include

"""

        retStr += "CC := " + self.cppCompiler + "\n"
        retStr += "C_COMP := " + self.cCompiler + "\n"
        if self.options.withMpi == True:
            retStr += "HAVE_MPI := 1\n"
        if self.options.silent == True:
            retStr += "SILENT_MODE := 1\n"
        if self.options.verbose == True:
            retStr += "VERBOSE := 1\n"
        if self.options.blueGene == True:
            retStr += "USING_BLUEGENE := 1\n"
        if self.options.pthreads == True:
            retStr += "HAVE_PTHREADS := 1\n"
        if self.options.profile == USE:
            retStr += "PROFILING := 1\n"
        return retStr

    def getMake64(self):
        retStr = "MAKE64 = "
        if self.objectMode == "64":
            if self.options.compiler == "gcc":
                retStr += AIX_GNU_64BIT_FLAG
            elif self.options.compiler == "xl":
                retStr += AIX_XL_64BIT_FLAG
            else:
                raise InternalError("Compiler " + self.options.compiler + " is not found")
        retStr += "\n"
        return retStr

    def getModuleDefinitions(self):
        retStr = \
"""\
####################################################################
# Name of all submodules we want to build
# 1. modules from the GSL frameworks
# 2. modules from the extension (i.e. user-defined)
#
# part 1 --> which include framework/...
#               utils/...
FRAMEWORK_MODULES := dca \\
		dataitems \\
		factories \\
		networks \\
		parser \\
		simulation \\
	 	functors \\

UTILS_MODULES := std \\
		img \\
"""
        if self.options.withMpi == True:
            retStr += \
"""\
		streams \\

"""

        retStr += \
"""\

BASE_MODULES := $(patsubst %,framework/%,$(FRAMEWORK_MODULES))
BASE_MODULES += $(patsubst %,utils/%,$(UTILS_MODULES))

CONSTANT_MODULES :=

EDGE_MODULES :=

INTERFACE_MODULES :=

NODE_MODULES :=

STRUCT_MODULES := CoordsStruct \

TRIGGER_MODULES :=

VARIABLE_MODULES := BasicNodeSetVariable \\
       	NodeSetSPMVariable \\

FUNCTOR_MODULES := BinomialDist \\
        CombineNVPairs \\
        ConnectNodeSetsFunctor \\
       	DstDimensionConstrainedSampler \\
       	DstRefDistanceModifier \\
       	DstRefGaussianWeightModifier \\
       	DstRefSumRsqrdInvWeightModifier \\
       	DstScaledContractedGaussianWeightModifier \\
       	DstScaledGaussianWeightModifier \\
       	IsoSampler \\
        ModifyParameterSet \\
        NameReturnValue \\
      	PolyConnectorFunctor \\
      	RefAngleModifier \\
      	RefDistanceModifier \\
      	ReversedDstRefGaussianWeightModifier \\
      	ReversedSrcRefGaussianWeightModifier \\
      	ReverseFunctor \\
      	ServiceConnectorFunctor \\
      	SrcDimensionConstrainedSampler \\
      	SrcRefDistanceModifier \\
      	SrcRefGaussianWeightModifier \\
      	SrcRefPeakedWeightModifier \\
      	SrcRefSumRsqrdInvWeightModifier \\
      	SrcScaledContractedGaussianWeightModifier \\
      	SrcScaledGaussianWeightModifier \\
      	TissueConnectorFunctor \\
      	TissueFunctor \\
      	TissueLayoutFunctor \\
      	TissueNodeInitFunctor \\
      	TissueProbeFunctor \\
        UniformDiscreteDist \\
        GetNodeCoordFunctor \\

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

MYOBJS :=$(shell for file in $(notdir $(MYSOURCES)); do \\
	       echo $${file} ; \\
	       done)
PURE_OBJS := $(patsubst %.C, %.o, $(MYOBJS))

NTI_OBJS := $(foreach dir,$(NTI_OBJ_DIR),$(wildcard $(dir)/*.o))
TEMP := $(filter-out $(NTI_OBJ_DIR)/neuroGen.o $(NTI_OBJ_DIR)/neuroDev.o $(NTI_OBJ_DIR)/touchDetect.o, $(NTI_OBJS))
NTI_OBJS := $(TEMP)

COMMON_DIR := ../common/obj
COMMON_OBJS := $(foreach dir,$(COMMON_DIR), $(wildcard $(dir)/*.o))

OBJS_DIR := obj
OBJS := $(patsubst %, $(OBJS_DIR)/%, $(PURE_OBJS))

vpath %.C $(SOURCES_DIRS)
vpath %.c $(SOURCES_DIRS)
vpath %.h $(HEADERS_DIRS) framework/parser/generated

$(OBJS) : | $(OBJS_DIR)

$(OBJS_DIR):
	mkdir $(OBJS_DIR)
"""
        return retStr

    def getObjectOnlyFlags(self):
        retStr = "#OBJECTONLYFLAGS is flags that only apply to objects, depend.sh generated code.\n"
        retStr += "OBJECTONLYFLAGS :="
        if self.objectMode == "64":
            retStr += " $(MAKE64)"
        if self.options.compiler == "xl":
            if self.options.blueGene == True:
                retStr += " " + XL_BG_RUNTIME_TYPE_INFO_FLAG
                if self.options.blueGeneP == True:
                    retStr += " " + BGP_FLAGS
                if self.options.blueGeneQ == True:
                    retStr += " " + BGQ_FLAGS
            else:
                retStr += " " + XL_RUNTIME_TYPE_INFO_FLAG
        retStr += "\n"
        return retStr

    def getCFlags(self):
        retStr = \
"""\
OTHER_LIBS :=-lgmp
CFLAGS := $(patsubst %,-I%/include,$(MODULES)) $(patsubst %,-I%/generated,$(PARSER_PATH)) $(patsubst %,-I%/include,$(SPECIAL_EXTENSION_MODULES))  -DLINUX -DDISABLE_DYNAMIC_LOADING -DHAVE_MPI
CFLAGS += -I../common/include -std=c++11 -Wno-deprecated-declarations \
"""
##CFLAGS := $(patsubst %,-I%/include,$(MODULES)) \
#$(patsubst %,-I%/generated,$(PARSER_PATH)) \
#$(patsubst %,-I%/include,$(SPECIAL_EXTENSION_MODULES)) \
#"""
        if self.options.blueGene == False :
            retStr += \
"""\
-MMD \
"""

        if self.options.compiler == "gcc":
            if self.options.withMpi == True:
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
            if self.options.compiler == "gcc":
                retStr += " " + GNU_DEBUGGING_FLAG
            elif self.options.compiler == "xl":
                retStr += " " + XL_DEBUGGING_FLAG
            else:
                retStr += " -g"

        if self.options.debug_assert == True:
            retStr += " " + DEBUG_ASSERT

        if self.options.debug_hh == True:
            retStr += " " + DEBUG_HH

        if self.options.debug_loops == True:
            retStr += " " + DEBUG_LOOPS

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

        if self.options.dynamicLoading == False:
            retStr += " -DDISABLE_DYNAMIC_LOADING"

        if self.options.pthreads == False:
            retStr += " -DDISABLE_PTHREADS"

        if self.options.withMpi == True:
            retStr += " -DHAVE_MPI"
            if self.options.compiler == "gcc":
               retStr += " -DLAM_BUILDING"

        if self.options.profile == USE:
            retStr += " -DPROFILING"

        if self.options.silent == True:
            retStr += " -DSILENT_MODE"

        if self.options.verbose == True:
            retStr += " -DVERBOSE"

        if self.options.compiler == "gcc":
            retStr += " -DWITH_GCC"

        if self.options.blueGene == True:
            retStr += " -DUSING_BLUEGENE"

        if self.options.blueGeneL == True:
            retStr += " -DUSING_BLUEGENEL"
            retStr += " -DMPICH_IGNORE_CXX_SEEK"
	    retStr += " -DMPICH_SKIP_MPICXX"

        if self.options.blueGeneP == True:
            retStr += " -DUSING_BLUEGENEP"

        if self.options.blueGeneQ == True:
            retStr += " -DUSING_BLUEGENEQ"

        if self.options.blueGene == True:
            retStr += " $(MPI_INC)"

        retStr += "\n"
        return retStr

    def getLensLibs(self):
        retStr = "LENS_LIBS := "
	if self.options.domainLibrary == True :
            #retStr += "lib/liblensdomain.a "
		retStr += "lib/liblens.a\n"
        return retStr

    def getLibs(self):
        retStr = "LENS_LIBS_EXT := $(LENS_LIBS) lib/liblensext.a\n"
        retStr += "LIBS := "
        if self.options.dynamicLoading == True:
            retStr += "-ldl "
        if self.options.pthreads == True:
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
            retStr += "-Wl,--whole-archive "
            retStr += "$(LENS_LIBS_EXT) "
            retStr += "-Wl,-no-whole-archive "
        else:
            raise InternalError("Unknown OS " + self.operatingSystem)

	if self.options.mpiTrace == True and self.options.blueGene != True:
	    printWarning("MPI Trace profiling not available.")
	elif self.options.mpiTrace == True:
	    retStr += " $(MPI_TRACE_LIBS)"

        if self.options.blueGene == True:
            retStr += " $(MPI_LIBS)"

        if self.options.compiler == "gcc":
           if self.options.withMpi == True:
              retStr += " -L$(LAMHOME)/lib -lmpi -llammpi++ -llam -ldl -lpthread"

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

        if self.operatingSystem == "Linux":
            retStr += LINUX_FINAL_TARGET_FLAG
        elif self.operatingSystem == "AIX":

            if self.options.dynamicLoading == True:
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

#        retStr += \
#"""\
#FILES_EDGESETSUBSCRIBERSOCKET = $(DCA_OBJ)/edgeSetOutboard.o $(DCA_OBJ)/EdgeSetSubscriberSocket.o $(DCA_OBJ)/socket.o
#FILES_NODESETSUBSCRIBERSOCKET = $(DCA_OBJ)/nodeSetOutboard.o $(DCA_OBJ)/NodeSetSubscriberSocket.o $(DCA_OBJ)/socket.o
#
#$(DCA_OBJ)/edgeSetOutboard.o: $(DCA_SRC)/outboard.c
#	$(C_COMP) $(DX_CFLAGS) -DUSERMODULE=m_EdgeWatchSocket -c $(DX_BASE)/lib/outboard.c -o $(DCA_OBJ)/edgeSetOutboard.o
#
#$(DCA_OBJ)/nodeSetOutboard.o: $(DCA_SRC)/outboard.c
#	$(C_COMP) $(DX_CFLAGS) -DUSERMODULE=m_NodeWatchSocket -c $(DX_BASE)/lib/outboard.c -o $(DCA_OBJ)/nodeSetOutboard.o
#
#$(DX_DIR)/EdgeSetSubscriberSocket: $(DCA_OBJ)/edgeSetOutboard.o $(DCA_SRC)/EdgeSetSubscriberSocket.c $(DCA_OBJ)/socket.o
#	$(C_COMP) $(DX_CFLAGS) -c $(DCA_SRC)/EdgeSetSubscriberSocket.c -o $(DCA_OBJ)/EdgeSetSubscriberSocket.o
#	$(C_COMP) $(FILES_EDGESETSUBSCRIBERSOCKET) $(DX_LITELIBS) -o $(DX_DIR)/EdgeSetSubscriberSocket;
#
#$(DX_DIR)/NodeSetSubscriberSocket: $(DCA_OBJ)/nodeSetOutboard.o $(DCA_SRC)/NodeSetSubscriberSocket.c $(DCA_OBJ)/socket.o
#	$(C_COMP) $(DX_CFLAGS) -c $(DCA_SRC)/NodeSetSubscriberSocket.c -o $(DCA_OBJ)/NodeSetSubscriberSocket.o
#	$(C_COMP) $(FILES_NODESETSUBSCRIBERSOCKET) $(DX_LITELIBS) -o $(DX_DIR)/NodeSetSubscriberSocket;
#
#"""
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

$(OBJS_DIR)/%.o : %.C
	$(CC) $(CFLAGS) -I$(NTI_INC_DIR) $(OBJECTONLYFLAGS) -c $< $(OTHER_LIBS) -o $@
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

"""
        return retStr


    def getModuleAndObjectIncludes(self):
        retStr = \
"""\
# Include the description of each module.
# This will be in the root/project, where root is the root subdir (E.g. dir1)
#include $(patsubst %,%/module.mk,$(MODULES))

# Include the C include dependencies
-include $(OBJS:.o=.d)

"""
        return retStr

    def getParserTargets(self):

        bisonOutputPrefix = ""
        if self.bisonVersion == "1.35":
            bisonOutputPrefix = "framework/parser/bison/"

        retStr = \
"""\

# Parser targets
speclang.tab.h:
	cd framework/parser/bison; $(BISON) -v -d speclang.y; \\
	mv speclang.tab.c ../generated/speclang.tab.C; mv speclang.tab.h ../generated

framework/parser/generated/speclang.tab.C: framework/parser/bison/speclang.y
	cd framework/parser/bison; $(BISON) -v -d speclang.y; \\
	mv speclang.tab.c ../generated/speclang.tab.C; mv speclang.tab.h ../generated

speclang.tab.o: framework/parser/generated/speclang.tab.C framework/parser/bison/speclang.y
	$(CC) -c $< -DYYDEBUG $(CFLAGS) $(OBJECTONLYFLAGS) -o $(OBJS_DIR)/$@

"""

        return retStr

    def getScannerTargets(self):

        if self.options.blueGene == False:
            retStr = \
                   """\
# Scanner targets
framework/parser/generated/lex.yy.C: framework/parser/flex/speclang.l
	$(FLEX) -+ framework/parser/flex/speclang.l
	sed 's/class istream;/#include <FlexFixer.h>/' \
           lex.yy.cc > lex.yy.cc.edited
	mv -f lex.yy.cc.edited framework/parser/generated/lex.yy.C
	rm lex.yy.cc
"""
        else:
            retStr = \
                   """\
# Scanner targets
framework/parser/generated/lex.yy.C: framework/parser/flex/speclang.l
	cp framework/parser/flex/lex.yy.C.linux.i386 framework/parser/generated/lex.yy.C
"""


        retStr += \
"""
lex.yy.o: framework/parser/generated/lex.yy.C framework/parser/flex/speclang.l
	$(CC) -c $< $(CFLAGS) $(OBJECTONLYFLAGS) -o $(OBJS_DIR)/$@
"""
        retStr += "\n"
        return retStr

    def getAllTarget(self):
        retStr = "cleanfirst:\n"
        retStr += "\t-rm $(BIN_DIR)/$(EXE_FILE)\n"
        #retStr = "final: $(BASE_OBJECTS) $(LENS_LIBS_EXT) $(BIN_DIR)/$(EXE_FILE) $(DCA_OBJ)/socket.o $(OBJS) $(MODULE_MKS)"
        retStr += "final: cleanfirst speclang.tab.h $(OBJS)  speclang.tab.o lex.yy.o socket.o  $(LENS_LIBS_EXT) "
        if self.options.dynamicLoading == True:
            retStr += " $(DEF_SYMBOLS) $(UNDEF_SYMBOLS) $(BIN_DIR)/createDF $(SHARED_OBJECTS) "
        if self.dx.exists == True:
            retStr += " $(DX_DIR)/EdgeSetSubscriberSocket $(DX_DIR)/NodeSetSubscriberSocket "
        retStr += "\n"
        retStr += "\t$(CC) $(FINAL_TARGET_FLAG) $(CFLAGS) $(OBJS_DIR)/speclang.tab.o $(OBJS_DIR)/lex.yy.o $(OBJS_DIR)/socket.o $(OBJS) $(LIBS) $(NTI_OBJS) $(COMMON_OBJS) $(OTHER_LIBS) -o $(BIN_DIR)/$(EXE_FILE) "
        return retStr

    def getDependfileTarget(self):
        retStr = "$(SO_DIR)/Dependfile: $(DEF_SYMBOLS) $(UNDEF_SYMBOLS) $(BIN_DIR)/createDF\n"
        if self.options.extMode == True:
            retStr += "\techo DependFile not generated, working in EXT_MODE"
        else:
            retStr += "\t$(BIN_DIR)/createDF $(SO_DIR) > $@"
        retStr += "\n"
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
        if self.options.withMpi == True:
            moduleList.append("streams")

        extraToken = ""
        if self.objectMode == "64":
            extraToken = " -X32_64"
        arCmd = "\tar" + extraToken + " rvu $@"
        for i in moduleList:
            retStr += arCmd + " $(OBJ_" + i + ")\n"
        retStr += "\tranlib $@\n\n"
        retStr += "lib/liblensext.a: $(EXTENSION_OBJECTS) $(LENS_LIBS)"
        if self.options.dynamicLoading == False:
            retStr += " $(GENERATED_DL_OBJECTS)"
        retStr += "\n"
        retStr += arCmd + " $(EXTENSION_OBJECTS)\n"
        if self.options.dynamicLoading == False:
            retStr += arCmd + " $(GENERATED_DL_OBJECTS)\n"
        retStr += "\tranlib $@\n\n"
        return retStr

    def getMainDefTarget(self):
        retStr = \
"""\
$(SO_DIR)/main.def: $(LENS_LIBS_EXT)
	echo \#\!. > $@
	$(SCRIPTS_DIR)/gen_def.sh $^ >> $@
"""
        return retStr

    def getLensparserTarget(self):
        retStr = "$(BIN_DIR)/$(EXE_FILE): cleanfirst "

        if self.operatingSystem == "AIX":
            retStr += LENSPARSER_TARGETS
            if self.options.dynamicLoading == True:
                retStr += "$(SO_DIR)/main.def "
            else:
                retStr += "$(LENS_LIBS_EXT) "
        elif self.operatingSystem == "Linux":
            retStr += LENSPARSER_TARGETS + "$(LENS_LIBS_EXT) "
        else:
            raise InternalError("Unknown OS " + self.operatingSystem)
        if self.options.dynamicLoading == True:
            retStr += " $(SHARED_OBJECTS)"
        retStr += "\n"
        retStr += "\t$(CC) $(FINAL_TARGET_FLAG) $(CFLAGS) $(PARSER_GENERATED)/speclang.tab.o $(PARSER_GENERATED)/lex.yy.o $(DCA_OBJ)/socket.o "
	retStr += "$(LIBS) "
	if self.options.tvMemDebug == True and self.operatingSystem != "AIX":
	    printWarning("Totalview memory debugging not available.")
	elif self.options.tvMemDebug == True:
	    if self.objectMode == "64":
                retStr += "-L$(TOTALVIEW_LIBPATH) -L$(TOTALVIEW_LIBPATH) $(TOTALVIEW_LIBPATH)/aix_malloctype64_5.o "
	    else:
		retStr += "-L$(TOTALVIEW_LIBPATH) -L$(TOTALVIEW_LIBPATH) $(TOTALVIEW_LIBPATH)/aix_malloctype.o "
	if self.dx.exists == True:
            retStr += "$(DX_LITELIBS)"
        if self.operatingSystem == "AIX":
            retStr += "$(XLINKER)"
	if self.options.profile == USE :
	    retStr += PROFILING_FLAGS
        retStr += " -o $(BIN_DIR)/$(EXE_FILE)\n"
        return retStr

    def getSocketTarget(self):
        retStr = "socket.o: "

        if self.dx.exists == True:
            retStr += "socket.c\n"
        else:
            retStr += "fakesocket.c\n"

        retStr += "\t$(C_COMP)"
        if self.objectMode == "64":
            retStr += " $(MAKE64)"
        if self.dx.exists == True:
            retStr += " $(DX_CFLAGS) -c $< "
        else:
            retStr += " -c $<"
        retStr += " -o $(OBJS_DIR)/$@\n"
        return retStr

    def getCleanTarget(self):
        retStr = \
"""\
.PHONY: clean
clean:
	-rm -f dx/EdgeSetSubscriberSocket
	-rm -f dx/NodeSetSubscriberSocket
	-rm -f $(BIN_DIR)/$(EXE_FILE)
	-rm -f $(BIN_DIR)/createDF
	-rm -f lib/liblens.a
	-rm -f lib/liblensext.a
	-rm -f $(SO_DIR)/Dependfile
	-rm -f framework/parser/bison/speclang.output
	-rm -f framework/parser/generated/lex.yy.C
	-rm -f framework/parser/generated/lex.yy.C
	-rm -f framework/parser/generated/speclang.tab.C
	-rm -f framework/parser/generated/speclang.tab.h
	-rm -f $(OBJS_DIR)/*
"""
#	-find . -name "*\.o" -exec /bin/rm {} \;
#	-find . -name "*\.so" -exec /bin/rm {} \;
#	-find . -name "*\.d" -exec /bin/rm {} \;
#	-find . -name "*\.ld" -exec /bin/rm {} \;
#	-find . -name "*\.yd" -exec /bin/rm {} \;
#	-find . -name "*\.ad" -exec /bin/rm {} \;
#	-find . -name "*\.def" -exec /bin/rm {} \;
#	-find . -name "*\.undef" -exec /bin/rm {} \;
        return retStr


    def generateMakefile(self, fileName):
        fileBody = self.getInitialValues()
        fileBody += "\n"
        fileBody += self.getMake64()
        fileBody += "\n"
        fileBody += self.getModuleDefinitions()

        fileBody += self.getObjectOnlyFlags()

        fileBody += self.getCFlags()
        fileBody += "\n"
	fileBody += self.getLensLibs()
        fileBody += self.getLibs()
        fileBody += "\n"

        if self.options.dynamicLoading == True:
            fileBody += self.getSharedPFix()
            fileBody += self.getSharedCC()

        fileBody += self.getFinalTargetFlag()
        fileBody += self.getXLinker()
        fileBody += "\n"
        fileBody += self.getSuffixRules()
        # getCreateDFTargets() has to be before getModuleAndObjectIncludes()
        if self.options.dynamicLoading == True:
            fileBody += self.getCreateDFTargets()
        fileBody += self.getModuleAndObjectIncludes()

        if self.dx.exists == True:
            fileBody += self.getDXSpecificCode()

        fileBody += self.getParserTargets()
        fileBody += self.getScannerTargets()
        fileBody += self.getAllTarget()
        fileBody += "\n"
        if self.options.dynamicLoading == True:
            fileBody += self.getDependfileTarget()
            fileBody += "\n"
        fileBody += self.getLibLensTarget()
        fileBody += "\n"
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


if __name__ == "__main__":
    try:
        buildSetup = BuildSetup()
        buildSetup.main()
    except FatalError, error:
        error.printError()
        sys.exit(-1)



