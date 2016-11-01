import sys
import argparse
__author__ = "Hoang Trong Minh Tuan"
__copyright__ = "Copyright 2015, IBM"
__credits__ = []  # list of people who file pugs
__version__ = "0.1"
__maintainer__ = "Hoang Trong Minh Tuan"
__email__ = "tmhoangt@us.ibm.com"
__status__ = "Prototype"  # (Prototype, Development, Production)


class Parser(object):

    _parser = None

    @classmethod
    def parse_command(cls, argv=sys.argv[1:]):
        _epilog = """
        The program reads the .swc file of a neuron, analyze all branches; and
        then generate the location at which synapses reside based on the
        statistics from MSN neurons \n
    Currently, synapses are all the same.
    This can be easily extended to other neuron types (e.g. pyramidal neurons)
    with different statistics. \n
    PLAN: support different synapse types (e.g. thin, stubby, mushroom)"""

        _description = """
        Generate the synapses for MSN neuron,
        in the form of multiple 'bouton' neurons
        """

        cls._parser = argparse.ArgumentParser(description=_description,
                                              epilog=_epilog)

        # -h, -help is implicitly added
        cls._parser.add_argument('--version', action='version',
                                 version=__version__)
        cls._parser.add_argument("-v", "--verbose",
                                 help="increase output verbosity",
                                 action="store_true")
        cls._parser.add_argument("swc", help="name of the .swc file", type=str)
        args = cls._parser.parse_args()
        return args

    @classmethod
    def print_help(cls):
        cls._parser.print_help()
