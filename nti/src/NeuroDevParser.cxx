// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "NeuroDevParser.h"
#include <mpi.h>

void NeuroDevParser::help() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::cout << "Usage: neuroDev/touchDetect input-file <options> <switches>\n"
              << std::endl;
    std::cout << "  For options, a value is needed:" << std::endl;

    std::cout << "    -a : apposition-sampling-rate : Prior to synapse "
                 "creation sampling, touch sampling rate." << std::endl;
    std::cout
        << "    -b : binary-input-file : Binary structural input file name."
        << std::endl;
    std::cout
        << "    -c : client-connect : Connect to visualization environment."
        << std::endl;
    std::cout << "    -d : n-detectors : Number of processors used for touch "
                 "and force detection. Cardinal value 0 yields MPI_Comm_size. "
                 "Must be power of 3 or product of slicing geometry."
              << std::endl;
    std::cout << "    -e : energy-conv-crit : Energy convergence criterion. MD "
                 "concept used in force field iteration." << std::endl;
    std::cout << "    -f : initial-front : Initial front from which growth "
                 "begins. Must be >= 0." << std::endl;
    std::cout << "    -g : geometric-resampling-factor : Tangent sphere "
                 "spacing, multiple of [r(n)+r(n+1)]." << std::endl;
    std::cout << "    -h : help" << std::endl;
    std::cout
        << "    -i : text-input-file : Input tissue specification file name."
        << std::endl;
    std::cout << "    -j : threads : Number of threads used for touch and "
                 "force detection inner loop." << std::endl;
    std::cout << "    -k : cell-body-migration : Initial position of cell "
                 "bodies determined by tissue interactions." << std::endl;
    std::cout << "    -m : max-iterations : Maximum number of iterations. Must "
                 "be >= 1." << std::endl;
    std::cout << "    -n : decomposition : Computational decomposition : "
                 "volume (default), neuron (experimental)." << std::endl;
    std::cout << "    -o : output-file : Output file name (text or binary)."
              << std::endl;
    std::cout << "    -p : param-file : Parameter file name." << std::endl;
    std::cout << "    -r : compartment-resampling-factor : Number of capsules "
                 "per physiological compartment, resampled following touch "
                 "detection." << std::endl;
    std::cout << "    -s : n-slicers : Number of processors used for column "
                 "slicing. Cardinal value 0 yields MPI_Comm_size. Must be <= "
                 "number of neurons." << std::endl;
    std::cout << "    -t : time-step : Time step. MD concept usesd in force "
                 "field iteration." << std::endl;
    std::cout << "    -u : dump-all-output : output file format : text 't' "
                 "(default) | binary 'b' | text and binary 'bt'." << std::endl;
    std::cout << "    -x : slicing-geom-x : X dimension of slicing geometry."
              << std::endl;
    std::cout << "    -y : slicing-geom-y : Y dimension of slicing geometry."
              << std::endl;
    std::cout << "    -z : slicing-geom-z : Z dimension of slicing geometry."
              << std::endl;
    std::cout << std::endl << std::endl;
  }
}

int NeuroDevParser::countOptions(String arg) {
  int count = 0;
  for (OptionVector::size_type i = 0; i < getOptions().size(); i++) {
    if (getOption(i).getLongName() !=
        NeuroDevParser::Option::ND_LONG_NAME_NONE) {
      String longName = "--" + getOption(i).getLongName();
      if (longName.compare(0, arg.length(), arg) == 0) count++;
    }
  }
  return (count);
}
int NeuroDevParser::findOption(String arg) {
  for (OptionVector::size_type i = 0; i < getOptions().size(); i++) {
    if (getOption(i).getLongName() !=
        NeuroDevParser::Option::ND_LONG_NAME_NONE) {
      String longName = "--" + getOption(i).getLongName();
      if (longName.compare(0, arg.length(), arg) == 0) return (i);
    }
  }
  return (-1);
}
int NeuroDevParser::countOptions(char c) {
  int count = 0;
  for (OptionVector::size_type i = 0; i < getOptions().size(); i++) {
    if (getOption(i).getShortName() !=
        NeuroDevParser::Option::ND_SHORT_NAME_NONE) {
      if (getOption(i).getShortName() == c) count++;
    }
  }
  return (count);
}
int NeuroDevParser::findOption(char c) {
  for (OptionVector::size_type i = 0; i < getOptions().size(); i++) {
    if (getOption(i).getShortName() !=
        NeuroDevParser::Option::ND_SHORT_NAME_NONE) {
      if (getOption(i).getShortName() == c) return (i);
    }
  }
  return (-1);
}
int NeuroDevParser::countType(String arg, NeuroDevParser::Option::Type type) {
  int count = 0;
  for (String::size_type i = 1; i < arg.length(); i++) {
    int j = findOption(arg[i]);
    if (j >= 0 && getOption(j).getType() == type) {
      count++;
    }
  }
  return (count);
}
int NeuroDevParser::countArgs(StringVector &args,
                              StringVector::size_type start) {
  int count = 0;
  for (StringVector::size_type i = start + 1; i < args.size(); i++) {
    if (args[i].compare(0, 1, "-") == 0) return (count);
    count++;
  }
  return (count);
}

NeuroDevParser::ParameterVector NeuroDevParser::parse(
    std::vector<std::string> &tokens) {
  ParameterVector parameterVector;
  StringVector args;
  // First we copy the argv array into the args vector,
  // breaking up argvs of the form --option=value as we go.
  int argc = tokens.size();
  for (std::vector<std::string>::iterator it = tokens.begin();
       it != tokens.end(); ++it) {
    String arg(*it);
    if (arg.length() > 2 && arg.compare(0, 2, "--") == 0) {
      String::size_type position = arg.find('=');
      if (position != std::string::npos) {
        String value = arg.substr(position + 1, arg.length() - position - 1);
        arg = arg.substr(0, position);
        args.push_back(arg);
        args.push_back(value);
      } else {
        args.push_back(arg);
      }
    } else {
      args.push_back(arg);
    }
  }
  // Now go through the args vector and convert them to Parameters...
  // This is incredibly ugly...and I don't like ugly...but for now...
  // we dance with the one that brung ya...
  for (StringVector::size_type i = 0, k = 0; i < args.size(); i++, k = i) {
    if (args[i].length() > 2 && args[i].compare(0, 2, "--") == 0) {
      int optionCount = countOptions(args[i]);
      if (countOptions(args[i]) < 1) {
        throw Exception("Parameter \"" + args[i] +
                        "\" did not match any options.");
      } else if (optionCount > 1) {
        throw Exception("Parameter \"" + args[i] +
                        "\" matched more than one option.");
      }
      Option option = getOption(findOption(args[i]));
      if (option.getType() == NeuroDevParser::Option::TYPE_REQUIRED) {
        if (i + 1 < args.size()) {
          parameterVector.push_back(Parameter(option, args[i + 1]));
          i = i + 1;
        } else {
          throw Exception("Option \"" + args[i] + "\" requires a value.");
        }
      } else if (option.getType() == NeuroDevParser::Option::TYPE_OPTIONAL) {
        if (i + 1 < args.size() && args[i + 1].compare(0, 1, "-") != 0) {
          parameterVector.push_back(Parameter(option, args[i + 1]));
          i = i + 1;
        } else {
          parameterVector.push_back(Parameter(option));
        }
      } else {
        parameterVector.push_back(Parameter(option));
      }
    } else if (args[i].length() == 2 && args[i].compare(0, 2, "--") == 0) {
      // What is this? An error? Assume yes...
      throw Exception("Option \"" + args[i] + "\" unknown.");
    } else if (args[i].length() > 1 && args[i].compare(0, 1, "-") == 0) {
      // First check optionals...
      int optionals = countType(args[i], NeuroDevParser::Option::TYPE_OPTIONAL);
      int requireds = countType(args[i], NeuroDevParser::Option::TYPE_REQUIRED);
      int argCount = countArgs(args, i);
      bool optionalsTakeValues = false;
      if (optionals + requireds <= argCount) {
        optionalsTakeValues = true;
      } else if (requireds <= argCount) {
        optionalsTakeValues = false;
      } else {
        throw Exception("Option \"" + args[i] +
                        "\" has wrong number of values.");
      }
      //
      for (String::size_type j = 1; j < args[i].length(); j++) {
        int optionCount = countOptions(args[i][j]);
        if (optionCount < 1) {
          throw Exception("Parameter \"-" + args[i].substr(j, 1) +
                          "\" did not match any options.");
        } else if (optionCount > 1) {
          throw Exception("Parameter \"-" + args[i].substr(j, 1) +
                          "\" matched more than one option.");
        }
        Option option = getOption(findOption(args[i][j]));
        if (option.getType() == NeuroDevParser::Option::TYPE_REQUIRED) {
          if (k + 1 < args.size()) {
            parameterVector.push_back(Parameter(option, args[k + 1]));
            k = k + 1;
          } else {
            throw Exception("Option \"-" + args[i].substr(j, 1) +
                            "\" requires a value.");
          }
        } else if (option.getType() == NeuroDevParser::Option::TYPE_OPTIONAL) {
          if (optionalsTakeValues) {
            if (k + 1 < args.size() && args[k + 1].compare(0, 1, "-") != 0) {
              parameterVector.push_back(Parameter(option, args[k + 1]));
              k = k + 1;
            } else {
              // Really shouldn't happen...we know there are enough
              // args because we counted them before looping...
              throw Exception("Option \"-" + args[i].substr(j, 1) +
                              "\" requires a value.");
            }
          } else {
            parameterVector.push_back(Parameter(option));
          }
        } else {
          parameterVector.push_back(Parameter(option));
        }
      }
      i = k;
    } else if (args[i].length() == 1 && args[i].compare(0, 1, "-") == 0) {
      // What is this? An error? Assume yes...
      throw Exception("Option \"" + args[i] + "\" unknown.");
    } else {
      // A value without a corresponding option...add it like this...
      parameterVector.push_back(Parameter(Option::ND_OPTION_NONE, args[i]));
    }
  }
  // Output debugging info...
  /*for (ParameterVector::size_type i = 0; i < parameterVector.size(); i++) {
          Option option = parameterVector[i].getOption();
          String value = parameterVector[i].getValue();
          std::cout << "Parameter Vector: " <<
                  option.getShortName() << " / " << option.getLongName() << " /
  " << value << "\n";
  }*/
  return (parameterVector);
}

NeuroDevParser::ParameterVector NeuroDevParser::parse(int argc, char *argv[]) {
  ParameterVector parameterVector;
  StringVector args;
  // First we copy the argv array into the args vector,
  // breaking up argvs of the form --option=value as we go.
  for (int i = 1; i < argc; i++) {
    String arg(argv[i]);
    if (arg.length() > 2 && arg.compare(0, 2, "--") == 0) {
      String::size_type position = arg.find('=');
      if (position != std::string::npos) {
        String value = arg.substr(position + 1, arg.length() - position - 1);
        arg = arg.substr(0, position);
        args.push_back(arg);
        args.push_back(value);
      } else {
        args.push_back(arg);
      }
    } else {
      args.push_back(arg);
    }
  }
  // Now go through the args vector and convert them to Parameters...
  // This is incredibly ugly...and I don't like ugly...but for now...
  // we dance with the one that brung ya...
  for (StringVector::size_type i = 0, k = 0; i < args.size(); i++, k = i) {
    if (args[i].length() > 2 && args[i].compare(0, 2, "--") == 0) {
      int optionCount = countOptions(args[i]);
      if (countOptions(args[i]) < 1) {
        throw Exception("Parameter \"" + args[i] +
                        "\" did not match any options.");
      } else if (optionCount > 1) {
        throw Exception("Parameter \"" + args[i] +
                        "\" matched more than one option.");
      }
      Option option = getOption(findOption(args[i]));
      if (option.getType() == NeuroDevParser::Option::TYPE_REQUIRED) {
        if (i + 1 < args.size()) {
          parameterVector.push_back(Parameter(option, args[i + 1]));
          i = i + 1;
        } else {
          throw Exception("Option \"" + args[i] + "\" requires a value.");
        }
      } else if (option.getType() == NeuroDevParser::Option::TYPE_OPTIONAL) {
        if (i + 1 < args.size() && args[i + 1].compare(0, 1, "-") != 0) {
          parameterVector.push_back(Parameter(option, args[i + 1]));
          i = i + 1;
        } else {
          parameterVector.push_back(Parameter(option));
        }
      } else {
        parameterVector.push_back(Parameter(option));
      }
    } else if (args[i].length() == 2 && args[i].compare(0, 2, "--") == 0) {
      // What is this? An error? Assume yes...
      throw Exception("Option \"" + args[i] + "\" unknown.");
    } else if (args[i].length() > 1 && args[i].compare(0, 1, "-") == 0) {
      // First check optionals...
      int optionals = countType(args[i], NeuroDevParser::Option::TYPE_OPTIONAL);
      int requireds = countType(args[i], NeuroDevParser::Option::TYPE_REQUIRED);
      int argCount = countArgs(args, i);
      bool optionalsTakeValues = false;
      if (optionals + requireds <= argCount) {
        optionalsTakeValues = true;
      } else if (requireds <= argCount) {
        optionalsTakeValues = false;
      } else {
        throw Exception("Option \"" + args[i] +
                        "\" has wrong number of values.");
      }
      //
      for (String::size_type j = 1; j < args[i].length(); j++) {
        int optionCount = countOptions(args[i][j]);
        if (optionCount < 1) {
          throw Exception("Parameter \"-" + args[i].substr(j, 1) +
                          "\" did not match any options.");
        } else if (optionCount > 1) {
          throw Exception("Parameter \"-" + args[i].substr(j, 1) +
                          "\" matched more than one option.");
        }
        Option option = getOption(findOption(args[i][j]));
        if (option.getType() == NeuroDevParser::Option::TYPE_REQUIRED) {
          if (k + 1 < args.size()) {
            parameterVector.push_back(Parameter(option, args[k + 1]));
            k = k + 1;
          } else {
            throw Exception("Option \"-" + args[i].substr(j, 1) +
                            "\" requires a value.");
          }
        } else if (option.getType() == NeuroDevParser::Option::TYPE_OPTIONAL) {
          if (optionalsTakeValues) {
            if (k + 1 < args.size() && args[k + 1].compare(0, 1, "-") != 0) {
              parameterVector.push_back(Parameter(option, args[k + 1]));
              k = k + 1;
            } else {
              // Really shouldn't happen...we know there are enough
              // args because we counted them before looping...
              throw Exception("Option \"-" + args[i].substr(j, 1) +
                              "\" requires a value.");
            }
          } else {
            parameterVector.push_back(Parameter(option));
          }
        } else {
          parameterVector.push_back(Parameter(option));
        }
      }
      i = k;
    } else if (args[i].length() == 1 && args[i].compare(0, 1, "-") == 0) {
      // What is this? An error? Assume yes...
      throw Exception("Option \"" + args[i] + "\" unknown.");
    } else {
      // A value without a corresponding option...add it like this...
      parameterVector.push_back(Parameter(Option::ND_OPTION_NONE, args[i]));
    }
  }
  // Output debugging info...
  /*for (ParameterVector::size_type i = 0; i < parameterVector.size(); i++) {
          Option option = parameterVector[i].getOption();
          String value = parameterVector[i].getValue();
          std::cout << "Parameter Vector: " <<
                  option.getShortName() << " / " << option.getLongName() << " /
  " << value << "\n";
  }*/
  return (parameterVector);
}

char NeuroDevParser::Option::ND_SHORT_NAME_NONE = 0;
std::string NeuroDevParser::Option::ND_LONG_NAME_NONE = "";
NeuroDevParser::Option NeuroDevParser::Option::ND_OPTION_NONE =
    NeuroDevParser::Option(NeuroDevParser::Option::ND_SHORT_NAME_NONE,
                           NeuroDevParser::Option::ND_LONG_NAME_NONE,
                           NeuroDevParser::Option::TYPE_NONE);

NeuroDevParser::Option::Option(char shortName, std::string longName,
                               Type type) {
  fieldShortName = shortName;
  fieldLongName = longName;
  fieldType = type;
}

NeuroDevParser::Option::Option(const Option &option) {
  fieldShortName = option.getShortName();
  fieldLongName = option.getLongName();
  fieldType = option.getType();
}

NeuroDevParser::Option::~Option() {}

NeuroDevParser::Option &NeuroDevParser::Option::operator=(
    const NeuroDevParser::Option &option) {
  setShortName(option.getShortName());
  setLongName(option.getLongName());
  setType(option.getType());
  return (*this);
}
