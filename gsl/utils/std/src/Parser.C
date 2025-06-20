// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Parser.h"
#include "Simulation.h"

std::vector<std::string> getNamesOfSupportedNodeType() {
   return FactoryMap<NodeType>::getFactoryMap()->getNamesOfSupportedType();
}
std::vector<std::string> getNamesOfSupportedFunctorType() {
   return FactoryMap<FunctorType>::getFactoryMap()->getNamesOfSupportedType();
}
std::vector<std::string> getNamesOfSupportedTriggerType() {
   return FactoryMap<TriggerType>::getFactoryMap()->getNamesOfSupportedType();
}
std::vector<std::string> getNamesOfSupportedVariableType() {
   return FactoryMap<VariableType>::getFactoryMap()->getNamesOfSupportedType();
}
std::vector<std::string> getNamesOfSupportedConstantType() {
   return FactoryMap<ConstantType>::getFactoryMap()->getNamesOfSupportedType();
}
void Parser::help() {
   std::cout << "Usage: gslparser gslFile <options> <switches>\n" << std::endl;
   std::cout << "  For options, a value is needed:" << std::endl;
   std::cout << "    -h :      Help" << std::endl;
#ifndef DISABLE_PTHREADS
   std::cout << "    -t :      Number of threads (default:1)" << std::endl;
   std::cout << "    -u :      User interface: text, gui or none. (default:text)" << std::endl;
   std::cout << "    -p :      Port for gui connection to simulation. (default:4000)" << std::endl;
   std::cout << "    -w :      Number of work units. (default:nThreads)" << std::endl;
   std::cout << "    -d :      The GPU device ID to use (same for all MPI ranks)" << std::endl;
#endif
   std::cout << std::endl;
   std::cout << "  switches:" << std::endl;
#ifndef DISABLE_PTHREADS
   std::cout << "    -b :      Bind threads to CPUs in AIX" << std::endl;
#endif
   std::cout << "    -s :      Set local/global random number generator seed." << std::endl;
   std::cout << "    -e :      Enable the use of edge relational data" << std::endl;
   std::cout << "    -r :      Read graph partitioning from file" << std::endl;
   std::cout << "    -o :      Output graph partitioning to file" << std::endl;
   std::cout << "    -m :      Suppress full simulatation (partition only)" << std::endl;
   std::cout << std::endl;
   std::cout << "  gslFile: the gsl specification file to parse (mandatory)" << std::endl;
   std::cout << std::endl << std::endl;

   std::cout << "Supported NodeTypes: ";
   getNamesOfSupportedNodeType();
   std::cout << "\n";
   std::cout << "Supported FunctorTypes: ";
   getNamesOfSupportedFunctorType();
   std::cout << "\n";
   std::cout << "Supported TriggerTypes: ";
   getNamesOfSupportedTriggerType();
   std::cout << "\n";
   std::cout << "Supported VariableTypes: ";
   getNamesOfSupportedVariableType();
   std::cout << "\n";
   std::cout << "Supported ConstantTypes: ";
   getNamesOfSupportedConstantType();
}

int Parser::countOptions(String arg) {
	int count = 0;
	for (OptionVector::size_type i = 0; i < getOptions().size(); i++) {
	  if (getOption(i).getLongName() != Parser::Option::LONG_NAME_NONE) {
			String longName = "--" + getOption(i).getLongName();
			if (longName.compare(0, arg.length(), arg) == 0) count++;
		}
	}
	return(count);
}
int Parser::findOption(String arg) {
	for (OptionVector::size_type i = 0; i < getOptions().size(); i++) {
		if (getOption(i).getLongName() != Parser::Option::LONG_NAME_NONE) { 
                        String longName = "--" + getOption(i).getLongName();
			if (longName.compare(0, arg.length(), arg) == 0) return(i);
		}
	}
	return(-1);
}
int Parser::countOptions(char c) {
	int count = 0;
	for (OptionVector::size_type i = 0; i < getOptions().size(); i++) {
		if (getOption(i).getShortName() != Parser::Option::SHORT_NAME_NONE) {
			if (getOption(i).getShortName() == c) count++;
		}
	}
	return(count);
}
int Parser::findOption(char c) {
	for (OptionVector::size_type i = 0; i < getOptions().size(); i++) {
		if (getOption(i).getShortName() != Parser::Option::SHORT_NAME_NONE) {
			if (getOption(i).getShortName() == c) return(i);
		}
	}
	return(-1);
}
int Parser::countType(String arg, Parser::Option::Type type) {
	int count = 0;
	for (String::size_type i = 1; i < arg.length(); i++) {
		int j = findOption(arg[i]);
		if (j >= 0 && getOption(j).getType() == type) {
			count++;
		}
	}
	return(count);
}
int Parser::countArgs(StringVector &args, StringVector::size_type start) {
	int count = 0;
	for (StringVector::size_type i = start + 1; i < args.size(); i++) {
		if (args[i].compare(0, 1, "-") == 0) return(count);
		count++;
	}
	return(count);
}
Parser::ParameterVector Parser::parse(int argc, char *argv[]) {
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
				throw Exception("Parameter \"" + args[i] + "\" did not match any options.");
			} else if (optionCount > 1) {
				throw Exception("Parameter \"" + args[i] + "\" matched more than one option.");
			}
			Option option = getOption(findOption(args[i]));
			if (option.getType() == Parser::Option::TYPE_REQUIRED) {
				if (i + 1 < args.size()) {
					parameterVector.push_back(Parameter(option, args[i+1]));
					i = i + 1;
				} else {
					throw Exception("Option \"" + args[i] + "\" requires a value.");
				}
			} else if (option.getType() == Parser::Option::TYPE_OPTIONAL) {
				if (i + 1 < args.size() && args[i+1].compare(0, 1, "-") != 0) {
					parameterVector.push_back(Parameter(option, args[i+1]));
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
			int optionals = countType(args[i], Parser::Option::TYPE_OPTIONAL);
			int requireds = countType(args[i], Parser::Option::TYPE_REQUIRED);
			int argCount = countArgs(args, i);
			bool optionalsTakeValues = false;
			if (optionals + requireds <= argCount) {
				optionalsTakeValues = true;
			} else if (requireds <= argCount) {
				optionalsTakeValues = false;
			} else {
				throw Exception("Option \"" + args[i] + "\" has wrong number of values.");
			}
			//
			for (String::size_type j = 1; j < args[i].length(); j++) {
				int optionCount = countOptions(args[i][j]);
				if (optionCount < 1) {
					throw Exception("Parameter \"-" + args[i].substr(j, 1) + "\" did not match any options.");
				} else if (optionCount > 1) {
					throw Exception("Parameter \"-" + args[i].substr(j, 1) + "\" matched more than one option.");
				}
				Option option = getOption(findOption(args[i][j]));
				if (option.getType() == Parser::Option::TYPE_REQUIRED) {
					if (k + 1 < args.size()) {
						parameterVector.push_back(Parameter(option, args[k + 1]));
						k = k + 1;
					} else {
						throw Exception("Option \"-" + args[i].substr(j, 1) + "\" requires a value.");
					}
				} else if (option.getType() == Parser::Option::TYPE_OPTIONAL) {
					if (optionalsTakeValues) {
						if (k + 1 < args.size() && args[k + 1].compare(0, 1, "-") != 0) {
							parameterVector.push_back(Parameter(option, args[k + 1]));
							k = k + 1;
						} else {
							// Really shouldn't happen...we know there are enough
							// args because we counted them before looping...
							throw Exception("Option \"-" + args[i].substr(j, 1) + "\" requires a value.");
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
			parameterVector.push_back(Parameter(Option::OPTION_NONE, args[i]));
		}
	}
	// Output debugging info...
	/*for (ParameterVector::size_type i = 0; i < parameterVector.size(); i++) {
		Option option = parameterVector[i].getOption();
		String value = parameterVector[i].getValue();
		std::cout << "Parameter Vector: " <<
			option.getShortName() << " / " << option.getLongName() << " / " << value << "\n";
	}*/
	return(parameterVector);
}


char Parser::Option::SHORT_NAME_NONE = 0;
std::string Parser::Option::LONG_NAME_NONE = "";
Parser::Option Parser::Option::OPTION_NONE = Parser::Option(Parser::Option::SHORT_NAME_NONE, Parser::Option::LONG_NAME_NONE, Parser::Option::TYPE_NONE);

Parser::Option::Option(char shortName, std::string longName, Type type) {
	fieldShortName = shortName;
	fieldLongName = longName;
	fieldType = type;
}

Parser::Option::Option(const Option &option) {
	fieldShortName = option.getShortName();
	fieldLongName = option.getLongName();
	fieldType = option.getType();
}

Parser::Option::~Option() {
}

Parser::Option &Parser::Option::operator=(const Parser::Option &option) {
	setShortName(option.getShortName());
	setLongName(option.getLongName());
	setType(option.getType());
	return(*this);
}


