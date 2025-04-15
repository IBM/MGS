// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef Generatable_H
#define Generatable_H
#include "Mdl.h"

#include <string>
#include <memory>
#include <vector>
#include <set>
#include <map>
#include <cassert>

#include "MemberContainer.h"
#include "Constants.h"

class Class;
class DataType;
class BaseClass; 
class CommandLine;

class Generatable {
   public:
      enum LinkType {_DYNAMIC, _STATIC};

      // Standard functions
      Generatable(const std::string& fileName);
      Generatable(const Generatable& rv);
      Generatable& operator=(const Generatable& rv);
      virtual void duplicate(std::unique_ptr<Generatable>&& rv) const = 0;
      virtual ~Generatable();        

      //void addSupportForMachineType(MachineType mach){ _supportedMachineTypes.insert(mach) };
      void addSupportForMachineType(MachineType mach){ _supportedMachineType = (mach);};
      bool isSupportedMachineType(MachineType mach) const { 
	      //const bool is_in = _supportedMachineTypes.find(mach) != _supportedMachineTypes.end();
	      const bool is_in = _supportedMachineType == mach;
	      return is_in;
      };
      MachineType getSupportedMachineType() const { 
	      return _supportedMachineType;
      };

      // Used to print out code for feedback to the user.
      virtual void generate() const = 0;
      // Used for code generation.
      void generateFiles(const std::string& originalFileName);

      // Returns the description of the generated type(NodeType, 
      // FunctorType, etc.). 
      virtual std::string getTypeDescription();

      // Fills in the map with itself; used for generating makefile extension.
      void addSelfToExtensionModules(std::map<std::string, 
				     std::vector<std::string> >& modules);
      // Fills in the map with itself; used for copyModules.
      void addSelfToCopyModules(std::map<std::string, 
				std::vector<std::string> >& modules);

      LinkType getLinkType() const {
	      return _linkType;
      }

      void setLinkType(LinkType obj) {
	      _linkType = obj;
      }

      bool isFrameWorkElement() const {
	      return _frameWorkElement;
      }
      
      void setFrameWorkElement(bool val = true) {
	      _frameWorkElement = val;
      }
      void setCommandLine(CommandLine& obj) {
         _cmdLine = &obj;
      }
      CommandLine* getCommandLine() { 
	      assert(_cmdLine != 0); 
	      return _cmdLine;
      }

   protected:
      void copyOwnedHeap(const Generatable& rv);
      void destructOwnedHeap();

      // This function is called by generateFiles. The programmer overrides
      // the function  and adds the classes, that will be code generated, 
      // to the generatable.
      virtual void internalGenerateFiles() = 0;

      // Generates the module.mk
      void generateModuleMk();

      // Generates the XType class.
      void generateType();

      // Generated the XFactory class.
      void generateFactory();

      // This function is used by generateFactory. It returns the name of 
      // the class that will be loaded by the factory.
      virtual std::string getLoadedInstanceTypeName();

      // This function is used by generateFactory. It returns the arguments 
      // that will be used while instantiating the loaded class.
      virtual std::string getLoadedInstanceTypeArguments();
      
      // Creates the necessary directories.
      void createDirectoryStructure();     
      
      // Returns the modules name.
      virtual std::string getModuleName() const = 0;
      
      // Returns the modules type name, e.g., NodeType.
      virtual std::string getModuleTypeName() const = 0;

      // used by generateType, dictates which insrance type will be created.
      virtual std::string getInstanceNameForType() const {
	      return getModuleName();
      }
      
      // This function is called by the getType function, it returns the
      // arguments that are necessary in creating a new instance.
      virtual std::string getInstanceNameForTypeArguments() const {
	      return "";
      }

      // This function adds a doInitialize function to the class given using
      // the members.
      void addDoInitializeMethods(
	      Class& instance, const MemberContainer<DataType>& members) const;

      std::string getDoInitializeMethodBody(
	      const MemberContainer<DataType>& members) const;

      std::string getSetupFromNDPairListMethodBody(
	      const MemberContainer<DataType>& members) const;

      // This function can be overridden to add an attribute to the class of generate type.
      virtual void addGenerateTypeClassAttribute(Class& c) const {
	      return;
      }

      // The classes that this generatable has.
      std::vector<Class*> _classes;
      // Write to disk if true.
      bool _fileOutput;

   private:
      // The filename the generatable was read from. More specifically
      // the components(generataables) in .mdl files which are 
      // #included do not generate the actual files.
      std::string _fileName; 
      LinkType _linkType;
      bool _frameWorkElement;
      //std::set<MachineType> _supportedMachineTypes;
      MachineType _supportedMachineType; //for now, it enable HAVE_GPU flag if GPU is supported
      CommandLine* _cmdLine; //hold the result parsed from command-line [use to determine code-behavior]
};

#endif // Generatable_H
