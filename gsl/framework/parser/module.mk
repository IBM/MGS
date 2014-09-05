# The pathname is relative to the lens directory
# Define some common prefixes/suffixes for use
THIS_DIR := framework/parser
THIS_STEM := parser

SRC_PREFIX := $(THIS_DIR)/src
OBJ_PREFIX := $(THIS_DIR)/obj

SOURCES := \
ArgumentListHelper.C \
C_argument.C \
C_argument_decl_args.C \
C_argument_void.C \
C_argument_constant.C \
C_argument_edgeset.C \
C_argument_gridset.C \
C_argument_nodeset.C \
C_argument_declarator.C \
C_argument_query_path_product.C \
C_argument_string.C \
C_argument_relative_nodeset.C \
C_argument_pset.C \
C_argument_ndpair_clause.C \
C_argument_ndpair_clause_list.C \
C_argument_argument_list.C \
C_argument_list.C \
C_argument_matrix.C \
C_complex_functor_declaration_body.C \
C_complex_functor_definition.C \
C_composite_definition_body.C \
C_composite_statement.C \
C_composite_statement_list.C \
C_connection_script_declaration.C \
C_connection_script_definition.C \
C_connection_script_definition_body.C \
C_constant.C \
C_constant_list.C \
C_complex_functor_clause_list.C \
C_complex_functor_clause.C \
C_constructor_clause.C \
C_declaration.C \
C_declaration_edgeset.C \
C_declaration_decl_decl_args.C \
C_declaration_float.C \
C_declaration_gridcoord.C \
C_declaration_index_set.C \
C_declaration_int.C \
C_declaration_init_phases.C \
C_declaration_final_phases.C \
C_declaration_list_parameter.C \
C_declaration_load_phases.C \
C_declaration_matrix_type.C \
C_declaration_node_type_set.C \
C_declaration_nodeset.C \
C_declaration_ndpair.C \
C_declaration_ndpairlist.C \
C_declaration_port.C \
C_declaration_pset.C \
C_declaration_publisher.C \
C_declaration_rel_nodeset.C \
C_declaration_repertoire.C \
C_declaration_repname.C \
C_declaration_runtime_phases.C \
C_declaration_service.C \
C_declaration_stride.C \
C_declaration_trigger.C \
C_declaration_typedef.C \
C_declaration_separation_constraint.C \
C_declarator.C \
C_declarator_nodeset_extension.C \
C_default_clause.C \
C_definition.C \
C_definition_edgetype.C \
C_definition_functor.C \
C_definition_constanttype.C \
C_definition_grid_granule.C \
C_definition_variabletype.C \
C_definition_nodetype.C \
C_definition_trigger.C \
C_definition_struct.C \
C_definition_type.C \
C_dim_declaration.C \
C_directive.C \
C_edgeset.C \
C_edgeset_extension.C \
C_equality_expression.C \
C_function_clause.C \
C_functor_category.C \
C_functor_declarator.C \
C_functor_definition.C \
C_functor_specifier.C \
C_grid_definition_body.C \
C_grid_function_name.C \
C_grid_function_specifier.C \
C_grid_translation_declaration.C \
C_grid_translation_declaration_list.C \
C_grid_translation_unit.C \
C_gridnodeset.C \
C_gridset.C \
C_index_entry.C \
C_index_set.C \
C_index_set_specifier.C \
C_init_attr_type_edge.C \
C_init_attr_type_node.C \
C_initializable_type_specifier.C \
C_int_constant_list.C \
C_layer_entry.C \
C_layer_name.C \
C_layer_set.C \
C_logical_AND_expression.C \
C_logical_NOT_expression.C \
C_logical_OR_expression.C \
C_matrix_init_declarator.C \
C_matrix_initializer.C \
C_matrix_initializer_clause.C \
C_matrix_initializer_clause_list.C \
C_matrix_initializer_expression.C \
C_matrix_initializer_list.C \
C_matrix_type_specifier.C \
C_name.C \
C_name_range.C \
C_node_type_set_specifier.C \
C_node_type_set_specifier_clause.C \
C_nodeset.C \
C_nodeset_extension.C \
C_non_initializable_type_specifier.C \
C_ndpair_clause.C \
C_ndpair_clause_list.C \
C_ndpair_clause_list_body.C \
C_order.C \
C_parameter_type.C \
C_parameter_type_list.C \
C_parameter_type_pair.C \
C_phase.C \
C_phase_list.C \
C_phase_mapping.C \
C_phase_mapping_list.C \
C_preamble.C \
C_primary_expression.C \
C_production.C \
C_production_adi.C \
C_production_grid.C \
C_query.C \
C_query_field_entry.C \
C_query_field_entry_list.C \
C_query_field_set.C \
C_query_path.C \
C_query_path_product.C \
C_query_list.C \
C_relative_nodeset.C \
C_repertoire_declaration.C \
C_repname.C \
C_return_clause.C \
C_separation_constraint.C \
C_separation_constraint_list.C \
C_service.C \
C_set_operation.C \
C_set_operation_specifier.C \
C_steps.C \
C_stride.C \
C_stride_list.C \
C_system_call.C \
C_trigger.C \
C_trigger_specifier.C \
C_type_definition.C \
C_type_specifier.C \
C_typedef_declaration.C \
C_types.C \
C_value.C \
LensLexer.C \
LensContext.C \
ScriptFunctorType.C \
SymbolTable.C \
ConnectionContext.C \
SimInitializer.C \
SyntaxError.C \
SyntaxErrorException.C \

# define the full pathname for each file
SRC_$(THIS_STEM) = $(patsubst %,$(SRC_PREFIX)/%, $(SOURCES))

THIS_SRC := $(SRC_$(THIS_STEM))
SRC += $(THIS_SRC)

# Create the list of object files by substituting .C with .o
TEMP :=  $(patsubst %.C,%.o,$(filter %.C,$(THIS_SRC)))

# Since the .o files will be in the directory 'obj', while the
# source is in the directory 'source', make this substitution.
# See Gnu Make documentation, section entitled 'Functions for 
# string substition and analysis'
# E.g this creates 'OBJ_datacollect', on which the libdatacollect will depend
OBJ_$(THIS_STEM) := $(subst src,obj,$(TEMP))

OBJS += $(OBJ_$(THIS_STEM))
BASE_OBJECTS += $(OBJ_$(THIS_STEM))
