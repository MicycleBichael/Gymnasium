Ні
Юс
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Ѕ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.10.12v2.10.0-76-gfdfc646704c8Бо
К
actor_critic/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameactor_critic/dense_2/bias
Г
-actor_critic/dense_2/bias/Read/ReadVariableOpReadVariableOpactor_critic/dense_2/bias*
_output_shapes
:*
dtype0
У
actor_critic/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*,
shared_nameactor_critic/dense_2/kernel
М
/actor_critic/dense_2/kernel/Read/ReadVariableOpReadVariableOpactor_critic/dense_2/kernel*
_output_shapes
:	А
*
dtype0
К
actor_critic/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameactor_critic/dense_1/bias
Г
-actor_critic/dense_1/bias/Read/ReadVariableOpReadVariableOpactor_critic/dense_1/bias*
_output_shapes
:*
dtype0
У
actor_critic/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*,
shared_nameactor_critic/dense_1/kernel
М
/actor_critic/dense_1/kernel/Read/ReadVariableOpReadVariableOpactor_critic/dense_1/kernel*
_output_shapes
:	А
*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А
*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:А
*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
РА
*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
РА
*
dtype0
К
serving_default_input_1Placeholder*/
_output_shapes
:€€€€€€€€€*
dtype0*$
shape:€€€€€€€€€
в
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biasactor_critic/dense_1/kernelactor_critic/dense_1/biasactor_critic/dense_2/kernelactor_critic/dense_2/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *.
f)R'
%__inference_signature_wrapper_1103606

NoOpNoOp
ѕ"
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*К"
valueА"Bэ! Bц!
н
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

common
		actor


critic
flatten

signatures*
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
∞
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
trace_0
trace_1
trace_2
trace_3* 
6
trace_0
trace_1
trace_2
trace_3* 
* 
ƒ
 layer-0
!layer_with_weights-0
!layer-1
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses*
¶
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

kernel
bias*
¶
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

kernel
bias*

4	keras_api* 

5serving_default* 
LF
VARIABLE_VALUEdense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEactor_critic/dense_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEactor_critic/dense_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEactor_critic/dense_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEactor_critic/dense_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
	1

2
3*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
О
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses* 
¶
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

kernel
bias*

0
1*

0
1*
* 
У
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*
P
Gtrace_0
Htrace_1
Itrace_2
Jtrace_3
Ktrace_4
Ltrace_5* 
P
Mtrace_0
Ntrace_1
Otrace_2
Ptrace_3
Qtrace_4
Rtrace_5* 

0
1*

0
1*
* 
У
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*

Xtrace_0* 

Ytrace_0* 

0
1*

0
1*
* 
У
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*

_trace_0* 

`trace_0* 
* 
* 
* 
* 
* 
С
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses* 

ftrace_0* 

gtrace_0* 

0
1*

0
1*
* 
У
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses*

mtrace_0* 

ntrace_0* 
* 

 0
!1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
z
StaticRegexFullMatchStaticRegexFullMatchsaver_filename"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
\
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
a
Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
h
SelectSelectStaticRegexFullMatchConst_1Const_2"/device:CPU:**
T0*
_output_shapes
: 
`

StringJoin
StringJoinsaver_filenameSelect"/device:CPU:**
N*
_output_shapes
: 
L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
x
ShardedFilenameShardedFilename
StringJoinShardedFilename/shard
num_shards"/device:CPU:0*
_output_shapes
: 
ъ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*£
valueЩBЦB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
{
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 
ю
SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp/actor_critic/dense_1/kernel/Read/ReadVariableOp-actor_critic/dense_1/bias/Read/ReadVariableOp/actor_critic/dense_2/kernel/Read/ReadVariableOp-actor_critic/dense_2/bias/Read/ReadVariableOpConst"/device:CPU:0*
dtypes
	2
Е
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
o
MergeV2CheckpointsMergeV2Checkpoints&MergeV2Checkpoints/checkpoint_prefixessaver_filename"/device:CPU:0
i
IdentityIdentitysaver_filename^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
э
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*£
valueЩBЦB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
~
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 
≤
	RestoreV2	RestoreV2saver_filenameRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	2
S

Identity_1Identity	RestoreV2"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOpAssignVariableOpdense/kernel
Identity_1"/device:CPU:0*
dtype0
U

Identity_2IdentityRestoreV2:1"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_1AssignVariableOp
dense/bias
Identity_2"/device:CPU:0*
dtype0
U

Identity_3IdentityRestoreV2:2"/device:CPU:0*
T0*
_output_shapes
:
k
AssignVariableOp_2AssignVariableOpactor_critic/dense_1/kernel
Identity_3"/device:CPU:0*
dtype0
U

Identity_4IdentityRestoreV2:3"/device:CPU:0*
T0*
_output_shapes
:
i
AssignVariableOp_3AssignVariableOpactor_critic/dense_1/bias
Identity_4"/device:CPU:0*
dtype0
U

Identity_5IdentityRestoreV2:4"/device:CPU:0*
T0*
_output_shapes
:
k
AssignVariableOp_4AssignVariableOpactor_critic/dense_2/kernel
Identity_5"/device:CPU:0*
dtype0
U

Identity_6IdentityRestoreV2:5"/device:CPU:0*
T0*
_output_shapes
:
i
AssignVariableOp_5AssignVariableOpactor_critic/dense_2/bias
Identity_6"/device:CPU:0*
dtype0

NoOp_1NoOp"/device:CPU:0
џ

Identity_7Identitysaver_filename^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp_1"/device:CPU:0*
T0*
_output_shapes
: йШ
ґ
Ъ
G__inference_sequential_layer_call_and_return_conditional_losses_1103267
flatten_input8
$dense_matmul_readvariableop_resource:
РА
4
%dense_biasadd_readvariableop_resource:	А

identityИҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOp^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  t
flatten/ReshapeReshapeflatten_inputflatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€РВ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
РА
*
dtype0И
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А

dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А
*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А
h
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А
Г
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:^ Z
/
_output_shapes
:€€€€€€€€€
'
_user_specified_nameflatten_input
Ѓ 
Ј
I__inference_actor_critic_layer_call_and_return_conditional_losses_1103714

inputsC
/sequential_dense_matmul_readvariableop_resource:
РА
?
0sequential_dense_biasadd_readvariableop_resource:	А
9
&dense_1_matmul_readvariableop_resource:	А
5
'dense_1_biasadd_readvariableop_resource:9
&dense_2_matmul_readvariableop_resource:	А
5
'dense_2_biasadd_readvariableop_resource:
identity

identity_1ИҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐdense_2/BiasAdd/ReadVariableOpҐdense_2/MatMul/ReadVariableOpҐ'sequential/dense/BiasAdd/ReadVariableOpҐ&sequential/dense/MatMul/ReadVariableOph
sequential/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:€€€€€€€€€i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  Р
sequential/flatten/ReshapeReshapesequential/Cast:y:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€РШ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
РА
*
dtype0©
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
Х
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А
*
dtype0™
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
s
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А
Е
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0Ц
dense_1/MatMulMatMul#sequential/dense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Е
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0Ц
dense_2/MatMulMatMul#sequential/dense/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€i

Identity_1Identitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ы
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
∞	
џ
)__inference_dense_1_layer_call_fn_1103832

inputs1
matmul_readvariableop_resource:	А
-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А

 
_user_specified_nameinputs
Ж
ш
,__inference_sequential_layer_call_fn_1103727

inputs8
$dense_matmul_readvariableop_resource:
РА
4
%dense_biasadd_readvariableop_resource:	А

identityИҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOp^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  m
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€РВ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
РА
*
dtype0И
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А

dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А
*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А
h
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А
Г
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
•

ц
B__inference_dense_layer_call_and_return_conditional_losses_1103896

inputs2
matmul_readvariableop_resource:
РА
.
biasadd_readvariableop_resource:	А

identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
РА
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А
*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А
b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€Р: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€Р
 
_user_specified_nameinputs
ґ
Ъ
G__inference_sequential_layer_call_and_return_conditional_losses_1103254
flatten_input8
$dense_matmul_readvariableop_resource:
РА
4
%dense_biasadd_readvariableop_resource:	А

identityИҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOp^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  t
flatten/ReshapeReshapeflatten_inputflatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€РВ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
РА
*
dtype0И
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А

dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А
*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А
h
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А
Г
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:^ Z
/
_output_shapes
:€€€€€€€€€
'
_user_specified_nameflatten_input
Ћ	
ц
D__inference_dense_1_layer_call_and_return_conditional_losses_1103842

inputs1
matmul_readvariableop_resource:	А
-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А

 
_user_specified_nameinputs
Ж
ш
,__inference_sequential_layer_call_fn_1103740

inputs8
$dense_matmul_readvariableop_resource:
РА
4
%dense_biasadd_readvariableop_resource:	А

identityИҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOp^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  m
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€РВ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
РА
*
dtype0И
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А

dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А
*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А
h
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А
Г
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
°
У
G__inference_sequential_layer_call_and_return_conditional_losses_1103794

inputs8
$dense_matmul_readvariableop_resource:
РА
4
%dense_biasadd_readvariableop_resource:	А

identityИҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOp^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  m
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€РВ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
РА
*
dtype0И
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А

dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А
*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А
h
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А
Г
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
°
У
G__inference_sequential_layer_call_and_return_conditional_losses_1103781

inputs8
$dense_matmul_readvariableop_resource:
РА
4
%dense_biasadd_readvariableop_resource:	А

identityИҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOp^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  m
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€РВ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
РА
*
dtype0И
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А

dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А
*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А
h
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А
Г
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ц 
Э
.__inference_actor_critic_layer_call_fn_1103295
input_1C
/sequential_dense_matmul_readvariableop_resource:
РА
?
0sequential_dense_biasadd_readvariableop_resource:	А
9
&dense_1_matmul_readvariableop_resource:	А
5
'dense_1_biasadd_readvariableop_resource:9
&dense_2_matmul_readvariableop_resource:	А
5
'dense_2_biasadd_readvariableop_resource:
identity

identity_1ИҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐdense_2/BiasAdd/ReadVariableOpҐdense_2/MatMul/ReadVariableOpҐ'sequential/dense/BiasAdd/ReadVariableOpҐ&sequential/dense/MatMul/ReadVariableOpi
sequential/CastCastinput_1*

DstT0*

SrcT0*/
_output_shapes
:€€€€€€€€€i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  Р
sequential/flatten/ReshapeReshapesequential/Cast:y:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€РШ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
РА
*
dtype0©
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
Х
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А
*
dtype0™
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
s
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А
Е
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0Ц
dense_1/MatMulMatMul#sequential/dense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Е
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0Ц
dense_2/MatMulMatMul#sequential/dense/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€i

Identity_1Identitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ы
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp:X T
/
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
М'
≠
"__inference__wrapped_model_1103123
input_1P
<actor_critic_sequential_dense_matmul_readvariableop_resource:
РА
L
=actor_critic_sequential_dense_biasadd_readvariableop_resource:	А
F
3actor_critic_dense_1_matmul_readvariableop_resource:	А
B
4actor_critic_dense_1_biasadd_readvariableop_resource:F
3actor_critic_dense_2_matmul_readvariableop_resource:	А
B
4actor_critic_dense_2_biasadd_readvariableop_resource:
identity

identity_1ИҐ+actor_critic/dense_1/BiasAdd/ReadVariableOpҐ*actor_critic/dense_1/MatMul/ReadVariableOpҐ+actor_critic/dense_2/BiasAdd/ReadVariableOpҐ*actor_critic/dense_2/MatMul/ReadVariableOpҐ4actor_critic/sequential/dense/BiasAdd/ReadVariableOpҐ3actor_critic/sequential/dense/MatMul/ReadVariableOpv
actor_critic/sequential/CastCastinput_1*

DstT0*

SrcT0*/
_output_shapes
:€€€€€€€€€v
%actor_critic/sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  Ј
'actor_critic/sequential/flatten/ReshapeReshape actor_critic/sequential/Cast:y:0.actor_critic/sequential/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р≤
3actor_critic/sequential/dense/MatMul/ReadVariableOpReadVariableOp<actor_critic_sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
РА
*
dtype0–
$actor_critic/sequential/dense/MatMulMatMul0actor_critic/sequential/flatten/Reshape:output:0;actor_critic/sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
ѓ
4actor_critic/sequential/dense/BiasAdd/ReadVariableOpReadVariableOp=actor_critic_sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А
*
dtype0—
%actor_critic/sequential/dense/BiasAddBiasAdd.actor_critic/sequential/dense/MatMul:product:0<actor_critic/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
Н
"actor_critic/sequential/dense/ReluRelu.actor_critic/sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А
Я
*actor_critic/dense_1/MatMul/ReadVariableOpReadVariableOp3actor_critic_dense_1_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0љ
actor_critic/dense_1/MatMulMatMul0actor_critic/sequential/dense/Relu:activations:02actor_critic/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ь
+actor_critic/dense_1/BiasAdd/ReadVariableOpReadVariableOp4actor_critic_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
actor_critic/dense_1/BiasAddBiasAdd%actor_critic/dense_1/MatMul:product:03actor_critic/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Я
*actor_critic/dense_2/MatMul/ReadVariableOpReadVariableOp3actor_critic_dense_2_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0љ
actor_critic/dense_2/MatMulMatMul0actor_critic/sequential/dense/Relu:activations:02actor_critic/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ь
+actor_critic/dense_2/BiasAdd/ReadVariableOpReadVariableOp4actor_critic_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
actor_critic/dense_2/BiasAddBiasAdd%actor_critic/dense_2/MatMul:product:03actor_critic/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€t
IdentityIdentity%actor_critic/dense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€v

Identity_1Identity%actor_critic/dense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€й
NoOpNoOp,^actor_critic/dense_1/BiasAdd/ReadVariableOp+^actor_critic/dense_1/MatMul/ReadVariableOp,^actor_critic/dense_2/BiasAdd/ReadVariableOp+^actor_critic/dense_2/MatMul/ReadVariableOp5^actor_critic/sequential/dense/BiasAdd/ReadVariableOp4^actor_critic/sequential/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2Z
+actor_critic/dense_1/BiasAdd/ReadVariableOp+actor_critic/dense_1/BiasAdd/ReadVariableOp2X
*actor_critic/dense_1/MatMul/ReadVariableOp*actor_critic/dense_1/MatMul/ReadVariableOp2Z
+actor_critic/dense_2/BiasAdd/ReadVariableOp+actor_critic/dense_2/BiasAdd/ReadVariableOp2X
*actor_critic/dense_2/MatMul/ReadVariableOp*actor_critic/dense_2/MatMul/ReadVariableOp2l
4actor_critic/sequential/dense/BiasAdd/ReadVariableOp4actor_critic/sequential/dense/BiasAdd/ReadVariableOp2j
3actor_critic/sequential/dense/MatMul/ReadVariableOp3actor_critic/sequential/dense/MatMul/ReadVariableOp:X T
/
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
У 
Ь
.__inference_actor_critic_layer_call_fn_1103633

inputsC
/sequential_dense_matmul_readvariableop_resource:
РА
?
0sequential_dense_biasadd_readvariableop_resource:	А
9
&dense_1_matmul_readvariableop_resource:	А
5
'dense_1_biasadd_readvariableop_resource:9
&dense_2_matmul_readvariableop_resource:	А
5
'dense_2_biasadd_readvariableop_resource:
identity

identity_1ИҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐdense_2/BiasAdd/ReadVariableOpҐdense_2/MatMul/ReadVariableOpҐ'sequential/dense/BiasAdd/ReadVariableOpҐ&sequential/dense/MatMul/ReadVariableOph
sequential/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:€€€€€€€€€i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  Р
sequential/flatten/ReshapeReshapesequential/Cast:y:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€РШ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
РА
*
dtype0©
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
Х
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А
*
dtype0™
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
s
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А
Е
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0Ц
dense_1/MatMulMatMul#sequential/dense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Е
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0Ц
dense_2/MatMulMatMul#sequential/dense/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€i

Identity_1Identitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ы
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
∞	
џ
)__inference_dense_2_layer_call_fn_1103852

inputs1
matmul_readvariableop_resource:	А
-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А

 
_user_specified_nameinputs
Ђ
E
)__inference_flatten_layer_call_fn_1103868

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€РY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
з
ш
,__inference_sequential_layer_call_fn_1103754

inputs8
$dense_matmul_readvariableop_resource:
РА
4
%dense_biasadd_readvariableop_resource:	А

identityИҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOp]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:€€€€€€€€€^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  o
flatten/ReshapeReshapeCast:y:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€РВ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
РА
*
dtype0И
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А

dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А
*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А
h
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А
Г
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
У 
Ь
.__inference_actor_critic_layer_call_fn_1103660

inputsC
/sequential_dense_matmul_readvariableop_resource:
РА
?
0sequential_dense_biasadd_readvariableop_resource:	А
9
&dense_1_matmul_readvariableop_resource:	А
5
'dense_1_biasadd_readvariableop_resource:9
&dense_2_matmul_readvariableop_resource:	А
5
'dense_2_biasadd_readvariableop_resource:
identity

identity_1ИҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐdense_2/BiasAdd/ReadVariableOpҐdense_2/MatMul/ReadVariableOpҐ'sequential/dense/BiasAdd/ReadVariableOpҐ&sequential/dense/MatMul/ReadVariableOph
sequential/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:€€€€€€€€€i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  Р
sequential/flatten/ReshapeReshapesequential/Cast:y:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€РШ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
РА
*
dtype0©
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
Х
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А
*
dtype0™
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
s
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А
Е
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0Ц
dense_1/MatMulMatMul#sequential/dense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Е
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0Ц
dense_2/MatMulMatMul#sequential/dense/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€i

Identity_1Identitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ы
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
± 
Є
I__inference_actor_critic_layer_call_and_return_conditional_losses_1103558
input_1C
/sequential_dense_matmul_readvariableop_resource:
РА
?
0sequential_dense_biasadd_readvariableop_resource:	А
9
&dense_1_matmul_readvariableop_resource:	А
5
'dense_1_biasadd_readvariableop_resource:9
&dense_2_matmul_readvariableop_resource:	А
5
'dense_2_biasadd_readvariableop_resource:
identity

identity_1ИҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐdense_2/BiasAdd/ReadVariableOpҐdense_2/MatMul/ReadVariableOpҐ'sequential/dense/BiasAdd/ReadVariableOpҐ&sequential/dense/MatMul/ReadVariableOpi
sequential/CastCastinput_1*

DstT0*

SrcT0*/
_output_shapes
:€€€€€€€€€i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  Р
sequential/flatten/ReshapeReshapesequential/Cast:y:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€РШ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
РА
*
dtype0©
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
Х
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А
*
dtype0™
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
s
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А
Е
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0Ц
dense_1/MatMulMatMul#sequential/dense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Е
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0Ц
dense_2/MatMulMatMul#sequential/dense/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€i

Identity_1Identitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ы
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp:X T
/
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
Ц 
Э
.__inference_actor_critic_layer_call_fn_1103531
input_1C
/sequential_dense_matmul_readvariableop_resource:
РА
?
0sequential_dense_biasadd_readvariableop_resource:	А
9
&dense_1_matmul_readvariableop_resource:	А
5
'dense_1_biasadd_readvariableop_resource:9
&dense_2_matmul_readvariableop_resource:	А
5
'dense_2_biasadd_readvariableop_resource:
identity

identity_1ИҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐdense_2/BiasAdd/ReadVariableOpҐdense_2/MatMul/ReadVariableOpҐ'sequential/dense/BiasAdd/ReadVariableOpҐ&sequential/dense/MatMul/ReadVariableOpi
sequential/CastCastinput_1*

DstT0*

SrcT0*/
_output_shapes
:€€€€€€€€€i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  Р
sequential/flatten/ReshapeReshapesequential/Cast:y:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€РШ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
РА
*
dtype0©
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
Х
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А
*
dtype0™
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
s
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А
Е
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0Ц
dense_1/MatMulMatMul#sequential/dense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Е
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0Ц
dense_2/MatMulMatMul#sequential/dense/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€i

Identity_1Identitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ы
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp:X T
/
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
∆
`
D__inference_flatten_layer_call_and_return_conditional_losses_1103874

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€РY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ћ	
ц
D__inference_dense_2_layer_call_and_return_conditional_losses_1103862

inputs1
matmul_readvariableop_resource:	А
-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А

 
_user_specified_nameinputs
К

џ
'__inference_dense_layer_call_fn_1103885

inputs2
matmul_readvariableop_resource:
РА
.
biasadd_readvariableop_resource:	А

identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
РА
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А
*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А
b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€Р: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€Р
 
_user_specified_nameinputs
В
У
G__inference_sequential_layer_call_and_return_conditional_losses_1103808

inputs8
$dense_matmul_readvariableop_resource:
РА
4
%dense_biasadd_readvariableop_resource:	А

identityИҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOp]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:€€€€€€€€€^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  o
flatten/ReshapeReshapeCast:y:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€РВ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
РА
*
dtype0И
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А

dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А
*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А
h
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А
Г
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ѓ 
Ј
I__inference_actor_critic_layer_call_and_return_conditional_losses_1103687

inputsC
/sequential_dense_matmul_readvariableop_resource:
РА
?
0sequential_dense_biasadd_readvariableop_resource:	А
9
&dense_1_matmul_readvariableop_resource:	А
5
'dense_1_biasadd_readvariableop_resource:9
&dense_2_matmul_readvariableop_resource:	А
5
'dense_2_biasadd_readvariableop_resource:
identity

identity_1ИҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐdense_2/BiasAdd/ReadVariableOpҐdense_2/MatMul/ReadVariableOpҐ'sequential/dense/BiasAdd/ReadVariableOpҐ&sequential/dense/MatMul/ReadVariableOph
sequential/CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:€€€€€€€€€i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  Р
sequential/flatten/ReshapeReshapesequential/Cast:y:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€РШ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
РА
*
dtype0©
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
Х
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А
*
dtype0™
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
s
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А
Е
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0Ц
dense_1/MatMulMatMul#sequential/dense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Е
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0Ц
dense_2/MatMulMatMul#sequential/dense/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€i

Identity_1Identitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ы
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
± 
Є
I__inference_actor_critic_layer_call_and_return_conditional_losses_1103585
input_1C
/sequential_dense_matmul_readvariableop_resource:
РА
?
0sequential_dense_biasadd_readvariableop_resource:	А
9
&dense_1_matmul_readvariableop_resource:	А
5
'dense_1_biasadd_readvariableop_resource:9
&dense_2_matmul_readvariableop_resource:	А
5
'dense_2_biasadd_readvariableop_resource:
identity

identity_1ИҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐdense_2/BiasAdd/ReadVariableOpҐdense_2/MatMul/ReadVariableOpҐ'sequential/dense/BiasAdd/ReadVariableOpҐ&sequential/dense/MatMul/ReadVariableOpi
sequential/CastCastinput_1*

DstT0*

SrcT0*/
_output_shapes
:€€€€€€€€€i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  Р
sequential/flatten/ReshapeReshapesequential/Cast:y:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€РШ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
РА
*
dtype0©
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
Х
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А
*
dtype0™
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
s
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А
Е
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0Ц
dense_1/MatMulMatMul#sequential/dense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Е
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype0Ц
dense_2/MatMulMatMul#sequential/dense/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€i

Identity_1Identitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ы
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp:X T
/
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
з
ш
,__inference_sequential_layer_call_fn_1103768

inputs8
$dense_matmul_readvariableop_resource:
РА
4
%dense_biasadd_readvariableop_resource:	А

identityИҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOp]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:€€€€€€€€€^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  o
flatten/ReshapeReshapeCast:y:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€РВ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
РА
*
dtype0И
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А

dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А
*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А
h
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А
Г
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ы
€
,__inference_sequential_layer_call_fn_1103137
flatten_input8
$dense_matmul_readvariableop_resource:
РА
4
%dense_biasadd_readvariableop_resource:	А

identityИҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOp^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  t
flatten/ReshapeReshapeflatten_inputflatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€РВ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
РА
*
dtype0И
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А

dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А
*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А
h
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А
Г
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:^ Z
/
_output_shapes
:€€€€€€€€€
'
_user_specified_nameflatten_input
В
У
G__inference_sequential_layer_call_and_return_conditional_losses_1103822

inputs8
$dense_matmul_readvariableop_resource:
РА
4
%dense_biasadd_readvariableop_resource:	А

identityИҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOp]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:€€€€€€€€€^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  o
flatten/ReshapeReshapeCast:y:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€РВ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
РА
*
dtype0И
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А

dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А
*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А
h
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А
Г
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ы

Ф
%__inference_signature_wrapper_1103606
input_1
unknown:
РА

	unknown_0:	А

	unknown_1:	А

	unknown_2:
	unknown_3:	А

	unknown_4:
identity

identity_1ИҐStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *+
f&R$
"__inference__wrapped_model_1103123o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
Ы
€
,__inference_sequential_layer_call_fn_1103241
flatten_input8
$dense_matmul_readvariableop_resource:
РА
4
%dense_biasadd_readvariableop_resource:	А

identityИҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOp^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  t
flatten/ReshapeReshapeflatten_inputflatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€РВ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
РА
*
dtype0И
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А

dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А
*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А
]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А
h
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А
Г
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:^ Z
/
_output_shapes
:€€€€€€€€€
'
_user_specified_nameflatten_input"µ	,
saver_filename:0
Identity:0
Identity_78"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*с
serving_defaultЁ
C
input_18
serving_default_input_1:0€€€€€€€€€<
output_10
StatefulPartitionedCall:0€€€€€€€€€<
output_20
StatefulPartitionedCall:1€€€€€€€€€tensorflow/serving/predict:а≥
В
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

common
		actor


critic
flatten

signatures"
_tf_keras_model
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
н
trace_0
trace_1
trace_2
trace_32В
.__inference_actor_critic_layer_call_fn_1103295
.__inference_actor_critic_layer_call_fn_1103633
.__inference_actor_critic_layer_call_fn_1103660
.__inference_actor_critic_layer_call_fn_1103531њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 ztrace_0ztrace_1ztrace_2ztrace_3
ў
trace_0
trace_1
trace_2
trace_32о
I__inference_actor_critic_layer_call_and_return_conditional_losses_1103687
I__inference_actor_critic_layer_call_and_return_conditional_losses_1103714
I__inference_actor_critic_layer_call_and_return_conditional_losses_1103558
I__inference_actor_critic_layer_call_and_return_conditional_losses_1103585њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 ztrace_0ztrace_1ztrace_2ztrace_3
ЌB 
"__inference__wrapped_model_1103123input_1"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ё
 layer-0
!layer_with_weights-0
!layer-1
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_sequential
ї
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
ї
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
(
4	keras_api"
_tf_keras_layer
,
5serving_default"
signature_map
 :
РА
2dense/kernel
:А
2
dense/bias
.:,	А
2actor_critic/dense_1/kernel
':%2actor_critic/dense_1/bias
.:,	А
2actor_critic/dense_2/kernel
':%2actor_critic/dense_2/bias
 "
trackable_list_wrapper
<
0
	1

2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
АBэ
.__inference_actor_critic_layer_call_fn_1103295input_1"њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
€Bь
.__inference_actor_critic_layer_call_fn_1103633inputs"њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
€Bь
.__inference_actor_critic_layer_call_fn_1103660inputs"њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
АBэ
.__inference_actor_critic_layer_call_fn_1103531input_1"њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ЪBЧ
I__inference_actor_critic_layer_call_and_return_conditional_losses_1103687inputs"њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ЪBЧ
I__inference_actor_critic_layer_call_and_return_conditional_losses_1103714inputs"њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ЫBШ
I__inference_actor_critic_layer_call_and_return_conditional_losses_1103558input_1"њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ЫBШ
I__inference_actor_critic_layer_call_and_return_conditional_losses_1103585input_1"њ
ґ≤≤
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
•
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
ї
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
х
Gtrace_0
Htrace_1
Itrace_2
Jtrace_3
Ktrace_4
Ltrace_52÷
,__inference_sequential_layer_call_fn_1103137
,__inference_sequential_layer_call_fn_1103727
,__inference_sequential_layer_call_fn_1103740
,__inference_sequential_layer_call_fn_1103241
,__inference_sequential_layer_call_fn_1103754
,__inference_sequential_layer_call_fn_1103768њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zGtrace_0zHtrace_1zItrace_2zJtrace_3zKtrace_4zLtrace_5
Ч
Mtrace_0
Ntrace_1
Otrace_2
Ptrace_3
Qtrace_4
Rtrace_52ш
G__inference_sequential_layer_call_and_return_conditional_losses_1103781
G__inference_sequential_layer_call_and_return_conditional_losses_1103794
G__inference_sequential_layer_call_and_return_conditional_losses_1103254
G__inference_sequential_layer_call_and_return_conditional_losses_1103267
G__inference_sequential_layer_call_and_return_conditional_losses_1103808
G__inference_sequential_layer_call_and_return_conditional_losses_1103822њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zMtrace_0zNtrace_1zOtrace_2zPtrace_3zQtrace_4zRtrace_5
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
н
Xtrace_02–
)__inference_dense_1_layer_call_fn_1103832Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zXtrace_0
И
Ytrace_02л
D__inference_dense_1_layer_call_and_return_conditional_losses_1103842Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zYtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
н
_trace_02–
)__inference_dense_2_layer_call_fn_1103852Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z_trace_0
И
`trace_02л
D__inference_dense_2_layer_call_and_return_conditional_losses_1103862Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z`trace_0
"
_generic_user_object
ћB…
%__inference_signature_wrapper_1103606input_1"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
н
ftrace_02–
)__inference_flatten_layer_call_fn_1103868Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zftrace_0
И
gtrace_02л
D__inference_flatten_layer_call_and_return_conditional_losses_1103874Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zgtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
л
mtrace_02ќ
'__inference_dense_layer_call_fn_1103885Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zmtrace_0
Ж
ntrace_02й
B__inference_dense_layer_call_and_return_conditional_losses_1103896Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zntrace_0
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ДBБ
,__inference_sequential_layer_call_fn_1103137flatten_input"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
эBъ
,__inference_sequential_layer_call_fn_1103727inputs"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
эBъ
,__inference_sequential_layer_call_fn_1103740inputs"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ДBБ
,__inference_sequential_layer_call_fn_1103241flatten_input"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
эBъ
,__inference_sequential_layer_call_fn_1103754inputs"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
эBъ
,__inference_sequential_layer_call_fn_1103768inputs"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ШBХ
G__inference_sequential_layer_call_and_return_conditional_losses_1103781inputs"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ШBХ
G__inference_sequential_layer_call_and_return_conditional_losses_1103794inputs"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЯBЬ
G__inference_sequential_layer_call_and_return_conditional_losses_1103254flatten_input"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЯBЬ
G__inference_sequential_layer_call_and_return_conditional_losses_1103267flatten_input"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ШBХ
G__inference_sequential_layer_call_and_return_conditional_losses_1103808inputs"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ШBХ
G__inference_sequential_layer_call_and_return_conditional_losses_1103822inputs"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЁBЏ
)__inference_dense_1_layer_call_fn_1103832inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
шBх
D__inference_dense_1_layer_call_and_return_conditional_losses_1103842inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЁBЏ
)__inference_dense_2_layer_call_fn_1103852inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
шBх
D__inference_dense_2_layer_call_and_return_conditional_losses_1103862inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЁBЏ
)__inference_flatten_layer_call_fn_1103868inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
шBх
D__inference_flatten_layer_call_and_return_conditional_losses_1103874inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
џBЎ
'__inference_dense_layer_call_fn_1103885inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
цBу
B__inference_dense_layer_call_and_return_conditional_losses_1103896inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 ќ
"__inference__wrapped_model_1103123І8Ґ5
.Ґ+
)К&
input_1€€€€€€€€€
™ "c™`
.
output_1"К
output_1€€€€€€€€€
.
output_2"К
output_2€€€€€€€€€н
I__inference_actor_critic_layer_call_and_return_conditional_losses_1103558ЯHҐE
.Ґ+
)К&
input_1€€€€€€€€€
™

trainingp "KҐH
AҐ>
К
0/0€€€€€€€€€
К
0/1€€€€€€€€€
Ъ н
I__inference_actor_critic_layer_call_and_return_conditional_losses_1103585ЯHҐE
.Ґ+
)К&
input_1€€€€€€€€€
™

trainingp"KҐH
AҐ>
К
0/0€€€€€€€€€
К
0/1€€€€€€€€€
Ъ м
I__inference_actor_critic_layer_call_and_return_conditional_losses_1103687ЮGҐD
-Ґ*
(К%
inputs€€€€€€€€€
™

trainingp "KҐH
AҐ>
К
0/0€€€€€€€€€
К
0/1€€€€€€€€€
Ъ м
I__inference_actor_critic_layer_call_and_return_conditional_losses_1103714ЮGҐD
-Ґ*
(К%
inputs€€€€€€€€€
™

trainingp"KҐH
AҐ>
К
0/0€€€€€€€€€
К
0/1€€€€€€€€€
Ъ ƒ
.__inference_actor_critic_layer_call_fn_1103295СHҐE
.Ґ+
)К&
input_1€€€€€€€€€
™

trainingp "=Ґ:
К
0€€€€€€€€€
К
1€€€€€€€€€ƒ
.__inference_actor_critic_layer_call_fn_1103531СHҐE
.Ґ+
)К&
input_1€€€€€€€€€
™

trainingp"=Ґ:
К
0€€€€€€€€€
К
1€€€€€€€€€√
.__inference_actor_critic_layer_call_fn_1103633РGҐD
-Ґ*
(К%
inputs€€€€€€€€€
™

trainingp "=Ґ:
К
0€€€€€€€€€
К
1€€€€€€€€€√
.__inference_actor_critic_layer_call_fn_1103660РGҐD
-Ґ*
(К%
inputs€€€€€€€€€
™

trainingp"=Ґ:
К
0€€€€€€€€€
К
1€€€€€€€€€•
D__inference_dense_1_layer_call_and_return_conditional_losses_1103842]0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А

™ "%Ґ"
К
0€€€€€€€€€
Ъ }
)__inference_dense_1_layer_call_fn_1103832P0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А

™ "К€€€€€€€€€•
D__inference_dense_2_layer_call_and_return_conditional_losses_1103862]0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А

™ "%Ґ"
К
0€€€€€€€€€
Ъ }
)__inference_dense_2_layer_call_fn_1103852P0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А

™ "К€€€€€€€€€§
B__inference_dense_layer_call_and_return_conditional_losses_1103896^0Ґ-
&Ґ#
!К
inputs€€€€€€€€€Р
™ "&Ґ#
К
0€€€€€€€€€А

Ъ |
'__inference_dense_layer_call_fn_1103885Q0Ґ-
&Ґ#
!К
inputs€€€€€€€€€Р
™ "К€€€€€€€€€А
©
D__inference_flatten_layer_call_and_return_conditional_losses_1103874a7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "&Ґ#
К
0€€€€€€€€€Р
Ъ Б
)__inference_flatten_layer_call_fn_1103868T7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "К€€€€€€€€€Рњ
G__inference_sequential_layer_call_and_return_conditional_losses_1103254tFҐC
<Ґ9
/К,
flatten_input€€€€€€€€€
p 

 
™ "&Ґ#
К
0€€€€€€€€€А

Ъ њ
G__inference_sequential_layer_call_and_return_conditional_losses_1103267tFҐC
<Ґ9
/К,
flatten_input€€€€€€€€€
p

 
™ "&Ґ#
К
0€€€€€€€€€А

Ъ Є
G__inference_sequential_layer_call_and_return_conditional_losses_1103781m?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ "&Ґ#
К
0€€€€€€€€€А

Ъ Є
G__inference_sequential_layer_call_and_return_conditional_losses_1103794m?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ "&Ґ#
К
0€€€€€€€€€А

Ъ Є
G__inference_sequential_layer_call_and_return_conditional_losses_1103808m?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ "&Ґ#
К
0€€€€€€€€€А

Ъ Є
G__inference_sequential_layer_call_and_return_conditional_losses_1103822m?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ "&Ґ#
К
0€€€€€€€€€А

Ъ Ч
,__inference_sequential_layer_call_fn_1103137gFҐC
<Ґ9
/К,
flatten_input€€€€€€€€€
p 

 
™ "К€€€€€€€€€А
Ч
,__inference_sequential_layer_call_fn_1103241gFҐC
<Ґ9
/К,
flatten_input€€€€€€€€€
p

 
™ "К€€€€€€€€€А
Р
,__inference_sequential_layer_call_fn_1103727`?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ "К€€€€€€€€€А
Р
,__inference_sequential_layer_call_fn_1103740`?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ "К€€€€€€€€€А
Р
,__inference_sequential_layer_call_fn_1103754`?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ "К€€€€€€€€€А
Р
,__inference_sequential_layer_call_fn_1103768`?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ "К€€€€€€€€€А
№
%__inference_signature_wrapper_1103606≤CҐ@
Ґ 
9™6
4
input_1)К&
input_1€€€€€€€€€"c™`
.
output_1"К
output_1€€€€€€€€€
.
output_2"К
output_2€€€€€€€€€