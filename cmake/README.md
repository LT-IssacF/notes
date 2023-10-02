## cmake命令笔记 ##
---
## 配置 & 构建 ##
```batch
cmake -B build
cmake --build build
```
配置阶段可以使用 -D 设置缓存变量，下次配置时，之前添加的缓存变量依然存在
```batch
cmake -B build -DCMAKE_INSTALL_PREFIX=./build
cmake -B build -DCMAKE_BUILD_TYPE=Release
```
## 基础 ##
```cmake
cmake_minimum_required(<VERSION 版本号>)

project(<project>
        [VERSION 项目版本号]
        [LANGUAGES 使用语言]
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
if(WIN32)
    add_definitions(-DNOMINMAX -D_USE_MATH_DEFINES)
endif()
```
---
## 常用命令
#### configure_file() ####
```cmake
configure_file(<input> <output>)
```
此命令可以配合后面打包发布使用，主要目的是配置程序的发行版本号
一般配置文件 .h.in 中这么写
```c++
#define PROJECT_VERSION_MAJOR @PROJECT_VERSION_MAJOR@
#define PROJECT_VERSION_MINOR @PROJECT_VERSION_MINOR@
#define PROJECT_VERSION_PATCH @PROJECT_VERSION_PATCH@
```

#### message() ####
```cmake
message([<mode>] "message text" ...)
```
mode 参数主要有 `FATAL_ERROR` `SEND_ERROR` `WARNING` `AUTHOR_WARNING` `DEPRECATION` `STATUS` `VERBOSE` `DEBUG` `TRACE`，一般最常使用 `STATUS` 或 `NONE`

#### option() ####
```cmake
option(<variable> "<help_text>" [value])
```
提供一个 BOOL 缓存变量，例如
```cmake
option(USE_MYMATH "Use tutorial provided math implementation" ON)
# 等价于 set(USE_MYMATH ON CACHE BOOL "Use tutorial provided math implementation" FORCE)
# FORCE 会强制在再次构建时更新缓存变量
if(USE_MYMATH)
    add_subdirectory(MathFunctions)
    list(APPEND EXTRA_LIBS MathFunctions)
    list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/MathFunctions")
endif()

target_link_libraries(Tutorial PUBLIC ${EXTRA_LIBS})
target_include_directories(Tutorial PUBLIC "${PROJECT_BINARY_DIR}" ${EXTRA_INCLUDES})

# .h.in
#cmakedefine USE_MYMATH

# .cpp
#ifdef USE_MYMATH
    #include "MathFunctions.h"
#endif
```

#### list() ####
```cmake
list(<mode> <varName> <directories>)
```
list 作用和 set 相似，大致意思是设置一个清单类型的变量供使用。mode 的常用主要参数有 `APPEND`

#### target_compile_definitions() ####
```cmake
target_compile_definitions(<target>
    <INTERFACE|PUBLIC|PRIVATE> [items1...]
    [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...])
```
向目标添加预设宏声明或定义，例如
```cpp
# CMakeLists.txt
target_compile_definitions(main PUBLIC PI=3.14)

// .cpp
std::cout << PI << std::endl;
```

#### target_compile_features() ####
```cmake
target_compile_features(<target> <PRIVATE|PUBLIC|INTERFACE> <feature> [...])
```
此函数专门向目标设置 C/C++ 编译特征：
~~`set(CMAKE_CXX_STANDARD 17)`~~
~~`set(CMAKE_CXX_STANDARD_REQUIRED TRUE)`~~
```cmake
add_library(project_compiler_flags INTERFACE)
target_compile_features(project_compiler_flags INTERFACE cxx_std_17)

target_link_libraries(<target> PUBLIC project_compiler_flags)
```

#### target_compile_options() ####
```cmake
target_compile_options(<target> [BEFORE]
    <INTERFACE|PUBLIC|PRIVATE> [items1...]
    [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...])
```
[向目标添加编译选项](#生成器表达式)

#### add_executable() ####
```cmake
add_executable(<tarhet> [WIN32] [MACOSX_BUNDLE]
    [EXCLUDE_FROM_ALL]
    [source1] [source2 ...]
)
```
该命令用于生成一个可执行程序，第一个参数为程序名且必需，后几个参数暂时不重要可省略，最后为生成所需的源码文件

#### target_source() ####
```cmake
target_source(<target> [PUBLIC|INTERFACE|PRIVATE] <source>)
```
这个命令可以为生成 target 的命令补充依赖文件

#### file() ####
```cmake
file(GLOB <source> [CONFIGURE_DEPENDS] *.cpp *.h)
target_source(<target> [PUBLIC|INTERFACE|PRIVATE] <${source}>)
```
使用 GLOB 参数可以自动查找目录下指定扩展名的文件，实现批量添加。CONFIGURE_DEPENDS 为可选参数，目的在于实现有新文件变动时，自动更新变量

#### aux_source_directory() ####
```cmake
aux_source_directory(<directory> <source>)
target_source(<target> [PUBLIC|INTERFACE|PRIVATE] <${source}>)
```
它会根据生成 target 的语言类型，自动在目标 directory 下寻找对应类型的文件

#### add_subdirectory() ####
```cmake
add_subdirectory(<path>)
```
告诉构建系统在子目录下继续寻找 `CMakeLists.txt`，产生一个新的变量作用域

#### include() ####
```cmake
include(<file> [OPTIONAL] [RESULT_VARIABLE myVar] [NO_POLICY_SCOPE])
```
此命令将一个新的 cmake 内容引入到当前 cmake 内容中，通常为 .cmake 文件，且不会引入新的变量作用域

#### add_library() ####
```cmake
add_library(<target> [STATIC|SHARED|MODULE]
    [EXCLUDE_FROM_ALL]
    [<source>...]
)
```
生成一个库文件
* 易错点：动态库链接静态库
```cmake
add_library(other_lib STATIC other_lib.cpp)
set_property(TARGET other_lib PERPERTY POSITION_INDEPENDENT_CODE ON)
# 因为静态库动态库重定位的时间不同，直接链接会出问题，需要加上 PIC 无关
add_library(my_lib SHARED my_lib.cpp)
target_link_libraries(my_lib PUBLIC other_lib)
```
* windows平台下运行 .exe 时如果需要 .dll，只会在 .exe 所在目录和 PATH 里寻找
```cmake
set_property(TARGET <target> PROPERTY ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set_property(TARGET <target> PROPERTY LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set_property(TARGET <target> PROPERTY RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
```

#### target_link_libraries() ####
```cmake
target_link_libraries(<target>
    <PRIVATE|PUBLIC|INTERFACE> item1 [item2 ...]
    [<PRIVATE|PUBLIC|INTERFACE> item3 [item4 ...]]
    ...
)
```
指定程序在编译阶段需要链接的库，`target` 可以是 `add_executable()` 和 `add_library()` 创建的文件，库既可以是自己生成的，也可以是外部导入的库

#### target_include_directories() ####
```cmake
target_include_directories(<target> [SYSTEM] [AFTER|BEFORE]
    <INTERFACE|PUBLIC|PRIVATE> [items1...]
    [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...]
)
```
编译 target 时为其添加 include 目录
`INTERFACE` 任何链接到 target 的都需要 include `items`，而 target 本身并不需要，即消费者需要但生产者不需要

---
## 变量 ##
#### 普通变量 ####
```cmake
set(varName value... [PARENT_SCOPE])
```
* cmake 将所有的变量的值视为字符串，普通变量的作用域是局部的，父传子，子不传父（可通过 `PARENT_SCOPE` 传递)
* 当给出变量的值不包含空格的时候，可以不使用引号，但建议都加上引号，不然一些奇怪的问题很难调试
* cmake 使用空格或者分号作为字符串的分隔符
* cmake 中想要获取变量的值，和 shell 脚本一样，采用 `${var}` 形式
* 使用 cmake 变量前，不需要这个变量已经定义了，如果使用了未定义的变量，那么它的值是一个空字符串
    * 默认情况下使用未定义的变量不会有警告信息，但是可以通过 cmake 的 -warn-uninitialized 选项启用警告信息
    * 使用未定义的变量非常常见，如果出现问题也不一定是因为变量未定义导致的，所以 cmake 的 -warn-uninitialized 选项用处很有限
* 变量的值可以包含换行，也可以包含引号，不过需要转义
```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(SOURCES ./main.cxx)
```
#### 环境变量 ####
在当前项目中定义类似于操作系统环境变量的变量，定义与使用与普通变量基本一样，只是需要在前面加上 `ENV` 字符以及一对 `{}`，环境变量的作用域是全局的
```cmake
set(ENV{PATH} "$/home/usr")
message(STATUS "PATH=$ENV{PATH}")
```

#### 缓存变量 ####
与普通变量不同，缓存变量的值可以缓存到 `CMakeCache.txt` 文件中，当再次运行 cmake 时，可以从中获取上一次的值，而不是重新去评估。所以缓存变量的作用域是**全局**的
和普通变量比起来，缓存变量携带了更多的信息。缓存变量有类型了，而且可以为缓存变量添加一个说明信息
```cmake
set(varName value... CACHE type "docstring" [FORCE])
```
第四个参数 `type` 是必选参数，而且其值必须是下列值之一
* BOOL
    * BOOL 类型的变量值如果是 ON、TRUE、1 则被评估为真，如果是 OFF、FALSE、0 则被评估为假
    * 当然除了上面列出的值还有其他值，但是判断真假就没那么清晰了，所以建议定义 BOOL 类型的缓存变量的时候，其值就采用上述列出的值。虽然不区分大小写，建议统一使用大写
* FILEPATH
    * 文件路径
* STRING
  * 字符串
* INTERNAL
    * 内部缓存变量不会对用户可见，一般是项目为了缓存某种内部信息时才使用，cmake 图形化界面工具也对其不可见
    * 内部缓存变量默认是 `FORCE` 的
* `FORCE` 关键字代表每次运行都强制更新缓存变量的值，如果没有该关键字，当再次运行 cmake 的时候，cmake 将使用 `CMakeCache.txt` 文件中缓存的值，而不是重新进行评估

第五个参数是一个说明性的字符串，可以为空，只在图形化 cmake 界面会展示
由于 BOOL 类型的变量使用频率非常高，CMake 为其单独提供了一条命令
```cmake
option(optVar helpString [initialValue])
```
其等价于
```cmake
set(optVar initialValue CACHE BOOL helpString)
```
不同之处在于`option()`命令没有 FORCE 关键字

---
## 生成器表达式 ##
使用生成器表达式需要 cmake 版本至少 3.15

    $<$<类型:值>:为真时的表达式>
利用生成器表达式实现只在**构建期间**使用编译器警告标志
```cmake
target_compile_definitions(<target> PUBLIC
    $<$<PLATFORM_ID:Linux>:expr>
)
set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")
set(msvc_cxx "$<COMPILE_LANG_AND_ID:CXX,MSVC>")
target_compile_options(project_compiler_flags INTERFACE
    "$<${gcc_like_cxx}:$<BUILD_INTERFACE:-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused>>"
    "$<${msvc_cxx}:$<BUILD_INTERFACE:-W3>>"
)
```

---
## 测试 ##
```cmake
include(CTest) # 或 enable_testing()
add_test(NAME <testName> COMMAND <project> [arg])
set_tests_properties(<testName> PROPERTIES PASS_REGULAR_EXPRESSION "result")
```
执行如下命令

    ctest -C Debug -VV

---
## 函数 ##
根据上面 测试 部分，可以为其制作一个专用的测试函数
```cmake
function(do_test target arg result)
    add_test(NAME Comp${arg} COMMAND ${target} ${arg})
    set_tests_properties(Comp${arg}
                       PROPERTIES
                       PASS_REGULAR_EXPRESSION ${result}
                       )
endfunction()

do_test(<target> arg "result")
```

---
## 安装 & 打包 ##
```cmake
install(TARGETS <target> DESTINATION <directory>)
install(FILES <file> DESTINATION <directory>)

include(InstallRequiredSystemLibraries)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
set(CPACK_PACKAGE_VERSION_MAJOR "${<PROJECT>_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${<PROJECT>_VERSION_MINOR}")
set(CPACK_SOURCE_GENERATOR "TGZ")
include(CPack)
```
执行如下命令即可完成安装

    cmake --install . --prefix <directory>

执行如下命名完成打包

    cpack
