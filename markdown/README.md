## 1. 标题
几个`#`使用几级标题(1~6级)

---
## 2. 段落
- 紧挨着的两行会生成一段，中间有一个空格
- 两行中间空一行才能分段
- 如果第一行后面有大于1个空格，那么第二行会分段

---
## 3. 换行
如上，在第一行后追加两个空格

---
## 4. 字体
#### 1. 加粗
使用两对`*`或`_`对内容加粗
#### 2. 斜体
使用一对`*`或`_`对内容斜体
#### 3. 删除线
使用两对`~`对内容加删除线
#### 4. 粗斜体
使用三对`*`或`_`对内容粗斜体

---
## 5. 列表
#### 1. 有序列表
在列表项前加数字加点加空格可以构成有序列表
#### 2. 无序列表
在列表项前加`-`、`+`或`*`加空格可以构成无序列表
#### 3. 列表嵌套
在列表项前空2个以上空格可以将该项变成子列表
#### 4. 任务列表
在序号与项的中间加上`[x]`表示已选中或`[]`表示未选中

---
## 6. 引用
在行前加`>`和一个空格，引用可以嵌套，可搭配字体、列表等

---
## 7. 代码块
#### 1. 行内代码块
使用一对反引号`` ` `` ，如反引号本身为代码块，则使用两对反引号
#### 2. 代码块
在行前面加一个制表符
#### 3. 围栏式代码块
````
```C++
code
```
````
在代码前后加三对`或~，如果代码中也含有三个符号，则外层需要变为四对。在第一行符号后可以紧跟语言名称用以获得高亮效果(插件功能)
```C++
#include <iostream>

const int a = 10;
int sum = 0;
for (int i = 1; i < 10; i++) {
  sum += i * a;
}
std::cout << sum << std::endl;
```

---
## 8. 分隔线
一行有三个及以上`-`、`*`或`_`可生成分割线，上下最好有一行空行

---
## 9. 超链接
#### 1. 网站
`[名称](url)`
#### 2. 本地文件
`[文件名](文件相对路径)`
#### 3. 无标签链接
`<url>`
#### 4. 标题
`[名称](url "点击跳转到标题页")`

超链接可以与字体、代码块等元素搭配使用，如
这是一个粗体超链接[**标题**](./标题.txt "点击跳转")

---
## 10. 图片
在上面的超链接格式前加一个`!`  
`![图片名称](url)`  
如：  
![纳米装](https://github.com/LT-IssacF/LearnOpenGL/blob/main/image/21_assimp.png "点击跳转")

---
## 11. Emoji表情
* 将表情短码放在两个`:`之间  
如 `:joy:` :joy:  
具体的表情短码可参考[文档](https://iamhefang.cn/tutorials/markdown/Emoji%E8%A1%A8%E6%83%85 "何方的个人小站")

---
## 12. 内嵌HTML
```HTML
<form>
  <div>
    <label>账号</label><input type="text" placeholder="请输入账号">
  </div>
  <div>
    <label>密码</label><input type="text" placeholder="请输入密码">
  </div>
</form>
```
**请输入账号和密码**
<form>
  <div>
    <label>账号</label><input type="text" placeholder="请输入账号">
  </div>
  <div>
    <label>密码</label><input type="text" placeholder="请输入密码">
  </div>
</form>

---
## 13. 表格
```
|第一列|第二列|第三列|
|---|---|---|
|第一行第一列|第一行第二列|第一行第三列|
|第二行第一列|第二行第二列|第二行第三列|
```
|第一列|第二列|第三列|
|---|---|---|
|第一行第一列|第一行第二列|第一行第三列|
|第二行第一列|第二行第二列|第二行第三列|

分割行使用不同的`:`可以实现对应的**列对其**方式
* `:---`左对齐
* `:---:`居中
* `---:`右对齐

表格中也可以使用字体、行内代码、超链接等元素

---
## 14. 脚注
* 定义: `[^脚注名称]: 内容`
* 引用: `[^脚注名称]`

---
## 15. 公式
行内公式和块公式都是在公式或块前后各加一个（行内）或两个（块）`$`

`行内公式: $a = b^2 + c^2$`  
* 行内公式: $a = b^2 + c^2$

```
$$
f(x) = \int_{-\infty}^\infty\hat{f}(\xi)\,e^{2 \pi i \xi x}\,d\xi
$$
```
* 块公式[^公式来源]
$$
f(x) = \int_{-\infty}^\infty\hat{f}(\xi)\,e^{2 \pi i \xi x}\,d\xi
$$
[^公式来源]: https://iamhefang.cn/tutorials/markdown/%E5%85%AC%E5%BC%8F

---
## 16. 图表
使用和围栏式代码一样的格式，只不过语言名称使用`mermaid`，之后不同的图使用对应的关键字，[系统学习](http://mermaid.js.org/intro/)
#### 1. 类图
````
```mermaid
---
title: 类图
---
classDiagram
	note for Fruit "can buy"
	Fruit <|-- Apple
	note "I DONT LIKE PEAR"
	Fruit <|-- Pear
	Fruit <|-- Orange
	Fruit : -string kind
	Fruit : +getName()
	class Apple {
		+string name
		+wash()
	}
	class Pear {
		+string name
		+plant()
	}
	class Orange {
		+string name
		+eat()
	}
```
````
#### 2. 流程图
````
```mermaid
flowchart LR
	A[直角矩形] -->|链接文本| B(圆角矩形)
	B --> C{选择}
	C --> D[结果一]
	C --> E[结果二]
```
````

---
## 17. 注释
* `<!-- 注释内容 -->`
* `<?注释内容>`

---
## 18. 文本块
以`>`为开头